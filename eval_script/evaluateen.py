import argparse
import numpy as np
import torch
import evaluate 
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def evaluate_model(model_dir, dataset_name, dataset_config, split, max_samples):
    # Load the evaluation metric
    wer_metric = evaluate.load("wer")

    # Load processor and model
    processor = WhisperProcessor.from_pretrained(model_dir)
    model      = WhisperForConditionalGeneration.from_pretrained(model_dir)
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    sample_records = []
    sample_wers = []

    # Load a slice of the dataset
    ds = load_dataset(dataset_name, dataset_config, split=split)
    ds = ds.shuffle(seed=42).select(range(max_samples))
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    def prepare(batch):
        audio = batch["audio"]
        waveform = audio["array"]  # already 16 kHz
        inputs = processor.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
        return {
            "input_features": inputs.input_features[0],
            "reference": batch["sentence"]
        }

    ds = ds.map(prepare, remove_columns=ds.column_names, num_proc=2)
    # Iterate in batches
    batch_size = 8
    for i in range(0, len(ds), batch_size):
        batch = ds[i : i + batch_size]
        input_features = torch.tensor(batch["input_features"]).to(device)

        # Generate predictions
        with torch.no_grad():
            generated_ids = model.generate(input_features, max_length=225)
        preds = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        refs = batch["reference"]

        #wer_metric.add_batch(predictions=preds, references=refs)   #only wer

        for p, r in zip(preds, refs):
            wer_i = wer_metric.compute(predictions=[p], references=[r])
            sample_wers.append(wer_i)
            sample_records.append((r, p, wer_i))

    #wer = wer_metric.compute()                                        #only wer
    #print(f"Word Error Rate (WER) on {split} ({len(ds)} samples): {wer:.2%}")

    # overall
    all_hyps = [p for (_, p, _) in sample_records]
    all_refs = [r for (r, _, _) in sample_records]
    overall_wer = wer_metric.compute(predictions=all_hyps, references=all_refs)
    print(f"\n=== Overall WER on {split} ({len(ds)} samples): {overall_wer:.2%} ===")



    sw = np.array(sample_wers)
    print(f"Avg  per‐sample WER: {sw.mean():.2%}")
    print(f"Median per‐sample WER: {np.median(sw):.2%}")
    print(f"Max   per‐sample WER: {sw.max():.2%}\n")

    # show the 5 *worst* examples
    worst_idx = sw.argsort()[-5:][::-1]
    print("5 worst examples:")
    for idx in worst_idx:
        r, p, w = sample_records[idx]
        print(f"  ref = {r!r}")
        print(f"  hyp = {p!r}")
        print(f"  wer = {w:.2%}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Whisper model.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the trained model directory (e.g. ft_quick_tiny_en)")
    parser.add_argument("--dataset_name", type=str, default="mozilla-foundation/common_voice_11_0",
                        help="Hugging Face dataset identifier")
    parser.add_argument("--dataset_config", type=str, default="en",
                        help="Configuration name for the dataset (e.g. 'en')")
    parser.add_argument("--split", type=str, default="validation",
                        help="Data split to evaluate on")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum number of examples to evaluate (shuffle+select)")
    args = parser.parse_args()

    evaluate_model(
        model_dir=args.model_dir,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        max_samples=args.max_samples
    )
