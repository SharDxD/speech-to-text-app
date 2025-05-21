from transformers import pipeline

# point this at wherever you saved your checkpoint
model_dir = "C:/Users/kirig/Documents/model_0/whisper-finetune/ft_quick_tiny"  

# create an ASR pipeline on GPU
stt = pipeline(
    "automatic-speech-recognition",
    model=model_dir,
    device=0,            # 0 for cuda:0
    chunk_length_s=30,     # whisper models like <=30s segments
    
)
# run it on a file
result = stt("C:/Users/kirig/Documents/thesis/uploads/common_voice_ru_18849869.mp3")  
print("Transcription:", result["text"])