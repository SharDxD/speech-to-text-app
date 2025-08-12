# speech-to-text-app
Web Application for Bilingual Speech Recognition Deployed in the Cloud

Description:
Modern voice-to-text services are powerful but often locked behind closed APIs, making integration difficult or costly. Some existing solutions don't support multiple languages, some have no offline capabilities, others have payment strategies that are basically not suitable for small projects. I wanted to create a solution that would run anywhere, require no external calls, and allow full control over the data and models.
I chose OpenAI’s Whisper model as a starting point.
To adapt Whisper for my needs, I used the smallest “tiny” variant, which has 39 million parameters, small enough to train quickly on a single GPU I have, yet large enough to capture speech patterns. 
As for the audio data source, I utilized Hugging Face’s datasets library to stream and cache Mozilla’s Common Voice. And some extra customizing was done to extend the official finetune script presented by openAI.
I optimized the finetuning script as much as possible.
I have added flags to toggle between streaming and full download to shorten the data gathering step in case you don't need the whole hundreds of gigabytes dataset and you are only working with a small slice of it.
The whole mapping step was encapsulated to enable the parallelized data preparation where you can choose the number of parallel processes (this feature wasn’t working from the box for some reason, hence I needed to implement it again), and I have integrated “accelerate” for GPU compatibility, by installing cuda and other drivers allowing finetuning to be run on my gpu (which led to iteration taking approximately 6-7 sec instead of 30+ for 1 iteration on cpu).
This pipeline allowed me to run experiments across three languages - Hungarian, English and Russian - in under two hours each.\
To evaluate my finetuning, I wrote a small evaluation script that loads my tuned model, selects a random slice of 1 000 validation samples, and computes the Word Error Rate (WER for short) metric. To note here, I just estimated that the WER should be comparable for all three languages, hence I run the evaluation script only on the English model.
Rather than comparing all predictions at once, I compute per-sample WER iteratively and then aggregate statistics: average, median, and worst cases.

Web-app:
1) Flask+MongoDB+(HTMSL, CSS, JS)+Gunicorn+Nginx //or MicroStack as an alternative
