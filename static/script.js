let mediaRecorder;
let audioChunks = [];
let recordingTimeout;

const recordButton = document.getElementById('recordButton');
const transcriptionBox = document.getElementById('transcription');

const audioPlayer = document.querySelector(".dark-player");
const audioWrapper = document.querySelector(".audio-wrapper");
recordButton.classList.add('idle');

recordButton.addEventListener('click', async () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        // to stop
        stopRecording();
        recordButton.textContent = "Start Recording";
        recordButton.classList.remove('recording');
        recordButton.classList.add('idle');
    } else {
        // to stop
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        audioChunks = [];
        mediaRecorder.start();
        recordButton.textContent = "Stop Recording";
        recordButton.classList.remove('idle');
        recordButton.classList.add('recording');
    
        // auto‑stop after 30s of rec
        recordingTimeout = setTimeout(() => {
          stopRecording();
          recordButton.textContent = "Start Recording";
          recordButton.classList.remove('recording');
          recordButton.classList.add('idle');
        }, 30000);

        mediaRecorder.addEventListener("dataavailable", event => {
            audioChunks.push(event.data);
        });

        mediaRecorder.addEventListener("stop", async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const url = URL.createObjectURL(audioBlob);

            audioPlayer.src = url;
            audioWrapper.style.display = 'block';

            const formData = new FormData();
            formData.append("audio", audioBlob, "recording.webm");

            const response = await fetch('/transcribe', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            transcriptionBox.textContent = result.transcription;
        });
    }
});

function stopRecording() {
    clearTimeout(recordingTimeout);
    mediaRecorder.stop();
    recordButton.textContent = "Start Recording";
}