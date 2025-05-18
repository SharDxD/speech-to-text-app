let mediaRecorder;
let audioChunks = [];
let recordingTimeout;

const recordButton = document.getElementById('recordButton');
const modelSelect   = document.getElementById("modelSelect");
const transcriptionBox = document.getElementById('transcription');
const historyList = document.querySelector('.transcript-list');

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
            
            


            const old = transcriptionBox.textContent.trim();
            if (old && old !== '...' && !old.startsWith('Error')) {
                const li = document.createElement('li');
                
                const pad2 = (n) => String(n).padStart(2, "0");
                // format a human‐readable timestamp
                const now = new Date();

                const YYYY = now.getFullYear();
                const MM   = pad2(now.getMonth() + 1);
                const DD   = pad2(now.getDate());
                const hh   = pad2(now.getHours());
                const mm   = pad2(now.getMinutes());
                const ss   = pad2(now.getSeconds());
                const ts   = `${YYYY}-${MM}-${DD} ${hh}:${mm}:${ss}`;

                // const ts = now.toLocaleString([], {
                //     year: 'numeric', month: '2-digit',
                //     day: '2-digit', hour: '2-digit',
                //     minute: '2-digit', second: '2-digit'
                // });

                    // insert both text + timestamp
                li.innerHTML = `
                    <span class="transcript-text">${old}</span>
                    <small class="transcript-time">${ts}</small>
                `;
                historyList.insertBefore(li, historyList.firstChild);
            }

                // 2) Build your blob and do the fetch…
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append("audio", audioBlob, "recording.webm");
            formData.append("model", document.getElementById('modelSelect').value);
            const url = URL.createObjectURL(audioBlob);
            audioPlayer.src = url;
            audioPlayer.controls = true;
            audioWrapper.style.display = 'block';


            try {
                const response = await fetch("/transcribe", { method: "POST", body: formData });
                const result   = await response.json();
                transcriptionBox.textContent = result.transcription;
            } catch (err) {
                transcriptionBox.textContent = "Fetch error: " + err;
            }

        });
    }
});

function stopRecording() {
    clearTimeout(recordingTimeout);
    mediaRecorder.stop();
    recordButton.textContent = "Start Recording";
}