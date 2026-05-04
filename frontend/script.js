const video = document.getElementById("video");
const scoreEl = document.getElementById("score");
const statusEl = document.getElementById("status");

// Start camera
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    });

// Chart setup
const ctx = document.getElementById('chart').getContext('2d');
let dataPoints = [];

const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Anxiety Score',
            data: [],
            borderColor: '#38BDF8',
            tension: 0.3
        }]
    },
    options: {
        scales: {
            y: {
                min: 0,
                max: 1
            }
        }
    }
});

// Capture frame and send to backend
setInterval(() => {
    captureAndSend();
}, 2000); // every 2 sec

function captureAndSend() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(blob => {
        let formData = new FormData();
        formData.append("frame", blob);

        fetch("http://localhost:5000/predict", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            updateUI(data);
        });
    });
}

function updateUI(data) {
    let score = data.anxiety_score;

    scoreEl.innerText = score.toFixed(2);

    if (score > 0.6) {
        statusEl.innerText = "Anxious";
        statusEl.className = "anxious";
    } else {
        statusEl.innerText = "Normal";
        statusEl.className = "normal";
    }

    // Update graph
    chart.data.labels.push('');
    chart.data.datasets[0].data.push(score);

    if (chart.data.labels.length > 20) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }

    chart.update();
}