const video = document.getElementById("video");
const scoreEl = document.getElementById("score");
const statusEl = document.getElementById("status");
const emotionEl = document.getElementById("emotion");
const componentsEl = document.getElementById("components");
const heatmapImg = document.getElementById("heatmap");

// Camera
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => video.srcObject = stream);

// Chart
const ctx = document.getElementById('chart').getContext('2d');

const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Anxiety Score',
            data: [],
            borderColor: '#38BDF8',
            tension: 0.4
        }]
    },
    options: {
        animation: false,
        scales: {
            y: { min: 0, max: 100 }
        }
    }
});

// Loop
setInterval(sendFrame, 1000);

function sendFrame() {
    if (!video.videoWidth) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    const base64Image = canvas.toDataURL("image/jpeg").split(',')[1];

    fetch("http://127.0.0.1:8000/analyse/stream", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ frame_b64: base64Image })
    })
    .then(res => res.json())
    .then(updateUI);
}

function updateUI(data) {

    // Score
    scoreEl.innerText = data.anxiety_score.toFixed(1);

    // Status
    let level = data.level.toLowerCase();

    if (level.includes("high")) {
        statusEl.innerText = "High Anxiety";
        statusEl.className = "anxious";
    } else if (level.includes("moderate")) {
        statusEl.innerText = "Moderate Anxiety";
        statusEl.className = "anxious";
    } else {
        statusEl.innerText = "Normal";
        statusEl.className = "normal";
    }

    // Emotion
    emotionEl.innerText = data.emotion + " (" + data.emotion_confidence + "%)";

    // Components breakdown
    componentsEl.innerHTML = "";
    for (let key in data.components) {
        let li = document.createElement("li");
        li.innerText = key + ": " + data.components[key];
        componentsEl.appendChild(li);
    }

    // Heatmap overlay
    heatmapImg.src = "data:image/jpeg;base64," + data.heatmap_b64;

    // Graph
    chart.data.labels.push('');
    chart.data.datasets[0].data.push(data.anxiety_score);

    if (chart.data.labels.length > 25) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }

    chart.update();
}