// ======================
// 全局状态
// ======================
let stream = null;           // MediaStream 引用
let autoCaptureTimer = null; // setInterval ID
let autoCaptureActive = false;
let capturing = false;       // 防抖标志
const AUTO_CAPTURE_INTERVAL = 3000; // 自动识别间隔（毫秒）

// ======================
// 模式切换
// ======================
function switchMode(mode) {
    const uploadSection = document.getElementById('uploadSection');
    const cameraSection = document.getElementById('cameraSection');
    const btnUpload = document.getElementById('btnUploadMode');
    const btnCamera = document.getElementById('btnCameraMode');

    if (mode === 'upload') {
        uploadSection.style.display = 'block';
        cameraSection.style.display = 'none';
        btnUpload.classList.add('active');
        btnCamera.classList.remove('active');
        stopCamera();
    } else {
        uploadSection.style.display = 'none';
        cameraSection.style.display = 'block';
        btnCamera.classList.add('active');
        btnUpload.classList.remove('active');
        startCamera();
    }
}

// ======================
// 启动摄像头
// ======================
async function startCamera() {
    const statusEl = document.getElementById('cameraStatus');
    const video = document.getElementById('video');

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'environment'  // 优先后置摄像头
            },
            audio: false
        });
        video.srcObject = stream;
        statusEl.innerText = '摄像头已就绪 — 请对准花卉后点击拍照';
        statusEl.style.color = '#4caf50';
    } catch (err) {
        statusEl.innerText = '摄像头权限被拒绝或不可用: ' + err.message;
        statusEl.style.color = '#d32f2f';
        // 回退到上传模式
        setTimeout(() => switchMode('upload'), 2000);
    }
}

// ======================
// 停止摄像头
// ======================
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    stopAutoCapture();
    const video = document.getElementById('video');
    if (video) video.srcObject = null;
    const statusEl = document.getElementById('cameraStatus');
    if (statusEl) statusEl.innerText = '';
}

// ======================
// 捕获单帧并识别
// ======================
async function captureFrame() {
    // 防抖：如果正在识别中则跳过
    if (capturing) return;
    capturing = true;

    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const statusEl = document.getElementById('cameraStatus');

    if (!video.srcObject) {
        statusEl.innerText = '请先打开摄像头';
        statusEl.style.color = '#d32f2f';
        capturing = false;
        return;
    }

    try {
        // --- 步骤1：从video绘制到canvas，中心裁剪正方形 → 224×224 ---
        const ctx = canvas.getContext('2d');
        const size = Math.min(video.videoWidth, video.videoHeight);
        const sx = (video.videoWidth - size) / 2;
        const sy = (video.videoHeight - size) / 2;
        ctx.drawImage(video, sx, sy, size, size, 0, 0, 224, 224);

        // --- 步骤2：canvas → blob → base64 data URL ---
        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg', 0.9);
        });
        const base64 = await blobToBase64(blob);

        // --- 步骤3：发送到后端 ---
        statusEl.innerHTML = '<span class="spinner"></span>识别中...';
        statusEl.style.color = '#666';

        const response = await fetch('/predict_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64 })
        });

        const data = await response.json();

        if (data.success) {
            displayResult(data.result, data.confidence, data.care);
            statusEl.innerText = '识别完成 ✓';
            statusEl.style.color = '#4caf50';
        } else {
            statusEl.innerText = '识别失败: ' + (data.error || '未知错误');
            statusEl.style.color = '#d32f2f';
        }
    } catch (err) {
        statusEl.innerText = '请求失败: ' + err.message;
        statusEl.style.color = '#d32f2f';
    } finally {
        capturing = false;
    }
}

// ======================
// 自动识别 开/关
// ======================
function toggleAutoCapture() {
    if (autoCaptureActive) {
        stopAutoCapture();
    } else {
        startAutoCapture();
    }
}

function startAutoCapture() {
    autoCaptureActive = true;
    document.getElementById('btnAutoCapture').innerText = '⏸ 停止自动识别';
    // 立即执行一次
    captureFrame();
    autoCaptureTimer = setInterval(captureFrame, AUTO_CAPTURE_INTERVAL);
}

function stopAutoCapture() {
    autoCaptureActive = false;
    if (autoCaptureTimer) {
        clearInterval(autoCaptureTimer);
        autoCaptureTimer = null;
    }
    const btn = document.getElementById('btnAutoCapture');
    if (btn) btn.innerText = '🔄 自动识别';
}

// ======================
// 工具函数：Blob → base64 data URL
// ======================
function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

// ======================
// 动态显示识别结果（无需刷新页面）
// ======================
function displayResult(plantName, confidence, care) {
    let card = document.querySelector('.result-card');
    if (!card) {
        card = document.createElement('div');
        card.className = 'result-card';
        const container = document.querySelector('.container');
        const footer = document.querySelector('.footer');
        container.insertBefore(card, footer);
    }

    card.innerHTML = `
        <h2 class="result-title">识别结果</h2>
        <div class="plant-name">${plantName}</div>
        <div class="confidence">识别置信度：${confidence}%</div>
        <div class="care-box">
            <strong>养护建议：</strong><br>
            ${care}
        </div>
    `;
    card.style.display = 'block';
}

// ======================
// 页面关闭时释放摄像头
// ======================
window.addEventListener('beforeunload', () => {
    stopCamera();
});
