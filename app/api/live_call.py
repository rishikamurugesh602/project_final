from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/api/v1", tags=["live-call"])

LIVE_CALL_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Live Call Monitor</title>
  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
      color: white;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .panel {
      width: min(980px, 92vw);
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.35);
      border-radius: 18px;
      padding: 28px;
      box-shadow: 0 20px 50px rgba(0,0,0,0.35);
    }
    h1 {
      margin-top: 0;
      font-size: 2.1rem;
    }
    .row {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 20px;
      margin-top: 18px;
    }
    .callbox, .stats {
      background: rgba(30, 41, 59, 0.9);
      border-radius: 16px;
      padding: 22px;
      border: 1px solid rgba(148, 163, 184, 0.3);
    }
    .status-pill {
      display: inline-block;
      background: #22c55e;
      color: #06210f;
      font-weight: 700;
      border-radius: 999px;
      padding: 8px 14px;
      margin-bottom: 10px;
      font-size: 12px;
      letter-spacing: 1px;
      text-transform: uppercase;
    }
    #callerName {
      font-size: 2rem;
      margin: 8px 0;
    }
    .alert {
      background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
      color: white;
      padding: 18px 20px;
      border-radius: 12px;
      font-weight: 700;
      margin-top: 12px;
      display: none;
    }
    .alert.show {
      display: block;
    }
    .controls {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 18px;
    }
    button {
      background: linear-gradient(135deg, #22c55e 0%, #15803d 100%);
      color: white;
      border: none;
      border-radius: 10px;
      padding: 12px 18px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
    }
    button.stop {
      background: linear-gradient(135deg, #ef4444 0%, #991b1b 100%);
    }
    button.secondary {
      background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    }
    .metric {
      padding: 10px 0;
      border-bottom: 1px solid rgba(148, 163, 184, 0.25);
    }
    .metric:last-child {
      border-bottom: none;
    }
    .label {
      font-size: 12px;
      color: #93c5fd;
      letter-spacing: 1px;
      text-transform: uppercase;
    }
    .value {
      font-size: 1.7rem;
      font-weight: 700;
      margin-top: 4px;
    }
    .log {
      margin-top: 18px;
      background: rgba(17, 24, 39, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 12px;
      padding: 14px;
      min-height: 120px;
      color: #dbeafe;
    }
    @media (max-width: 760px) {
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="panel">
    <h1>Real-Time Deepfake Voice Interceptor</h1>

    <div class="row">
      <div class="callbox">
        <div class="status-pill" id="callStatus">Waiting</div>
        <div id="callerName">Unknown Caller</div>
        <div>Call ID: LIVE-VOICE-01</div>
        <div class="controls">
          <button id="startBtn">Start Live Call</button>
          <button id="stopBtn" class="stop">Stop</button>
        </div>
        <div class="alert" id="alertBox">ALERT: HIGH RISK DETECTED</div>
      </div>

      <div class="stats">
        <div class="metric">
          <div class="label">Deepfake Risk</div>
          <div class="value" id="deepfakeValue">--</div>
        </div>
        <div class="metric">
          <div class="label">Scam Intent</div>
          <div class="value" id="scamValue">--</div>
        </div>
        <div class="metric">
          <div class="label">Overall Risk</div>
          <div class="value" id="riskValue">--</div>
        </div>
      </div>
    </div>

    <div class="log" id="logBox">
      System idle. Press Start Live Call to begin microphone monitoring.
    </div>
  </div>

  <script>
    const wsUrl = 'ws://' + location.hostname + ':8000/api/v1/ws/analyze';
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const alertBox = document.getElementById('alertBox');
    const logBox = document.getElementById('logBox');
    const deepfakeValue = document.getElementById('deepfakeValue');
    const scamValue = document.getElementById('scamValue');
    const riskValue = document.getElementById('riskValue');
    const callStatus = document.getElementById('callStatus');

    let socket = null;
    let mediaStream = null;
    let audioContext = null;
    let processor = null;
    let source = null;
    let liveSession = false;

    function log(message) {
      const time = new Date().toLocaleTimeString();
      logBox.innerHTML = '[' + time + '] ' + message + '<br>' + logBox.innerHTML;
    }

    function updateRiskDisplay(data) {
      if (!data) return;

      const deepfakeProbability = Number(
        (data.deepfake && data.deepfake.deepfake_probability) ??
        data.deepfake_probability ??
        0
      );

      const scamProbability = Number(
        (data.scam && data.scam.scam_probability) ??
        data.scam_probability ??
        0
      );

      const riskScore = Number(
        (data.risk && data.risk.overall_risk_score) ??
        data.overall_risk_score ??
        0
      );

      const riskCategory = (data.risk && data.risk.risk_category) || data.risk_category || 'LOW';

      const deepfake = deepfakeProbability * 100;
      const scam = scamProbability * 100;
      const risk = riskScore * 100;
      deepfakeValue.textContent = deepfake.toFixed(1) + '%';
      scamValue.textContent = scam.toFixed(1) + '%';
      riskValue.textContent = risk.toFixed(1) + '%';

      const level = String(riskCategory).toUpperCase();
      const message = 'ALERT: ' + level + ' RISK DETECTED';
      if (level === 'HIGH') {
        alertBox.classList.add('show');
        alertBox.textContent = message;
      } else if (level === 'MEDIUM') {
        alertBox.classList.add('show');
        alertBox.textContent = 'WARNING: MODERATE RISK DETECTED';
      } else {
        alertBox.classList.remove('show');
      }

      log('Received risk result: ' + level + ' (' + risk.toFixed(1) + '%)');
    }

    function connectSocket() {
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        log('WebSocket connected to backend');
        callStatus.textContent = 'Monitoring';
        callStatus.style.background = '#fbbf24';
        callStatus.style.color = '#1f2937';
      };

      socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        log('Received message from backend: ' + msg.type);

        if (msg.type === 'status') {
          log(msg.data.message || 'Status update');
        }

        if (msg.type === 'result') {
          updateRiskDisplay(msg.data);
        }

        if (msg.type === 'error') {
          log('Backend error: ' + (msg.data && msg.data.error ? msg.data.error : 'Unknown error'));
        }
      };

      socket.onclose = () => {
        log('WebSocket disconnected');
        if (liveSession) {
          callStatus.textContent = 'Disconnected';
          callStatus.style.background = '#ef4444';
          callStatus.style.color = 'white';
        }
      };
    }

    async function startLiveCall() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Microphone access is not supported in this browser.');
        return;
      }

      if (!socket || socket.readyState !== WebSocket.OPEN) {
        connectSocket();
      }

      const waitForSocket = () => new Promise((resolve, reject) => {
        const started = Date.now();
        const check = () => {
          if (socket && socket.readyState === WebSocket.OPEN) {
            resolve();
            return;
          }
          if (Date.now() - started > 5000) {
            reject(new Error('WebSocket did not connect in time'));
            return;
          }
          setTimeout(check, 100);
        };
        check();
      });

      try {
        log('Requesting microphone access...');
        await waitForSocket();

        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: 16000,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        });

        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        await audioContext.resume();

        source = audioContext.createMediaStreamSource(mediaStream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);

        processor.onaudioprocess = (event) => {
          if (!socket || socket.readyState !== WebSocket.OPEN) return;
          const input = event.inputBuffer.getChannelData(0);
          const float32Array = new Float32Array(input.length);
          for (let i = 0; i < input.length; i++) {
            float32Array[i] = input[i];
          }

          const bytes = new ArrayBuffer(float32Array.length * 4);
          const view = new DataView(bytes);
          for (let i = 0; i < float32Array.length; i++) {
            view.setFloat32(i * 4, float32Array[i], true);
          }

          let binary = '';
          const uint8 = new Uint8Array(bytes);
          for (let i = 0; i < uint8.length; i++) {
            binary += String.fromCharCode(uint8[i]);
          }

          const packet = {
            type: 'audio_chunk',
            data: btoa(binary),
            sample_rate: 16000,
            is_final: false
          };

          socket.send(JSON.stringify(packet));
        };

        source.connect(processor);
        processor.connect(audioContext.destination);

        liveSession = true;
        callStatus.textContent = 'Monitoring';
        callStatus.style.background = '#22c55e';
        callStatus.style.color = '#06210f';
        log('Microphone live. Audio is now being monitored.');
      } catch (error) {
        log('Microphone access error: ' + error.message);
      }
    }

    function stopLiveCall() {
      liveSession = false;
      callStatus.textContent = 'Stopped';
      callStatus.style.background = '#ef4444';
      callStatus.style.color = 'white';
      log('Stopping live call monitor.');

      if (processor) {
        processor.disconnect();
      }
      if (source) {
        source.disconnect();
      }
      if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
      }
      if (audioContext) {
        audioContext.close();
      }
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'reset' }));
        socket.close();
      }
    }

    startBtn.addEventListener('click', startLiveCall);
    stopBtn.addEventListener('click', stopLiveCall);
  </script>
</body>
</html>
"""


@router.get("/live-call")
async def live_call_page():
    return HTMLResponse(LIVE_CALL_HTML)
