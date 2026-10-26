"""
Live call management and WebRTC signaling router.
Provides browser-to-browser WebRTC call sessions, mutual consent tracking,
remote audio track analysis, and local microphone test mode.
"""
import uuid
import json
import random
import string
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from app.services.call_manager import call_session_manager
from app.utils.logging import logger

router = APIRouter(prefix="/api/v1", tags=["live-call"])


@router.post("/call/session/create")
async def create_call_session():
    """Create a new WebRTC call session ID."""
    call_id = "CALL-" + "".join(random.choices(string.digits, k=4))
    session = call_session_manager.get_or_create_session(call_id)
    return {
        "call_id": call_id,
        "status": "created",
        "message": f"Session {call_id} ready. Share this ID with the callee."
    }


@router.get("/call/session/{call_id}")
async def get_call_session_status(call_id: str):
    """Retrieve participant and consent status for a call session."""
    session = call_session_manager.get_or_create_session(call_id)
    return session.get_status()


@router.websocket("/call/ws/{call_id}/{peer_id}")
async def call_signaling_websocket(websocket: WebSocket, call_id: str, peer_id: str):
    """
    WebSocket signaling endpoint for WebRTC offer/answer, ICE candidates, and consent.
    """
    await websocket.accept()
    session = call_session_manager.get_or_create_session(call_id)
    participant = session.add_participant(peer_id, websocket)

    logger.info(f"Peer {peer_id} joined call {call_id} as {participant.role}")

    # Notify caller/callee of joined peer
    other_peer = session.get_other_participant(peer_id)
    await websocket.send_json({
        "type": "session_info",
        "role": participant.role,
        "call_id": call_id,
        "peer_id": peer_id,
        "peer_connected": other_peer is not None,
        "both_consented": session.both_consented()
    })

    if other_peer and other_peer.websocket:
        try:
            await other_peer.websocket.send_json({
                "type": "peer_joined",
                "peer_id": peer_id,
                "role": participant.role
            })
        except Exception:
            pass

    try:
        while True:
            text = await websocket.receive_text()
            data = json.loads(text)
            msg_type = data.get("type")

            if msg_type == "consent":
                consent_val = bool(data.get("consent", False))
                session.set_consent(peer_id, consent_val)
                both = session.both_consented()

                # Broadcast consent update to all participants in room
                for p in session.participants.values():
                    try:
                        await p.websocket.send_json({
                            "type": "consent_update",
                            "peer_id": peer_id,
                            "consent": consent_val,
                            "both_consented": both
                        })
                    except Exception:
                        pass

            elif msg_type in ("offer", "answer", "ice_candidate", "hangup"):
                # Forward WebRTC signaling to other peer
                other = session.get_other_participant(peer_id)
                if other and other.websocket:
                    try:
                        await other.websocket.send_json({
                            "type": msg_type,
                            "sender_id": peer_id,
                            "payload": data.get("payload")
                        })
                    except Exception as e:
                        logger.warning(f"Error forwarding signal to peer: {e}")

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"Peer {peer_id} disconnected from call {call_id}")
    except Exception as e:
        logger.error(f"Signaling error for {peer_id}: {e}")
    finally:
        session.remove_participant(peer_id)
        other = session.get_other_participant(peer_id)
        if other and other.websocket:
            try:
                await other.websocket.send_json({
                    "type": "peer_left",
                    "peer_id": peer_id
                })
            except Exception:
                pass
        call_session_manager.remove_session_if_empty(call_id)


LIVE_CALL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Real-Time Deepfake Voice Interceptor | Live WebRTC Call</title>
  <style>
    :root {
      --bg: #07111f;
      --card-bg: rgba(15, 23, 42, 0.92);
      --card-border: rgba(148, 163, 184, 0.25);
      --accent: #2563eb;
      --accent-hover: #1d4ed8;
      --text: #edf4ff;
      --text-muted: #94a3b8;
      --green: #22c55e;
      --red: #ef4444;
      --amber: #f59e0b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: linear-gradient(135deg, #07111f 0%, #0d1b2a 100%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      padding: 24px 16px;
    }
    .container {
      width: min(1080px, 98vw);
    }
    .header {
      background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
      padding: 24px;
      border-radius: 16px;
      border: 1px solid var(--card-border);
      margin-bottom: 20px;
      text-align: center;
    }
    .header h1 {
      margin: 0 0 8px 0;
      font-size: 1.8rem;
    }
    .header p {
      margin: 0;
      color: #bfdbfe;
      font-size: 0.95rem;
    }
    .disclaimer-banner {
      background: rgba(245, 158, 11, 0.12);
      border: 1px solid rgba(245, 158, 11, 0.35);
      border-radius: 12px;
      padding: 14px 18px;
      margin-bottom: 20px;
      font-size: 0.88rem;
      line-height: 1.45;
      color: #fde68a;
    }
    .tabs {
      display: flex;
      gap: 10px;
      margin-bottom: 18px;
    }
    .tab-btn {
      padding: 12px 20px;
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      color: var(--text-muted);
      cursor: pointer;
      font-weight: 600;
      font-size: 0.95rem;
      transition: all 0.2s;
    }
    .tab-btn.active {
      background: linear-gradient(135deg, #2563eb, #1d4ed8);
      color: white;
      border-color: #3b82f6;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.25fr 0.95fr;
      gap: 20px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 22px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .card h2 {
      margin-top: 0;
      font-size: 1.25rem;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 10px;
    }
    .pill {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      margin-bottom: 12px;
    }
    .pill-idle { background: #334155; color: #cbd5e1; }
    .pill-calling { background: #d97706; color: white; }
    .pill-connected { background: #16a34a; color: white; }
    .pill-error { background: #dc2626; color: white; }
    
    .consent-box {
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(59, 130, 246, 0.3);
      border-radius: 12px;
      padding: 16px;
      margin: 16px 0;
    }
    .consent-box label {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      font-size: 0.88rem;
      color: #e2e8f0;
      cursor: pointer;
    }
    .consent-box input[type="checkbox"] {
      margin-top: 3px;
      transform: scale(1.2);
    }
    .btn-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }
    button.action-btn {
      background: linear-gradient(135deg, #2563eb, #1d4ed8);
      color: white;
      border: none;
      padding: 10px 18px;
      border-radius: 10px;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      transition: filter 0.15s;
    }
    button.action-btn:hover { filter: brightness(1.1); }
    button.action-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    button.hangup-btn {
      background: linear-gradient(135deg, #dc2626, #991b1b);
    }
    button.secondary-btn {
      background: rgba(51, 65, 85, 0.8);
    }
    input.text-input {
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      color: white;
      padding: 8px 12px;
      font-size: 0.95rem;
      width: 160px;
    }
    .metric-row {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      padding: 10px 0;
      border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }
    .metric-row:last-child { border-bottom: none; }
    .metric-label { font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 1.4rem; font-weight: 700; }
    .alert-banner {
      background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
      color: white;
      padding: 14px 18px;
      border-radius: 10px;
      font-weight: 700;
      margin-top: 14px;
      display: none;
      text-align: center;
    }
    .alert-banner.show { display: block; }
    .log-box {
      margin-top: 18px;
      background: #0a0f1d;
      border: 1px solid rgba(148, 163, 184, 0.15);
      border-radius: 10px;
      padding: 12px;
      font-family: monospace;
      font-size: 0.82rem;
      color: #93c5fd;
      max-height: 160px;
      overflow-y: auto;
    }
    .model-badge {
      font-size: 0.75rem;
      padding: 4px 8px;
      border-radius: 6px;
      background: rgba(245, 158, 11, 0.2);
      color: #fde68a;
      border: 1px solid rgba(245, 158, 11, 0.4);
      display: inline-block;
      margin-top: 6px;
    }
    @media (max-width: 820px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🎙️ Real-Time Deepfake Voice Interceptor</h1>
      <p>Browser-to-Browser WebRTC Audio Call & Remote Track Deepfake Analysis</p>
    </div>

    <div class="disclaimer-banner">
      <strong>⚠️ Telephony / PSTN Limitation Notice:</strong> This module performs in-app, browser-to-browser voice calling via WebRTC.
      It intercepts and analyzes the <em>received remote WebRTC audio track</em>. It does <strong>not</strong> intercept standard cellular (GSM/LTE) or landline (PSTN) telephone calls, which strictly requires telecommunication provider integration (SIP trunking / PBX gateway) and carrier credentials that are not present in this academic prototype.
    </div>

    <div class="tabs">
      <button class="tab-btn active" id="tabWebRTCBtn">Browser-to-Browser WebRTC Call</button>
      <button class="tab-btn" id="tabLocalBtn">Local Microphone Test Mode</button>
    </div>

    <div class="grid">
      <!-- Left Column: Controls & Call State -->
      <div class="card">
        <div id="webrtcControls">
          <div class="pill pill-idle" id="webrtcStatusPill">Idle</div>
          <h2>WebRTC Call Session</h2>
          <p style="font-size: 0.88rem; color: #94a3b8;">
            To test a live 2-party call: Open this page in two separate browser windows (or tabs). Create a session in Window A, copy the Call ID, and Join in Window B.
          </p>

          <div style="margin: 14px 0;">
            <label style="font-size: 0.85rem; color: #94a3b8;">Call Session ID:</label><br />
            <input type="text" id="callIdInput" class="text-input" placeholder="e.g. CALL-1234" />
            <button class="action-btn secondary-btn" id="createCallBtn">Create New Call</button>
            <button class="action-btn" id="joinCallBtn">Join Call</button>
            <button class="action-btn hangup-btn" id="hangupBtn" style="display:none;">Leave Call</button>
          </div>

          <div class="consent-box">
            <strong style="color: #60a5fa; font-size: 0.9rem;">Mutual Participant Consent</strong>
            <p style="font-size: 0.82rem; color: #cbd5e1; margin: 6px 0 10px 0;">
              Both callers must grant explicit consent before audio transmission and deepfake analysis are enabled.
            </p>
            <label>
              <input type="checkbox" id="consentCheckbox" />
              <span>I consent to live audio transmission and real-time deepfake/scam analysis for this call.</span>
            </label>
            <div id="consentStatusText" style="font-size: 0.8rem; color: #f59e0b; margin-top: 8px;">
              Your consent: Pending | Remote consent: Pending
            </div>
          </div>

          <div style="font-size: 0.85rem; color: #cbd5e1;">
            <div>Target Track: <strong id="analyzedTrackLabel">None</strong></div>
            <div>Latency (End-to-End): <strong id="latencyDisplay">-- ms</strong></div>
            <div>Processing State: <strong id="processingStateDisplay">Idle</strong></div>
          </div>
        </div>

        <!-- Local Test Mode Controls -->
        <div id="localControls" style="display: none;">
          <div class="pill pill-idle" id="localStatusPill">Idle</div>
          <h2>Local Microphone Test Mode</h2>
          <p style="font-size: 0.88rem; color: #94a3b8;">
            Single-client test mode. Captures your local microphone and sends bounded chunks to the analysis engine to test acoustic feature extraction without needing a second participant.
          </p>
          <div class="consent-box">
            <label>
              <input type="checkbox" id="localConsentCheckbox" />
              <span>I consent to microphone capture and automated analysis for testing.</span>
            </label>
          </div>
          <div class="btn-row">
            <button class="action-btn" id="startLocalBtn">Start Local Test</button>
            <button class="action-btn hangup-btn" id="stopLocalBtn" style="display:none;">Stop Test</button>
          </div>
        </div>

        <div class="alert-banner" id="alertBox">HIGH RISK SCAM / DEEPFAKE DETECTED</div>

        <audio id="remoteAudio" autoplay playsinline></audio>
      </div>

      <!-- Right Column: Live Analysis Metrics -->
      <div class="card">
        <h2>Live Authenticity & Risk Metrics</h2>
        
        <div class="metric-row">
          <div>
            <div class="metric-label">Deepfake Probability</div>
            <div class="model-badge" id="deepfakeStatusBadge">Checking model status...</div>
          </div>
          <div class="metric-value" id="deepfakeVal">--</div>
        </div>

        <div class="metric-row">
          <div>
            <div class="metric-label">Scam Intent</div>
            <div class="model-badge" id="scamStatusBadge">Rule-based / NLP</div>
          </div>
          <div class="metric-value" id="scamVal">--</div>
        </div>

        <div class="metric-row">
          <div class="metric-label">Overall Risk Score</div>
          <div class="metric-value" id="overallRiskVal">--</div>
        </div>

        <div class="metric-row">
          <div class="metric-label">Risk Category</div>
          <div class="metric-value" id="riskCategoryVal" style="font-size: 1.1rem; color: #94a3b8;">--</div>
        </div>

        <div style="margin-top: 14px;">
          <div class="metric-label">Transcript</div>
          <div id="transcriptBox" style="background:#0f172a; padding:10px; border-radius:8px; font-size:0.85rem; min-height:50px; color:#e2e8f0; margin-top:6px;">
            (Transcript will appear here as speech is processed)
          </div>
        </div>

        <div style="margin-top: 14px;">
          <div class="metric-label">Recommendation</div>
          <div id="recommendationBox" style="font-size:0.85rem; color:#93c5fd; margin-top:4px;">
            Awaiting call start...
          </div>
        </div>

        <div class="log-box" id="logBox">
          [System] Live Call Monitor initialized. Ready.
        </div>
      </div>
    </div>
  </div>

  <script>
    // State variables
    let mode = 'webrtc'; // 'webrtc' or 'local'
    let peerId = 'peer_' + Math.random().toString(36).substring(2, 9);
    let currentCallId = null;
    let signalingSocket = null;
    let analysisSocket = null;
    let peerConnection = null;
    let localStream = null;
    let remoteStream = null;
    let audioCtx = null;
    let audioSource = null;
    let audioProcessor = null;
    let bothConsented = false;
    let isCaller = false;

    // DOM Elements
    const tabWebRTCBtn = document.getElementById('tabWebRTCBtn');
    const tabLocalBtn = document.getElementById('tabLocalBtn');
    const webrtcControls = document.getElementById('webrtcControls');
    const localControls = document.getElementById('localControls');
    const callIdInput = document.getElementById('callIdInput');
    const createCallBtn = document.getElementById('createCallBtn');
    const joinCallBtn = document.getElementById('joinCallBtn');
    const hangupBtn = document.getElementById('hangupBtn');
    const consentCheckbox = document.getElementById('consentCheckbox');
    const consentStatusText = document.getElementById('consentStatusText');
    const startLocalBtn = document.getElementById('startLocalBtn');
    const stopLocalBtn = document.getElementById('stopLocalBtn');
    const localConsentCheckbox = document.getElementById('localConsentCheckbox');
    const webrtcStatusPill = document.getElementById('webrtcStatusPill');
    const localStatusPill = document.getElementById('localStatusPill');
    const analyzedTrackLabel = document.getElementById('analyzedTrackLabel');
    const latencyDisplay = document.getElementById('latencyDisplay');
    const processingStateDisplay = document.getElementById('processingStateDisplay');
    const deepfakeVal = document.getElementById('deepfakeVal');
    const scamVal = document.getElementById('scamVal');
    const overallRiskVal = document.getElementById('overallRiskVal');
    const riskCategoryVal = document.getElementById('riskCategoryVal');
    const transcriptBox = document.getElementById('transcriptBox');
    const recommendationBox = document.getElementById('recommendationBox');
    const alertBox = document.getElementById('alertBox');
    const logBox = document.getElementById('logBox');
    const remoteAudio = document.getElementById('remoteAudio');
    const deepfakeStatusBadge = document.getElementById('deepfakeStatusBadge');
    const scamStatusBadge = document.getElementById('scamStatusBadge');

    function log(msg) {
      const ts = new Date().toLocaleTimeString();
      logBox.innerHTML = `[${ts}] ${msg}<br/>` + logBox.innerHTML;
    }

    // Tab switching
    tabWebRTCBtn.onclick = () => {
      mode = 'webrtc';
      tabWebRTCBtn.classList.add('active');
      tabLocalBtn.classList.remove('active');
      webrtcControls.style.display = 'block';
      localControls.style.display = 'none';
    };
    tabLocalBtn.onclick = () => {
      mode = 'local';
      tabLocalBtn.classList.add('active');
      tabWebRTCBtn.classList.remove('active');
      webrtcControls.style.display = 'none';
      localControls.style.display = 'block';
    };

    // Analysis WebSocket Connection
    function connectAnalysisSocket() {
      if (analysisSocket && analysisSocket.readyState === WebSocket.OPEN) return;
      const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const url = `${wsProto}//${location.host}/api/v1/ws/analyze`;
      analysisSocket = new WebSocket(url);

      analysisSocket.onopen = () => log("Connected to backend analysis streaming socket.");
      analysisSocket.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'status') {
            if (msg.data && msg.data.state) processingStateDisplay.textContent = msg.data.state;
          } else if (msg.type === 'result') {
            updateResults(msg.data);
          }
        } catch (e) {
          console.error("Error parsing analysis message:", e);
        }
      };
      analysisSocket.onclose = () => log("Analysis streaming socket closed.");
    }

    function updateResults(data) {
      if (!data) return;
      const dfProb = Number(data.deepfake_probability ?? 0) * 100;
      const scamProb = Number(data.scam_probability ?? 0) * 100;
      const overall = Number(data.overall_risk_score ?? 0) * 100;
      const category = data.risk_category || 'LOW';

      deepfakeVal.textContent = dfProb.toFixed(1) + '%';
      scamVal.textContent = scamProb.toFixed(1) + '%';
      overallRiskVal.textContent = overall.toFixed(1) + '%';
      riskCategoryVal.textContent = category;

      if (category === 'HIGH') {
        riskCategoryVal.style.color = '#ef4444';
        alertBox.textContent = 'HIGH RISK: SUSPECTED DEEPFAKE / FRAUDULENT CALL';
        alertBox.classList.add('show');
      } else if (category === 'MEDIUM') {
        riskCategoryVal.style.color = '#f59e0b';
        alertBox.textContent = 'WARNING: MODERATE RISK DETECTED';
        alertBox.classList.add('show');
      } else {
        riskCategoryVal.style.color = '#22c55e';
        alertBox.classList.remove('show');
      }

      if (data.transcript) {
        transcriptBox.textContent = data.transcript;
      }

      if (data.recommendation) {
        recommendationBox.textContent = data.recommendation;
      }

      if (data.latency_metrics && data.latency_metrics.end_to_end_latency_ms !== undefined) {
        latencyDisplay.textContent = `${data.latency_metrics.end_to_end_latency_ms} ms`;
      }

      if (data.processing_state) {
        processingStateDisplay.textContent = data.processing_state;
      }

      if (data.deepfake_model_status) {
        deepfakeStatusBadge.textContent = data.deepfake_model_status;
      }
      if (data.scam_detection_method) {
        scamStatusBadge.textContent = data.scam_detection_method;
      }
    }

    // Audio Analysis Tapping using Web Audio API
    function attachAudioAnalysis(mediaStream, label) {
      connectAnalysisSocket();
      analyzedTrackLabel.textContent = label;
      log(`Attached real-time audio pipeline to ${label}`);

      audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      audioSource = audioCtx.createMediaStreamSource(mediaStream);
      audioProcessor = audioCtx.createScriptProcessor(4096, 1, 1);

      audioProcessor.onaudioprocess = (e) => {
        if (!bothConsented && mode === 'webrtc') return;
        if (!analysisSocket || analysisSocket.readyState !== WebSocket.OPEN) return;

        const input = e.inputBuffer.getChannelData(0);
        const floatArr = new Float32Array(input.length);
        for (let i = 0; i < input.length; i++) floatArr[i] = input[i];

        const bytes = new ArrayBuffer(floatArr.length * 4);
        const view = new DataView(bytes);
        for (let i = 0; i < floatArr.length; i++) view.setFloat32(i * 4, floatArr[i], true);

        let binary = '';
        const uint8 = new Uint8Array(bytes);
        for (let i = 0; i < uint8.length; i++) binary += String.fromCharCode(uint8[i]);

        const packet = {
          type: "audio_chunk",
          data: btoa(binary),
          sample_rate: 16000,
          is_final: false,
          timestamp: Date.now() / 1000.0
        };
        analysisSocket.send(JSON.stringify(packet));
      };

      audioSource.connect(audioProcessor);
      audioProcessor.connect(audioCtx.destination);
    }

    function detachAudioAnalysis() {
      if (audioProcessor) {
        audioProcessor.disconnect();
        audioProcessor = null;
      }
      if (audioSource) {
        audioSource.disconnect();
        audioSource = null;
      }
      if (audioCtx) {
        audioCtx.close();
        audioCtx = null;
      }
      analyzedTrackLabel.textContent = "None";
    }

    // WebRTC Implementation
    async function startWebRTCCall(callId, creating) {
      if (!consentCheckbox.checked) {
        alert("You must check the consent box before starting the call.");
        return;
      }

      currentCallId = callId;
      isCaller = creating;
      webrtcStatusPill.className = "pill pill-calling";
      webrtcStatusPill.textContent = "Connecting...";

      try {
        localStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: 16000,
            echoCancellation: true,
            noiseSuppression: true
          }
        });
        log("Local microphone access granted for WebRTC call.");
      } catch (err) {
        alert("Microphone access denied: " + err.message);
        webrtcStatusPill.className = "pill pill-error";
        webrtcStatusPill.textContent = "Mic Denied";
        return;
      }

      const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const sigUrl = `${wsProto}//${location.host}/api/v1/call/ws/${callId}/${peerId}`;
      signalingSocket = new WebSocket(sigUrl);

      signalingSocket.onopen = () => {
        log(`Signaling connected to room ${callId}`);
        // Send initial consent
        signalingSocket.send(JSON.stringify({
          type: "consent",
          consent: consentCheckbox.checked
        }));
      };

      signalingSocket.onmessage = async (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "session_info") {
          log(`Registered as ${msg.role}. Peer connected: ${msg.peer_connected}`);
          if (msg.peer_connected && isCaller) {
            setupPeerConnection();
            createAndSendOffer();
          }
        } else if (msg.type === "peer_joined") {
          log(`Peer joined call room.`);
          if (isCaller) {
            setupPeerConnection();
            createAndSendOffer();
          }
        } else if (msg.type === "consent_update") {
          bothConsented = msg.both_consented;
          const myConsent = consentCheckbox.checked ? "Granted" : "Pending";
          const remConsent = bothConsented || (msg.peer_id !== peerId && msg.consent) ? "Granted" : "Pending";
          consentStatusText.textContent = `Your consent: ${myConsent} | Remote consent: ${remConsent}`;
          if (bothConsented) {
            log("✅ Mutual consent granted by both participants. Audio analysis activated.");
          } else {
            log("Waiting for mutual consent from both participants...");
          }
        } else if (msg.type === "offer") {
          log("Received WebRTC Offer from remote peer.");
          setupPeerConnection();
          await peerConnection.setRemoteDescription(new RTCSessionDescription(msg.payload));
          const answer = await peerConnection.createAnswer();
          await peerConnection.setLocalDescription(answer);
          signalingSocket.send(JSON.stringify({
            type: "answer",
            payload: answer
          }));
        } else if (msg.type === "answer") {
          log("Received WebRTC Answer from remote peer.");
          await peerConnection.setRemoteDescription(new RTCSessionDescription(msg.payload));
        } else if (msg.type === "ice_candidate") {
          if (peerConnection && msg.payload) {
            try {
              await peerConnection.addIceCandidate(new RTCIceCandidate(msg.payload));
            } catch (e) {
              console.warn("Error adding ICE candidate:", e);
            }
          }
        } else if (msg.type === "peer_left") {
          log("Remote peer hung up.");
          endCall();
        }
      };

      createCallBtn.style.display = 'none';
      joinCallBtn.style.display = 'none';
      hangupBtn.style.display = 'inline-block';
    }

    function setupPeerConnection() {
      if (peerConnection) return;
      const config = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };
      peerConnection = new RTCPeerConnection(config);

      // Add local audio track
      localStream.getTracks().forEach(track => peerConnection.addTrack(track, localStream));

      peerConnection.onicecandidate = (e) => {
        if (e.candidate && signalingSocket && signalingSocket.readyState === WebSocket.OPEN) {
          signalingSocket.send(JSON.stringify({
            type: "ice_candidate",
            payload: e.candidate
          }));
        }
      };

      peerConnection.ontrack = (event) => {
        log("Received remote audio track from peer.");
        remoteStream = event.streams[0];
        remoteAudio.srcObject = remoteStream;

        webrtcStatusPill.className = "pill pill-connected";
        webrtcStatusPill.textContent = "Call Connected";

        // Tapping REMOTE audio track for deepfake analysis
        attachAudioAnalysis(remoteStream, "Remote Audio Track (WebRTC)");
      };

      peerConnection.onconnectionstatechange = () => {
        log(`WebRTC connection state: ${peerConnection.connectionState}`);
        if (peerConnection.connectionState === 'disconnected' || peerConnection.connectionState === 'failed') {
          endCall();
        }
      };
    }

    async function createAndSendOffer() {
      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);
      signalingSocket.send(JSON.stringify({
        type: "offer",
        payload: offer
      }));
    }

    function endCall() {
      log("Ending call session.");
      detachAudioAnalysis();
      if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
      }
      if (localStream) {
        localStream.getTracks().forEach(t => t.stop());
        localStream = null;
      }
      if (signalingSocket) {
        signalingSocket.close();
        signalingSocket = null;
      }
      if (analysisSocket && analysisSocket.readyState === WebSocket.OPEN) {
        analysisSocket.send(JSON.stringify({ type: "reset" }));
      }
      webrtcStatusPill.className = "pill pill-idle";
      webrtcStatusPill.textContent = "Call Ended";
      createCallBtn.style.display = 'inline-block';
      joinCallBtn.style.display = 'inline-block';
      hangupBtn.style.display = 'none';
      bothConsented = false;
      consentStatusText.textContent = "Your consent: Pending | Remote consent: Pending";
    }

    // UI Button Handlers
    createCallBtn.onclick = async () => {
      try {
        const res = await fetch("/api/v1/call/session/create", { method: "POST" });
        const data = await res.json();
        callIdInput.value = data.call_id;
        log(`Created call session ${data.call_id}. Waiting for peer to join...`);
        startWebRTCCall(data.call_id, true);
      } catch (e) {
        alert("Failed to create call session: " + e.message);
      }
    };

    joinCallBtn.onclick = () => {
      const cid = callIdInput.value.trim();
      if (!cid) {
        alert("Please enter a Call ID to join.");
        return;
      }
      log(`Joining call session ${cid}...`);
      startWebRTCCall(cid, false);
    };

    hangupBtn.onclick = () => endCall();

    consentCheckbox.onchange = () => {
      if (signalingSocket && signalingSocket.readyState === WebSocket.OPEN) {
        signalingSocket.send(JSON.stringify({
          type: "consent",
          consent: consentCheckbox.checked
        }));
      }
    };

    // Local Test Mode
    startLocalBtn.onclick = async () => {
      if (!localConsentCheckbox.checked) {
        alert("Please check the consent box to proceed with local microphone testing.");
        return;
      }
      try {
        localStream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, sampleRate: 16000 }
        });
        localStatusPill.className = "pill pill-connected";
        localStatusPill.textContent = "Testing Mic";
        startLocalBtn.style.display = "none";
        stopLocalBtn.style.display = "inline-block";
        bothConsented = true;
        attachAudioAnalysis(localStream, "Local Microphone (Test Mode)");
      } catch (err) {
        alert("Local mic access error: " + err.message);
      }
    };

    stopLocalBtn.onclick = () => {
      detachAudioAnalysis();
      if (localStream) {
        localStream.getTracks().forEach(t => t.stop());
        localStream = null;
      }
      localStatusPill.className = "pill pill-idle";
      localStatusPill.textContent = "Idle";
      startLocalBtn.style.display = "inline-block";
      stopLocalBtn.style.display = "none";
      if (analysisSocket && analysisSocket.readyState === WebSocket.OPEN) {
        analysisSocket.send(JSON.stringify({ type: "reset" }));
      }
    };
  </script>
</body>
</html>
"""


@router.get("/live-call")
async def live_call_page():
    """Serves the in-app WebRTC live call & audio analyzer interface."""
    return HTMLResponse(LIVE_CALL_HTML)
