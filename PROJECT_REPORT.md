# REAL-TIME DEEPFAKE VOICE INTERCEPTOR & AUDIO AUTHENTICATOR

**BCSE497J - Project-I**  
**School of Computer Science and Engineering (SCOPE)**  
**Vellore Institute of Technology, Vellore**  

**Students**:
- 23BAI0005 RISHIKA MURUGESH
- 23BCB0143 ANISH S.A

**Supervisor**:
- Dr. VIJAYA KUMAR K  
  Associate Professor Senior Grade 1  
  School of Computer Science and Engineering (SCOPE)  

**B.Tech. in Computer Science and Engineering**  
(Specialization: Information Security)  
October 2026  

---

## ABSTRACT

The rapid proliferation of generative artificial intelligence and voice cloning technologies has introduced severe cyber threats, enabling malicious actors to impersonate trusted individuals, executives, and family members during live voice conversations. Conventional voice anti-spoofing and deepfake detection solutions operate primarily in post-call or offline environments, leaving live conversations defenseless against real-time social-engineering and financial fraud.

This project, titled **“Real-Time Deepfake Voice Interceptor & Audio Authenticator,”** designs and implements an end-to-end protective layer for live voice communication and uploaded audio verification. The system continuously analyzes speech to determine whether acoustic characteristics are genuine or AI-synthesized, while simultaneously applying automatic speech recognition (ASR) and natural language processing (NLP) to detect fraudulent or social-engineering intent in real time.

The system features:
1. An **acoustic deepfake detection model** built on a CNN + GRU architecture that inspects spectral representations for synthesis artifacts, transparently reporting model checkpoint status.
2. A **Whisper-based ASR pipeline** integrated with a **hybrid scam-intent detector** that identifies high-risk conversational patterns (financial urgency, credential theft, authority impersonation).
3. A **risk-fusion engine** combining acoustic spoof probabilities and semantic scam scores into actionable risk levels (`LOW`, `MEDIUM`, `HIGH`).
4. An **in-app browser-to-browser WebRTC audio calling module** with room session management, explicit mutual consent verification, and remote audio track interception via the Web Audio API, complemented by a local microphone test mode.
5. A **low-latency streaming architecture** featuring a bounded sliding audio window (3.5 seconds), a bounded queue with backpressure handling, asynchronous event-loop offloading, and throttled ASR cadence.
6. A **privacy-first data handling framework** guaranteeing zero persistence of raw audio on disk or database, in-memory decoding with guaranteed temporary cleanup, and an on-demand analysis history purge.

The system is evaluated using an automated test suite verifying streaming boundaries, queue backpressure, protocol parsing, and model pipelines. Limitations regarding carrier/PSTN telephone network interception (which necessitates external SIP trunking infrastructure) and dataset licensing protocols are transparently documented.

**Keywords**: Deepfake Voice Detection, Audio Authentication, Anti-Spoofing, WebRTC VoIP, Automatic Speech Recognition, Scam-Intent Analysis, Bounded Streaming, Real-Time Systems, Privacy by Design.

---

## TABLE OF CONTENTS

1. INTRODUCTION  
   1.1 Background  
   1.2 Motivation  
   1.3 Scope and Delimitations  
2. LITERATURE REVIEW AND RESEARCH GAPS  
   2.1 Literature Review  
   2.2 Research Gaps  
   2.3 Project Objectives  
   2.4 Problem Statement  
3. SYSTEM REQUIREMENTS AND FEASIBILITY  
   3.1 Functional Requirements  
   3.2 Non-Functional Requirements  
   3.3 Security, Privacy, and Consent Architecture  
   3.4 Feasibility Analysis  
   3.5 System Specifications  
4. DESIGN APPROACH AND ARCHITECTURE  
   4.1 System Architecture  
   4.2 WebRTC Signaling and Remote Track Interception  
   4.3 Bounded Audio Streaming and Backpressure Control  
   4.4 Data Flow and Sequence Diagrams  
5. IMPLEMENTATION METHODOLOGY  
   5.1 Audio Preprocessing and Feature Extraction  
   5.2 Deepfake Detection Model  
   5.3 Automatic Speech Recognition (ASR)  
   5.4 Scam Intent Analysis  
   5.5 Risk Fusion Engine  
   5.6 API and Interactive Dashboard Integration  
6. DATASET PROTOCOLS AND EVALUATION  
   6.1 ASVspoof 2019 Protocol Integration  
   6.2 Protocol Validation Tool  
   6.3 Demonstration Audio vs. Evaluation Speech  
   6.4 Experimental Verification and Test Results  
7. CONCLUSION, LIMITATIONS AND FUTURE WORK  
   7.1 Summary of Contributions  
   7.2 Technical Limitations (WebRTC vs. PSTN Telephony)  
   7.3 Future Research Directions  
8. REFERENCES  

---

## 1. INTRODUCTION

### 1.1 Background
Recent advances in generative deep learning—particularly diffusion models, neural vocoders, and zero-shot voice cloning algorithms—enable high-fidelity synthesis of human voices using only seconds of target audio. These synthetic voices can match pitch, cadence, and timbre with alarming realism. While these tools have legitimate applications in accessibility, localization, and media production, they have also enabled a dangerous wave of voice-cloning fraud, vishing (voice phishing), and virtual kidnapping scams.

### 1.2 Motivation
Traditional voice biometrics and anti-spoofing techniques are largely designed for post-call forensic investigations or batch offline verification. By the time a forensic analysis is completed, financial transactions have settled and sensitive credentials have been compromised. To prevent harm during active communication, detection must occur in real time, concurrently evaluating acoustic authenticity and semantic intent before a victim complies with fraudulent instructions.

### 1.3 Scope and Delimitations
This project implements a complete, end-to-end prototype supporting:
- Pre-recorded audio file upload analysis across standard formats (WAV, MP3, FLAC, M4A, OGG).
- In-app browser-to-browser WebRTC audio calling with room creation/joining, signaling, mutual consent enforcement, and real-time remote audio track interception.
- Local microphone test mode for single-browser developmental testing.
- Bounded sliding-window acoustic buffering, bounded queue processing, and event-loop offloading.
- Unified canonical analysis output formats and privacy-first data handling.

**Delimitations**:
- **Carrier/PSTN Telephone Interception**: This project does **not** intercept conventional landline or mobile cellular telephone network (PSTN) calls. Ingesting cellular voice streams requires specialized telecommunication carrier partnerships, PBX integrations, or third-party SIP trunking providers (e.g., Twilio SIP, Asterisk) with dedicated credentials and telephony routing. The live calling capability implemented here is browser-based WebRTC VoIP.
- **Model Checkpoints**: Due to the proprietary terms and large storage footprints of official anti-spoofing corpora (such as ASVspoof 2019), the repository provides the complete neural network architecture and training/evaluation pipeline. When uninitialized, the system explicitly reports demo/uninitialized model status rather than claiming fabricated accuracy.

---

## 2. LITERATURE REVIEW AND RESEARCH GAPS

### 2.1 Literature Review
1. **Acoustic Spoof Detection (RawNet2 and Spectrogram Models)**: Modern anti-spoofing research highlights spectral anomalies introduced by vocoders and text-to-speech (TTS) architectures. Raw waveform CNNs (RawNet2) and spectrogram-based CNN + GRU networks learn spectral discontinuity artifacts at high frequencies that are imperceptible to human ears but distinguishable to deep neural networks.
2. **Speech Recognition (OpenAI Whisper)**: Whisper utilizes an encoder-decoder Transformer trained on 680,000 hours of multilingual audio. It demonstrates robustness against diverse acoustic environments, accents, and background noise, making it suitable for conversational transcription.
3. **Intent Detection in Voice Fraud**: Traditional NLP scam detectors rely heavily on rule-based keyword sets (e.g., "bank account", "OTP", "urgent", "police"). Recent work integrates zero-shot cross-encoders (e.g., BART-large-MNLI) to classify conversational turns into semantic fraud categories without requiring extensive labeled training data.

### 2.2 Research Gaps
- **Lack of Multi-Modal Fusion**: Most existing systems analyze either the acoustic signal (anti-spoofing) or the text transcript (fraud detection) in isolation. Acoustic detectors frequently generate false alarms on compressed audio, while text-only detectors miss synthesized voices reading benign scripts.
- **Memory Accumulation in Streaming**: Many proof-of-concept streaming detectors append incoming chunks to an unbounded list, causing memory bloat, high garbage-collection pauses, and escalating processing latency as call duration increases.
- **Privacy Disregard in Real-Time Systems**: Existing voice-interceptor prototypes frequently save unencrypted raw audio files to disk, violating user privacy rights and wiretapping regulations.

### 2.3 Project Objectives
1. Implement a unified risk-scoring engine that fuses acoustic deepfake probability with semantic scam probability.
2. Build an in-app WebRTC VoIP calling interface with mutual consent verification and remote track interception.
3. Engineer a low-latency, bounded-buffer streaming pipeline with backpressure handling and non-blocking model execution.
4. Establish strict privacy protections: zero raw audio persistence, in-memory decoding, and on-demand history purge.
5. Provide protocol validation tools and honest documentation aligned strictly with empirical reality.

---

## 3. SYSTEM REQUIREMENTS AND FEASIBILITY

### 3.1 Functional Requirements
- **FR1 (Upload Analysis)**: Users can upload audio files and receive deepfake probability, transcript, scam indicators, and overall risk classification.
- **FR2 (WebRTC Peer Calling)**: Users can create or join a call room using a session ID, exchange SDP offers/answers and ICE candidates over a WebSocket signaling channel, and engage in peer-to-peer VoIP audio.
- **FR3 (Remote Track Interception)**: The system must capture only the incoming remote audio track of the call peer (not the local user's own voice) for deepfake and scam analysis.
- **FR4 (Local Mic Test Mode)**: Users can test detection pipelines using their local microphone without creating a call room.
- **FR5 (Mutual Consent)**: Live audio capture and transmission must require explicit opt-in confirmation from participants.
- **FR6 (Analysis History Purge)**: Users can delete all stored session records via API or dashboard.

### 3.2 Non-Functional Requirements
- **NFR1 (Bounded Memory & Low Latency)**: Memory usage must remain bounded regardless of call length. Acoustic sliding window is capped at 3.5 seconds; inference calls must not block the ASGI event loop.
- **NFR2 (Backpressure Resilience)**: If model inference falls behind audio chunk arrival, the queue must drop oldest chunks rather than accumulating unbounded latency.
- **NFR3 (Privacy by Design)**: No raw audio recordings shall be saved to persistent disk or SQLite database by default.
- **NFR4 (Configurable CORS)**: Strict origin validation must restrict cross-site WebSocket and HTTP requests.

### 3.3 Security, Privacy, and Consent Architecture
- **In-Memory Audio Processing**: Raw PCM and encoded audio buffers are handled strictly in volatile RAM using `io.BytesIO`. When third-party decoders require filesystem access, files are managed inside `try ... finally` blocks guaranteeing immediate unlinking upon completion.
- **Dual-Sided Consent**: WebRTC session signaling synchronizes consent states. Live audio transmission begins only when participants acknowledge the monitoring terms.
- **Compliance Warning**: Prominent banners remind users that recording or analyzing calls without mutual consent is legally prohibited in many jurisdictions.

---

## 4. DESIGN APPROACH AND ARCHITECTURE

### 4.1 System Architecture

```
                    ┌─────────────────────────────────────────┐
                    │            Client Browsers              │
                    │   (Participant A)     (Participant B)   │
                    └───────────┬─────────────────┬───────────┘
                                │ WebRTC Audio    │
                                └───────▲─────────┘
                                        │ (Remote Audio Track)
                                        ▼
                            [Web Audio API Node]
                           (16kHz PCM Downsample)
                                        │
                                        ▼ (WebSocket)
                    ┌─────────────────────────────────────────┐
                    │            FastAPI Backend              │
                    │                                         │
                    │  ┌───────────────────────────────────┐  │
                    │  │ StreamingAnalysisSession          │  │
                    │  │  - Bounded Queue (maxsize=10)     │  │
                    │  │  - Backpressure Dropping          │  │
                    │  │  - Bounded Sliding Window (3.5s)  │  │
                    │  └─────────────────┬─────────────────┘  │
                    │                    │                    │
                    │       ┌────────────┴────────────┐       │
                    │       ▼ (asyncio.to_thread)     ▼       │
                    │  ┌──────────────┐      ┌─────────────┐  │
                    │  │ Deepfake CNN │      │ Whisper ASR │  │
                    │  │   + GRU      │      │ (Cadenced)  │  │
                    │  └──────┬───────┘      └──────┬──────┘  │
                    │         │                     │         │
                    │         │              ┌──────▼──────┐  │
                    │         │              │  NLP Scam   │  │
                    │         │              │  Detector   │  │
                    │         │              └──────┬──────┘  │
                    │         └──────────┬──────────┘         │
                    │                    ▼                    │
                    │         ┌─────────────────────┐         │
                    │         │ Risk Fusion Engine  │         │
                    │         └──────────┬──────────┘         │
                    │                    │                    │
                    │                    ▼                    │
                    │         [Canonical JSON Result]         │
                    └────────────────────┬────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │ Streamlit Dashboard & WebRTC Visualizer │
                    │  - Deepfake / Scam Gauges               │
                    │  - Live Transcript & Alerts             │
                    │  - Latency Metrics & Timestamps         │
                    │  - Model Status Disclosures             │
                    └─────────────────────────────────────────┘
```

### 4.2 WebRTC Signaling and Remote Track Interception
The system deploys a dedicated signaling channel at `/call/ws/{call_id}/{peer_id}` managed by `CallSessionManager`. Peers negotiate connections via SDP exchange (`offer`, `answer`) and ICE candidates. Once the peer connection is established, the client application executes:

```javascript
peerConnection.ontrack = (event) => {
    remoteAudioElement.srcObject = event.streams[0];
    // Tap the incoming remote track for live analysis
    const sourceNode = audioContext.createMediaStreamSource(event.streams[0]);
    sourceNode.connect(scriptProcessorNode);
    scriptProcessorNode.connect(audioContext.destination);
};
```
This guarantees that the analyzer inspects the **remote caller's voice**, avoiding acoustic self-interference from the local speaker.

### 4.3 Bounded Audio Streaming and Backpressure Control
- **Bounded Buffer**: Implemented in `BoundedAudioBuffer`, which maintains a numpy float32 ring buffer capped at 56,000 samples (3.5 seconds @ 16kHz). Older acoustic context rolls off smoothly as new chunks arrive.
- **Queue Limits & Backpressure**: Incoming chunks are routed through `asyncio.Queue(maxsize=10)`. When the worker loop is busy running inference, subsequent chunks trigger an oldest-frame drop policy with a backpressure warning logged, preventing latency from cascading into multiple seconds.
- **Off-Loop Inference**: Intensive model computation is dispatched using Python's `asyncio.to_thread()`, keeping FastAPI's asynchronous event loop free to handle signaling and WebSocket frames.

---

## 5. IMPLEMENTATION METHODOLOGY

### 5.1 Audio Preprocessing
Audio inputs are standardized to 16,000 Hz mono PCM. Preprocessing steps include:
- Peak normalization to [-1.0, 1.0].
- Silence trimming using energy thresholding.
- Log-Mel spectrogram computation with 128 Mel bands, FFT window of 2048, and hop length of 512 samples.

### 5.2 Deepfake Detection Model
The acoustic deepfake detector utilizes a hybrid architecture:
- **Convolutional Frontend**: 4 convolutional blocks with batch normalization, ReLU activations, and max pooling to capture local time-frequency synthesis artifacts.
- **Recurrent Layer**: Bidirectional GRU to model temporal voice continuity across frames.
- **Attention & Pooling**: Temporal average pooling feeding fully connected layers that output a softmax probability of synthetic origin.
- **Model Status Reporting**: The model class checks for verified weights on disk. When absent, it outputs `"Untrained / Random Weights (Demo Only - No Verified Checkpoint)"` to ensure transparent evaluation.

### 5.3 ASR and Transcript Generation
Automatic speech recognition is powered by OpenAI Whisper (`base` model by default). In streaming mode, Whisper inference is cadenced—invoked periodically or upon speech turn boundaries—to avoid per-chunk computational bottlenecks.

### 5.4 Scam Intent Analysis
Semantic analysis combines:
1. **Rule-Based Indicator Extraction**: Categorized keyword sets targeting financial demands (e.g., "gift card", "crypto", "wire transfer"), urgency (e.g., "immediately", "arrest warrant", "account suspended"), and credential requests (e.g., "password", "OTP", "PIN").
2. **Zero-Shot NLP Classification**: Uses `facebook/bart-large-mnli` to evaluate hypothesis entailment against fraud labels. Offline and local configurations fall back cleanly to keyword heuristics without hanging on network timeouts.

### 5.5 Risk Fusion Engine
The engine fuses acoustic and semantic evidence:
$$\text{Risk Score} = w_d \cdot P(\text{Deepfake}) + w_s \cdot P(\text{Scam})$$
where default weights are $w_d = 0.5$ and $w_s = 0.5$. Risk categories are assigned as:
- **LOW** ($0.00 \le \text{Score} < 0.40$): Low risk, standard vigilance advised.
- **MEDIUM** ($0.40 \le \text{Score} < 0.70$): Moderate risk; heightened caution recommended.
- **HIGH** ($0.70 \le \text{Score} \le 1.00$): Critical fraud warning triggered; user advised to terminate call.

### 5.6 API and Interactive Dashboard
- **REST Endpoints**: `/api/v1/analyze/audio`, `/api/v1/analyze/transcript`, `/api/v1/results`, `/api/v1/history`, `/api/v1/privacy`, `/api/v1/health`.
- **WebRTC & WebSocket Endpoints**: `/call/app`, `/call/session/create`, `/call/ws/{call_id}/{peer_id}`, `/api/v1/ws/analyze`.
- **Streamlit Frontend**: Visualizes probability gauges, live scrolling transcripts, detected fraud indicators, model status badges, and history purge controls.

---

## 6. DATASET PROTOCOLS AND EVALUATION

### 6.1 ASVspoof 2019 Protocol Integration
To maintain rigorous research standards, the project adheres to the official ASVspoof 2019 Logical Access (LA) protocol standard. The dataset partitions speakers into strictly non-overlapping subsets:
- **Train Split**: Used for training acoustic feature extractors.
- **Development Split**: Used for hyperparameter tuning and model checkpoint selection.
- **Evaluation Split (Held-Out)**: Dedicated strictly to unbiased performance testing.

Arbitrary directory mixing (such as copying files into unpartitioned `data/real` or `data/fake` folders) or random splitting of the official corpus is strictly prohibited, as it causes speaker leakage across splits and inflates accuracy metrics.

### 6.2 Protocol Validation Tool
The project includes a standalone protocol validator (`training/validate_protocol.py`):
```bash
python -m training.validate_protocol --protocol path/to/protocol.txt --audio_dir path/to/audio/
```
The validator inspects each protocol line, verifies that the referenced audio file exists on disk, confirms format validity, and tallies bona fide versus spoof sample distributions.

### 6.3 Demonstration Audio vs. Evaluation Speech
Synthetic audio files generated by `populate_audio_datasets.py` in `data/demo_audio/` are pure mathematical test tones (sine waves and noise bursts). They are engineered exclusively for pipeline verification, format compatibility, and unit testing. They do **not** represent human speech and must not be used to claim model accuracy.

### 6.4 Experimental Verification and Test Results
An automated test suite comprising 24 unit and integration tests verifies the entire system:

| Test Class / Suite | Test Case | Status | Verified Functionality |
| :--- | :--- | :---: | :--- |
| `TestBoundedAudioBuffer` | `test_buffer_bounds_max_samples` | PASSED | Buffer strictly capped at 56,000 samples (3.5s) |
| `TestBoundedAudioBuffer` | `test_chunk_boundaries_and_empty_chunks` | PASSED | Handles variable chunk sizes, rejects empty frames |
| `TestBoundedAudioBuffer` | `test_buffer_reset` | PASSED | Clean buffer clearing on disconnect |
| `TestStreamingAnalysisSession` | `test_queue_limit_and_backpressure_drop` | PASSED | Bounded queue (max 10), drops oldest on saturation |
| `TestStreamingAnalysisSession` | `test_session_close_and_disconnect` | PASSED | Graceful session termination and task cleanup |
| `TestCallSessionManager` | `test_session_creation_roles_and_consent` | PASSED | Room creation, role assignment, dual consent |
| `TestHistoryClearing` | `test_clear_all_records` | PASSED | Complete purging of SQLite analysis records |
| `TestASVspoofProtocolValidation` | `test_protocol_validator_success` | PASSED | Correct parsing and class counting of protocol |
| `TestASVspoofProtocolValidation` | `test_protocol_validator_missing_files` | PASSED | Accurate detection and reporting of missing audio |
| `TestAudioProcessor` | Normalization, Mel Spectrogram, Chunking | PASSED | Audio loading, transformation, and segmentation |
| `TestDeepfakeDetector` | Model creation, forward pass, prediction | PASSED | Tensor dimensions, forward inference, softmax |
| `TestScamDetector` | Indicators, normal text, empty input | PASSED | Keyword matching, edge-case resilience |
| `TestRiskEngine` | LOW, MEDIUM, HIGH risk, recommendations | PASSED | Weighted score calculation and threshold mapping |
| `TestIntegration` | End-to-end preprocessing pipeline | PASSED | Integration between processor and tensor pipelines |

**Total Test Results**: 24 passed in 34.92s with 0 errors.  
**System Verification**: `verify_system.py` confirms 10/10 subsystems operational.

---

## 7. CONCLUSION, LIMITATIONS AND FUTURE WORK

### 7.1 Summary of Contributions
This project successfully designed, implemented, and verified an end-to-end real-time deepfake voice interception and fraud detection prototype. Key achievements include:
1. Real-time browser-to-browser WebRTC audio calling with mutual consent and remote track tapping.
2. Low-latency, bounded sliding-window streaming architecture with backpressure management.
3. Privacy-first data handling ensuring zero persistence of raw audio.
4. Transparent model status reporting and ASVspoof protocol validation tooling.
5. A comprehensive 24-test verification suite confirming stability across all pipeline components.

### 7.2 Technical Limitations (WebRTC vs. PSTN Telephony)
- **Carrier PSTN Telephony Barrier**: The system cannot directly tap or intercept traditional telephone or cellular carrier calls. Doing so requires carrier-grade telephony hardware, PBX switches, or SIP trunking provider credentials (e.g. Twilio, Asterisk), which are external to this project. The live call functionality operates via browser WebRTC VoIP.
- **Acoustic Generalization**: Deepfake detection accuracy in real-world scenarios is contingent upon training on diverse, high-quality datasets (such as ASVspoof 2019/2021). The uninitialized architecture provided in the repository requires training with official data before deployment.

### 7.3 Future Research Directions
1. **Telephony Integration**: Connecting the FastAPI streaming pipeline to a cloud PBX or Twilio SIP trunk to enable inbound/outbound cellular call analysis.
2. **Speaker Diarization**: Integrating neural diarization (e.g., PyAnnote) to separate multiple speakers in conference calls.
3. **Multilingual NLP Models**: Fine-tuning multilingual cross-encoders to identify fraud patterns across regional languages and dialects.
4. **Hardware Acceleration**: Quantizing models using ONNX Runtime or TensorRT for embedded and mobile deployment.

---

## 8. REFERENCES

[1] H. Tak, J. Patino, M. Todisco, A. Nautsch, N. Evans, and A. Larcher, “End-to-end anti-spoofing with RawNet2,” in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2021, pp. 6369–6373.

[2] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” *arXiv preprint arXiv:2212.04356*, 2022.

[3] M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, and L. Zettlemoyer, “BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension,” in *Proc. ACL*, 2020, pp. 7871–7880.

[4] ASVspoof Consortium, “The ASVspoof 2019 challenge: Evaluation plan,” *Tech. Rep.*, 2019. Available: https://www.asvspoof.org/

[5] World Wide Web Consortium (W3C), “WebRTC 1.0: Real-Time Communication Between Browsers,” *W3C Recommendation*, 2021. Available: https://www.w3.org/TR/webrtc/

[6] FastAPI Framework Documentation, Sebastián Ramírez, 2024. Available: https://fastapi.tiangolo.com/

[7] Streamlit Documentation, Snowflake Inc., 2024. Available: https://docs.streamlit.io/
