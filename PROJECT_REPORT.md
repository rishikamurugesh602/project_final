# REAL-TIME DEEPFAKE VOICE INTERCEPTOR & AUDIO AUTHENTICATOR

BCSE497J - Project-I

Students:
- 23BAI0005 RISHIKA MURUGESH
- 23BCB0143 ANISH S.A

Supervisor:
- Dr. VIJAYA KUMAR K
  Associate Professor Senior Grade 1
  School of Computer Science and Engineering (SCOPE)

B.Tech. in Computer Science and Engineering
(Specialization: Information Security)
School of Computer Science and Engineering
September 2026

---

## ABSTRACT

The rapid growth of AI-based voice cloning technologies has introduced a new class of cyber threats in which attackers impersonate trusted individuals to manipulate victims during live phone or VoIP conversations. Existing voice anti-spoofing solutions mostly perform offline or post-call analysis, leaving real-time conversations vulnerable to emerging voice-based fraud.

This project, titled “Real-Time Deepfake Voice Interceptor & Audio Authenticator,” proposes a real-time voice authentication and scam-detection system that acts as a protective layer during an ongoing call. The system continuously analyses live audio streams to determine whether a voice is genuine or AI-generated, while simultaneously applying automatic speech recognition and natural language processing techniques to identify suspicious conversational patterns that indicate social-engineering or scam intent.

An acoustic deepfake-detection model built on a RawNet2-style architecture flags synthetic speech directly from raw audio, while a Whisper-based transcription pipeline feeds a scam-intent classifier. The outputs of both branches are fused by a lightweight risk-scoring engine that raises real-time alerts through a FastAPI backend and an interactive dashboard. By combining acoustic deepfake detection with scam-intent analysis in a single low-latency pipeline, the system helps users identify and respond to voice-based fraud as it happens rather than after the damage is done.

Keywords: Deepfake Voice Detection, Audio Authentication, Anti-Spoofing, Natural Language Processing, Scam-Intent Classification, Real-Time Systems.

---

## TABLE OF CONTENTS

1. INTRODUCTION
   1.1 Background
   1.2 Motivation
   1.3 Scope of the Project

2. PROJECT DESCRIPTION AND GOALS
   2.1 Literature Review
   2.2 Research Gap
   2.3 Objectives
   2.4 Problem Statement
   2.5 Project Plan

3. TECHNICAL SPECIFICATION
   3.1 Requirements
   3.1.1 Functional
   3.1.2 Non-Functional
   3.2 Feasibility Study
   3.2.1 Technical Feasibility
   3.2.2 Economic Feasibility
   3.2.3 Social Feasibility
   3.3 System Specification
   3.3.1 Hardware Specification
   3.3.2 Software Specification

4. DESIGN APPROACH AND DETAILS
   4.1 System Architecture
   4.2 Design
   4.2.1 Data Flow Diagram
   4.2.2 Use Case Diagram

5. IMPLEMENTATION METHODOLOGY
   5.1 Audio Preprocessing
   5.2 Deepfake Detection Model
   5.3 ASR and Transcript Generation
   5.4 Scam Intent Analysis
   5.5 Risk Fusion Engine
   5.6 Backend and Dashboard Integration

6. RESULTS AND EVALUATION
   6.1 Experimental Setup
   6.2 Performance Metrics
   6.3 Sample Output Analysis
   6.4 Observations

7. CONCLUSION AND FUTURE WORK

8. REFERENCES

---

## 1. INTRODUCTION

### 1.1 Background

The rapid digitisation of voice communication has been accompanied by an equally rapid rise in AI-based voice cloning technologies. Deepfake audio generators can now synthesise a convincing replica of a person’s voice using only a few seconds of reference audio. This enables attackers to impersonate relatives, officials, colleagues, or trusted individuals during phone calls and VoIP communication.

Traditional voice anti-spoofing solutions are largely designed for offline or post-call analysis. Such systems are not useful in live conversations, where the decision must be made in near real time. As a result, victims may be manipulated before any alert can be triggered.

### 1.2 Motivation

Voice-based fraud is becoming increasingly common due to the ease with which AI systems can clone a target voice. Attackers now combine deepfake audio with a scripted social-engineering narrative to steal money or confidential information. The need for a real-time protective mechanism is therefore critical.

This project is motivated by the need for an intelligent system that can act during an active call and warn the user if the voice is synthetic or if the conversation is scam-like. The project uses deep learning, speech recognition, and NLP to make such detection possible.

### 1.3 Scope of the Project

The project focuses on building a real-time audio authentication and scam-detection system for live calls. It includes:

- audio preprocessing and feature extraction,
- deepfake detection using a deep learning model,
- ASR transcription using Whisper,
- scam-intent analysis from transcript,
- risk fusion and alert generation,
- dashboard-based visualization and system logging.

The solution is designed to be practical and deployable in VoIP or WebRTC-based environments while keeping the implementation lightweight and usable for academic demonstration.

---

## 2. PROJECT DESCRIPTION AND GOALS

### 2.1 Literature Review

The system is inspired by established research in three domains.

1. RawNet2 / end-to-end spoof detection
   - Tak et al. (2021) proposed an end-to-end anti-spoofing model that processes raw waveforms and learns directly from speech signals.
   - This approach avoids manual feature engineering and provides strong detection performance for synthetic speech.

2. Whisper speech recognition
   - Radford et al. (2022) introduced Whisper, a robust general-purpose ASR system trained on large-scale weak supervision.
   - Whisper is effective for noisy and varied audio conditions and is suitable for transcribing live conversational audio.

3. RoBERTa for NLP and scam-intent classification
   - Liu et al. (2019) proposed RoBERTa, which improves language understanding using optimized pretraining.
   - It is widely used for downstream classification tasks, including harmful content and scam-intent detection.

These works show that a hybrid system combining deepfake audio analysis and textual intent analysis is a strong direction for real-time fraud mitigation.

### 2.2 Research Gap

Although many anti-spoofing systems exist, most are designed for offline analysis rather than live conversation security. The key gaps are:

- Most systems detect spoofing only from audio and do not analyze conversational context.
- There is limited integration between deepfake detection and scam-intent detection in a single framework.
- Existing systems rarely provide real-time alerting during active calls.
- There is a lack of interactive dashboards that visualize risk for live voice analysis.

### 2.3 Objectives

The main objectives of the project are:

- to detect AI-generated or cloned voice in real time,
- to transcribe call audio accurately,
- to detect scam or social-engineering intent from transcript text,
- to fuse both audio and text signals into a unified risk score,
- to provide real-time alerts and dashboard-based monitoring,
- to develop a working academic prototype with practical deployment potential.

### 2.4 Problem Statement

AI-generated voice cloning has made it possible for attackers to convincingly impersonate trusted individuals during live phone calls or VoIP communications. Existing anti-spoofing tools typically operate after the call has ended and consider either audio alone or text alone. There is therefore a need for a real-time, integrated warning system that can simultaneously verify voice authenticity and detect scam-like intent before users disclose sensitive information.

### 2.5 Project Plan

The project is organized in phases:

- literature review,
- requirement analysis,
- system design,
- audio preprocessing pipeline,
- deepfake detection model development,
- ASR and scam detection integration,
- backend and dashboard implementation,
- testing and validation,
- documentation and final reporting.

---

## 3. TECHNICAL SPECIFICATION

### 3.1 Requirements

#### 3.1.1 Functional Requirements

- The system shall capture audio streams from live or uploaded calls.
- The system shall analyze audio frames for deepfake probability.
- The system shall transcribe the spoken content using ASR.
- The system shall analyze transcripts for scam intent.
- The system shall combine both scores into a unified risk category.
- The system shall generate alerts when risk crosses a threshold.
- The dashboard shall display risk, transcript, and alerts visually.

#### 3.1.2 Non-Functional Requirements

- Low latency for real-time analysis.
- Reliable operation under moderate load.
- Secure handling of voice data and session logs.
- Intuitive dashboard for analyst interaction.
- Modular and maintainable architecture.

### 3.2 Feasibility Study

#### 3.2.1 Technical Feasibility

The project is feasible because the required technologies are widely available and open-source. Python, FastAPI, Streamlit, PyTorch, Whisper, and Hugging Face libraries provide a practical implementation path.

#### 3.2.2 Economic Feasibility

The system uses open-source software and open models, making it economically feasible for an academic project. The cost is limited to infrastructure and compute resources.

#### 3.2.3 Social Feasibility

The solution is socially useful because it helps reduce fraud, identity theft, and scam victimization in voice-based communication. It also promotes awareness of the risks of AI-generated speech.

### 3.3 System Specification

#### 3.3.1 Hardware Specification

- Multi-core CPU
- Minimum 16 GB RAM
- SSD storage
- Optional NVIDIA GPU for faster inference

#### 3.3.2 Software Specification

- Operating System: Windows 11 / Ubuntu
- Programming Language: Python
- Frameworks: FastAPI, Streamlit
- Libraries: PyTorch, librosa, whisper, transformers, scikit-learn, pandas, plotly
- Database: SQLite
- Tools: VS Code, Jupyter Notebook

---

## 4. DESIGN APPROACH AND DETAILS

### 4.1 System Architecture

The proposed architecture follows a streaming pipeline:

1. Audio input is captured from a live call or uploaded file.
2. The audio is preprocessed and segmented into chunks.
3. The deepfake branch analyzes the audio for synthetic voice patterns.
4. The ASR branch transcribes the speech.
5. The scam-intent branch analyzes the transcript text.
6. A risk-fusion engine computes the overall risk score.
7. The backend sends alerts and results to the dashboard.

### 4.2 Design

#### 4.2.1 Data Flow Diagram

The data flow is as follows:

Audio Input -> Preprocessing -> Deepfake Detection
                                -> ASR -> Scam Analysis
                                -> Risk Fusion -> Alerts and Dashboard

#### 4.2.2 Use Case Diagram

Key actors in the system are:

- Call Participant
- Security Analyst

Use cases include:

- analyze incoming call audio,
- detect deepfake voice,
- transcribe conversation,
- detect suspicious intent,
- generate alerts,
- view dashboard statistics and reports.

---

## 5. IMPLEMENTATION METHODOLOGY

### 5.1 Audio Preprocessing

The audio preprocessing pipeline includes:

- loading audio from file or streaming source,
- resampling to standard sample rate,
- normalization,
- trimming silence,
- noise reduction,
- Mel spectrogram extraction.

This step makes the input suitable for deepfake and ASR models.

### 5.2 Deepfake Detection Model

The deepfake detection component uses a CNN + GRU-based architecture similar to RawNet2-style design. It works on waveform or spectrogram features and classifies whether the voice is authentic or synthetic.

### 5.3 ASR and Transcript Generation

The ASR module uses OpenAI Whisper to convert spoken audio into transcript text. This transcript acts as the semantic signal for scam analysis.

### 5.4 Scam Intent Analysis

The scam detector examines transcript text using:

- suspicious keyword rules,
- financial and urgency indicators,
- social-engineering patterns,
- transformation-based NLP scoring.

The output is a scam probability score.

### 5.5 Risk Fusion Engine

The risk engine combines:

- deepfake probability,
- scam probability,
- user-configurable thresholds.

A weighted score is generated and classified as LOW, MEDIUM, or HIGH.

### 5.6 Backend and Dashboard Integration

The project uses:

- FastAPI for backend APIs,
- WebSocket support for streaming audio,
- SQLite for logging and result storage,
- Streamlit dashboard for upload-based and live-call monitoring simulation.

---

## 6. RESULTS AND EVALUATION

### 6.1 Experimental Setup

The prototype was tested using:

- audio upload workflows,
- demo sample audio files,
- backend API testing,
- live demo streaming simulation,
- dashboard result validation.

### 6.2 Performance Metrics

The system typically evaluates:

- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC
- Processing latency
- Risk classification effectiveness

### 6.3 Sample Output Analysis

The system produces outputs such as:

- deepfake probability,
- scam probability,
- transcript,
- overall risk score,
- risk category,
- alert recommendation,
- processing time.

### 6.4 Observations

The prototype is effective as an academic proof-of-concept because it successfully combines:

- audio authenticity analysis,
- voice-to-text conversion,
- scam-intent detection,
- real-time dashboard alerting.

The model is lightweight and demonstrable, although full deployment would require a large training dataset and more robust live-call infrastructure.

---

## 7. CONCLUSION AND FUTURE WORK

This project demonstrates a practical framework for real-time deepfake voice detection and scam-intent analysis. The implemented solution provides a strong foundation for voice fraud protection in live communication systems. It combines deep learning, ASR, NLP, and alert-driven backend logic into a single system that is suitable for academic demonstration and future extension.

Future enhancements may include:

- real WebRTC call integration,
- larger anti-spoofing training datasets,
- multilingual ASR support,
- better real-time latency optimization,
- improved visualization for analyst decision-making,
- deployment-ready secure architecture for production use.

---

## 8. REFERENCES

[1] H. Tak, J. Patino, M. Todisco, A. Nautsch, N. Evans, and A. Larcher, “End-to-end anti-spoofing with RawNet2,” in Proc. IEEE ICASSP, 2021, pp. 6369–6373.

[2] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” arXiv:2212.04356, 2022.

[3] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “RoBERTa: A robustly optimized BERT pretraining approach,” arXiv:1907.11692, 2019.

[4] FastAPI Documentation, https://fastapi.tiangolo.com/

[5] Streamlit Documentation, https://docs.streamlit.io/

[6] Whisper Model Documentation, https://github.com/openai/whisper

---

## FINAL NOTE

This project offers a complete academic prototype for real-time deepfake voice interception and audio authentication. It is suitable for demonstration in a final-year project review and can be extended into a more production-ready voice security system.
