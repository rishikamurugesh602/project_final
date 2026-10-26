"""
Call and session management for browser-to-browser WebRTC audio calls.
Handles signaling, peer registration, and participant consent tracking.
"""
import time
from typing import Dict, Optional
from fastapi import WebSocket
from app.utils.logging import logger


class CallParticipant:
    """Represents a participant in a WebRTC call session."""
    def __init__(self, client_id: str, websocket: WebSocket, role: str = "peer"):
        self.client_id = client_id
        self.websocket = websocket
        self.role = role
        self.consent_given = False
        self.joined_at = time.time()


class CallSession:
    """Represents an active WebRTC call room between two browser participants."""
    def __init__(self, call_id: str):
        self.call_id = call_id
        self.created_at = time.time()
        self.participants: Dict[str, CallParticipant] = {}
        self.call_active = False

    def add_participant(self, client_id: str, websocket: WebSocket) -> CallParticipant:
        role = "caller" if len(self.participants) == 0 else "callee"
        participant = CallParticipant(client_id=client_id, websocket=websocket, role=role)
        self.participants[client_id] = participant
        logger.info(f"Participant {client_id} ({role}) joined call {self.call_id}")
        return participant

    def remove_participant(self, client_id: str):
        if client_id in self.participants:
            del self.participants[client_id]
            logger.info(f"Participant {client_id} removed from call {self.call_id}")

    def set_consent(self, client_id: str, consent: bool) -> bool:
        if client_id in self.participants:
            self.participants[client_id].consent_given = consent
            logger.info(f"Participant {client_id} in call {self.call_id} consent set to: {consent}")
            return True
        return False

    def both_consented(self) -> bool:
        if len(self.participants) < 2:
            return False
        return all(p.consent_given for p in self.participants.values())

    def get_other_participant(self, client_id: str) -> Optional[CallParticipant]:
        for cid, p in self.participants.items():
            if cid != client_id:
                return p
        return None

    def get_status(self) -> dict:
        return {
            "call_id": self.call_id,
            "participant_count": len(self.participants),
            "participants": {
                cid: {
                    "role": p.role,
                    "consent_given": p.consent_given,
                    "joined_at": p.joined_at
                }
                for cid, p in self.participants.items()
            },
            "both_consented": self.both_consented(),
            "active": self.call_active
        }


class CallSessionManager:
    """Manages active WebRTC call sessions."""
    def __init__(self):
        self.sessions: Dict[str, CallSession] = {}

    def get_or_create_session(self, call_id: str) -> CallSession:
        if call_id not in self.sessions:
            self.sessions[call_id] = CallSession(call_id)
        return self.sessions[call_id]

    def remove_session_if_empty(self, call_id: str):
        if call_id in self.sessions and len(self.sessions[call_id].participants) == 0:
            del self.sessions[call_id]
            logger.info(f"Call session {call_id} removed (empty)")


call_session_manager = CallSessionManager()
