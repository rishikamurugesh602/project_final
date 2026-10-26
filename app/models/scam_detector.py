"""
NLP-based scam/fraud intent detection.
Uses transformer-based classification with fallback to rule-based detection.
"""
import os
import torch
import numpy as np
from typing import Optional, List, Dict
from app.utils.config import config
from app.utils.logging import logger


class ScamDetector:
    """
    Scam detection using NLP.
    
    Combines:
    1. Transformer-based zero-shot classification (when available)
    2. Rule-based suspicious indicator detection (fast, deterministic fallback)
    """
    
    # Suspicious keywords and patterns
    SUSPICIOUS_KEYWORDS = {
        "password": ["password", "pwd", "passphrase", "pin"],
        "banking": ["bank", "account", "card", "credit card", "debit card"],
        "otp": ["otp", "code", "verification code", "one-time", "one time"],
        "money_transfer": ["transfer", "wire", "send money", "payment", "transaction"],
        "urgency": ["urgent", "immediately", "asap", "right now", "quickly", "don't delay"],
        "threat": ["suspend", "block", "freeze", "close", "disabled", "locked"],
        "impersonation": ["i'm calling from", "this is", "on behalf of", "representative of"],
        "credential_harvest": ["confirm", "verify", "validate", "update", "reset"],
        "financial_request": ["payment", "invoice", "bill", "fee", "charge"]
    }
    
    def __init__(self, device: str = config.MODEL_DEVICE):
        """
        Initialize scam detector.
        
        Args:
            device: Device to use (cpu/cuda)
        """
        self.device = device
        self.tokenizer = None
        self.model = None
        self.model_loaded = False
        self.status_message = "Rule-based only (transformer weights not provided)"
        
        model_name = "facebook/bart-large-mnli"
        local_path = config.SCAM_MODEL_PATH if os.path.exists(config.SCAM_MODEL_PATH) else None
        
        # Only attempt to import/load transformers if local weights exist or online download allowed
        if local_path is not None:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                logger.info(f"Loading transformer model from local path: {local_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(local_path, local_files_only=True)
                self.model.to(device)
                self.model.eval()
                self.model_loaded = True
                self.status_message = f"Transformer model loaded from {local_path}"
            except Exception as e:
                logger.warning(f"Failed to load transformer from {local_path}: {e}")
                self.model_loaded = False
        elif not config.SCAM_LOCAL_ONLY:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                logger.info(f"Attempting download for: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.to(device)
                self.model.eval()
                self.model_loaded = True
                self.status_message = "Transformer model downloaded and loaded"
            except Exception as dl_err:
                self.model_loaded = False
                self.status_message = f"Transformer download unavailable ({dl_err}); rule-based active"
        else:
            self.model_loaded = False
            self.status_message = "Rule-based detection active (transformer offline / not cached)"
    
    def detect_suspicious_indicators(self, text: str) -> Dict[str, List[str]]:
        """
        Detect suspicious indicators in text using rules.
        
        Args:
            text: Input text/transcript
        
        Returns:
            Dictionary of detected indicators by category
        """
        text_lower = text.lower()
        detected = {}
        
        for category, keywords in self.SUSPICIOUS_KEYWORDS.items():
            found = [kw for kw in keywords if kw in text_lower]
            if found:
                detected[category] = found
        
        return detected
    
    def calculate_risk_score_from_indicators(
        self,
        indicators: Dict[str, List[str]]
    ) -> float:
        """
        Calculate risk score based on detected indicators.
        
        Args:
            indicators: Dictionary of detected indicators
        
        Returns:
            Risk score between 0 and 1
        """
        if not indicators:
            return 0.0
        
        category_weights = {
            "password": 0.25,
            "banking": 0.25,
            "otp": 0.25,
            "money_transfer": 0.30,
            "urgency": 0.15,
            "threat": 0.20,
            "impersonation": 0.20,
            "credential_harvest": 0.25,
            "financial_request": 0.20
        }
        
        total_weight = 0.0
        score = 0.0
        
        for category, keywords in indicators.items():
            weight = category_weights.get(category, 0.1)
            category_score = min(len(keywords) * 0.1, 1.0)
            score += weight * category_score
            total_weight += weight
        
        if total_weight > 0:
            score = score / total_weight
        
        return min(score, 1.0)
    
    def predict_transformer(self, text: str) -> Dict[str, float]:
        """
        Use transformer model for classification.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with predictions
        """
        if not self.model_loaded or not text.strip():
            return {"scam_probability": 0.5, "confidence": 0.0}
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                
                # For BART-MNLI: logits are [contradiction, neutral, entailment]
                scam_prob = probabilities[0, 2].item()
                confidence = torch.max(probabilities[0]).item()
                
                return {
                    "scam_probability": scam_prob,
                    "confidence": confidence
                }
        except Exception as e:
            logger.error(f"Error in transformer prediction: {str(e)}")
            return {"scam_probability": 0.5, "confidence": 0.0}
    
    def predict(self, text: str) -> Dict:
        """
        Predict scam probability for given text.
        
        Args:
            text: Input text/transcript
        
        Returns:
            Dictionary with predictions and detected indicators
        """
        if not text or not text.strip():
            return {
                "scam_probability": 0.0,
                "risk_category": "LOW",
                "indicators": [],
                "detection_method": "rule-based",
                "model_loaded": self.model_loaded,
                "model_status": self.status_message
            }
        
        indicators = self.detect_suspicious_indicators(text)
        indicator_list = []
        for category, keywords in indicators.items():
            indicator_list.append(f"{category}: {', '.join(keywords)}")
        
        rule_score = self.calculate_risk_score_from_indicators(indicators)
        
        if self.model_loaded:
            transformer_result = self.predict_transformer(text)
            transformer_score = transformer_result.get("scam_probability", 0.5)
            combined_score = 0.6 * transformer_score + 0.4 * rule_score
            detection_method = "transformer + rules"
            confidence = transformer_result.get("confidence", 0.0)
        else:
            transformer_score = None
            combined_score = rule_score
            detection_method = "rule-based (transformer weights not loaded)"
            confidence = min(rule_score * 1.2, 1.0)
        
        if combined_score < config.SCAM_THRESHOLD:
            risk_category = "LOW"
        elif combined_score < (config.SCAM_THRESHOLD + 0.3):
            risk_category = "MEDIUM"
        else:
            risk_category = "HIGH"
        
        return {
            "scam_probability": float(min(max(combined_score, 0.0), 1.0)),
            "risk_category": risk_category,
            "indicators": indicator_list,
            "detection_method": detection_method,
            "rule_score": float(rule_score),
            "transformer_score": float(transformer_score) if transformer_score is not None else None,
            "confidence": float(confidence),
            "model_loaded": self.model_loaded,
            "model_status": self.status_message
        }
