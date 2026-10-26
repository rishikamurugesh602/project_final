"""
NLP-based scam/fraud intent detection.
Uses transformer-based classification with fallback to rule-based detection.
"""
import torch
import numpy as np
from typing import Optional, List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.utils.config import config
from app.utils.logging import logger


class ScamDetector:
    """
    Scam detection using NLP.
    
    Combines:
    1. Transformer-based zero-shot classification
    2. Rule-based suspicious indicator detection
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
        
        try:
            # Try to load a lightweight transformer model
            model_name = "facebook/bart-large-mnli"
            logger.info(f"Loading transformer model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            self.model_loaded = True
            logger.info("Transformer model loaded successfully")
        except Exception as e:
            logger.warning(
                f"Could not load transformer model: {str(e)}. "
                "Using rule-based detection only."
            )
            self.model_loaded = False
    
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
        
        # Weight different categories
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
            # Each keyword in a category increases the score
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
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Predict
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                
                # For BART-MNLI: logits are [contradiction, neutral, entailment]
                # We map: high entailment confidence -> scam
                scam_prob = probabilities[0, 2].item()  # Entailment
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
                "detection_method": "rule-based"
            }
        
        # Detect indicators
        indicators = self.detect_suspicious_indicators(text)
        indicator_list = []
        for category, keywords in indicators.items():
            indicator_list.append(f"{category}: {', '.join(keywords)}")
        
        # Calculate rule-based score
        rule_score = self.calculate_risk_score_from_indicators(indicators)
        
        # Get transformer score if available
        transformer_result = self.predict_transformer(text)
        transformer_score = transformer_result.get("scam_probability", 0.5)
        
        # Combine scores (weighted average)
        if self.model_loaded:
            combined_score = 0.6 * transformer_score + 0.4 * rule_score
            detection_method = "transformer + rules"
        else:
            combined_score = rule_score
            detection_method = "rule-based"
        
        # Determine risk category
        if combined_score < config.SCAM_THRESHOLD:
            risk_category = "LOW"
        elif combined_score < (config.SCAM_THRESHOLD + 0.3):
            risk_category = "MEDIUM"
        else:
            risk_category = "HIGH"
        
        return {
            "scam_probability": min(max(combined_score, 0.0), 1.0),
            "risk_category": risk_category,
            "indicators": indicator_list,
            "detection_method": detection_method,
            "rule_score": rule_score,
            "transformer_score": transformer_score if self.model_loaded else None,
            "confidence": transformer_result.get("confidence", 0.0) if self.model_loaded else 0.0
        }
