"""
Risk decision engine that combines deepfake and scam probabilities.
"""
from typing import Dict
from app.utils.config import config
from app.utils.logging import logger


class RiskEngine:
    """
    Combines deepfake probability and scam probability to determine overall risk.
    """
    
    def __init__(
        self,
        deepfake_weight: float = config.DEEPFAKE_WEIGHT,
        scam_weight: float = config.SCAM_WEIGHT,
        low_threshold: float = config.RISK_LOW_THRESHOLD,
        medium_threshold: float = config.RISK_MEDIUM_THRESHOLD,
        high_threshold: float = config.RISK_HIGH_THRESHOLD
    ):
        """
        Initialize risk engine.
        
        Args:
            deepfake_weight: Weight for deepfake probability
            scam_weight: Weight for scam probability
            low_threshold: Threshold for LOW risk
            medium_threshold: Threshold for MEDIUM risk
            high_threshold: Threshold for HIGH risk
        """
        # Normalize weights
        total = deepfake_weight + scam_weight
        self.deepfake_weight = deepfake_weight / total
        self.scam_weight = scam_weight / total
        
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        
        # Validate thresholds
        if not (low_threshold < medium_threshold < high_threshold):
            raise ValueError("Thresholds must be in ascending order")
        
        logger.info(
            f"Risk engine initialized: deepfake_weight={self.deepfake_weight:.2f}, "
            f"scam_weight={self.scam_weight:.2f}"
        )
    
    def calculate_risk(
        self,
        deepfake_probability: float,
        scam_probability: float
    ) -> Dict:
        """
        Calculate overall risk score and category.
        
        Args:
            deepfake_probability: Probability of deepfake (0-1)
            scam_probability: Probability of scam (0-1)
        
        Returns:
            Dictionary with risk assessment
        """
        # Validate inputs
        if not (0 <= deepfake_probability <= 1):
            logger.warning(f"Invalid deepfake probability: {deepfake_probability}")
            deepfake_probability = max(0, min(1, deepfake_probability))
        
        if not (0 <= scam_probability <= 1):
            logger.warning(f"Invalid scam probability: {scam_probability}")
            scam_probability = max(0, min(1, scam_probability))
        
        # Calculate weighted risk score
        overall_risk_score = (
            self.deepfake_weight * deepfake_probability +
            self.scam_weight * scam_probability
        )
        
        # Ensure score is in [0, 1]
        overall_risk_score = max(0, min(1, overall_risk_score))
        
        # Determine risk category
        if overall_risk_score <= self.low_threshold:
            risk_category = "LOW"
            severity = 1
        elif overall_risk_score <= self.medium_threshold:
            risk_category = "MEDIUM"
            severity = 2
        else:
            risk_category = "HIGH"
            severity = 3
        
        # Generate reasoning
        reasons = self._generate_reasons(
            deepfake_probability,
            scam_probability,
            risk_category
        )
        
        return {
            "overall_risk_score": float(overall_risk_score),
            "risk_category": risk_category,
            "severity_level": severity,
            "deepfake_probability": float(deepfake_probability),
            "scam_probability": float(scam_probability),
            "deepfake_weight": self.deepfake_weight,
            "scam_weight": self.scam_weight,
            "reasons": reasons,
            "thresholds": {
                "low": self.low_threshold,
                "medium": self.medium_threshold,
                "high": self.high_threshold
            }
        }
    
    def _generate_reasons(
        self,
        deepfake_prob: float,
        scam_prob: float,
        category: str
    ) -> list:
        """
        Generate human-readable reasons for the risk assessment.
        
        Args:
            deepfake_prob: Deepfake probability
            scam_prob: Scam probability
            category: Risk category
        
        Returns:
            List of reason strings
        """
        reasons = []
        
        # Deepfake reasons
        if deepfake_prob > 0.7:
            reasons.append("High AI-generated voice probability (potential deepfake)")
        elif deepfake_prob > 0.5:
            reasons.append("Moderate AI-generated voice probability")
        elif deepfake_prob > 0.3:
            reasons.append("Some suspicious voice characteristics detected")
        else:
            reasons.append("Audio appears to be genuine human voice")
        
        # Scam reasons
        if scam_prob > 0.7:
            reasons.append("High suspicious intent detected in conversation")
        elif scam_prob > 0.5:
            reasons.append("Moderate suspicious intent indicators present")
        elif scam_prob > 0.3:
            reasons.append("Some suspicious conversation patterns detected")
        else:
            reasons.append("No obvious scam intent detected")
        
        # Category-specific messages
        if category == "HIGH":
            reasons.append("Combined factors indicate HIGH risk of fraud/deepfake")
        elif category == "MEDIUM":
            reasons.append("Multiple risk factors present - caution recommended")
        else:
            reasons.append("Risk assessment indicates LOW threat level")
        
        return reasons
    
    def get_recommendation(self, risk_category: str) -> str:
        """
        Get user recommendation based on risk category.
        
        Args:
            risk_category: Risk category (LOW, MEDIUM, HIGH)
        
        Returns:
            Recommendation string
        """
        recommendations = {
            "LOW": "This call appears safe. Proceed normally.",
            "MEDIUM": "Exercise caution. Verify identity of caller independently.",
            "HIGH": "ALERT: High risk detected. Do not share personal/financial information. "
                   "Hang up and call official numbers."
        }
        return recommendations.get(risk_category, "Unknown risk category")
    
    def should_alert(self, risk_category: str) -> bool:
        """
        Determine if an alert should be triggered.
        
        Args:
            risk_category: Risk category
        
        Returns:
            True if alert should be shown
        """
        return risk_category in ["MEDIUM", "HIGH"]
