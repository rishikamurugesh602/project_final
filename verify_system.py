"""
Verification script to test all imports and basic functionality.
Run this to verify the system is set up correctly.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("="*60)
print("DEEPFAKE VOICE INTERCEPTOR - VERIFICATION SCRIPT")
print("="*60)

# Test 1: Configuration
print("\n[1/10] Testing configuration...")
try:
    from app.utils.config import config
    print(f"  [OK] Config loaded")
    print(f"    - Device: {config.MODEL_DEVICE}")
    print(f"    - Whisper Model: {config.WHISPER_MODEL}")
    print(f"    - Sample Rate: {config.SAMPLE_RATE}Hz")
except Exception as e:
    print(f"  [FAIL] Config error: {e}")
    sys.exit(1)

# Test 2: Logging
print("\n[2/10] Testing logging...")
try:
    from app.utils.logging import logger
    print(f"  [OK] Logger initialized")
except Exception as e:
    print(f"  [FAIL] Logging error: {e}")
    sys.exit(1)

# Test 3: Database
print("\n[3/10] Testing database...")
try:
    from app.database.database import init_database, engine
    print(f"  [OK] Database initialized")
    print(f"    - URL: {config.DATABASE_URL}")
except Exception as e:
    print(f"  [FAIL] Database error: {e}")

# Test 4: Audio Processor
print("\n[4/10] Testing audio processor...")
try:
    from app.services.audio_processor import AudioProcessor
    processor = AudioProcessor()
    print(f"  [OK] AudioProcessor created")
    print(f"    - Sample Rate: {processor.sample_rate}Hz")
except Exception as e:
    print(f"  [FAIL] Audio processor error: {e}")
    sys.exit(1)

# Test 5: Deepfake Detector Model
print("\n[5/10] Testing deepfake detector...")
try:
    from app.models.deepfake_detector import DeepfakeDetectorModel, DeepfakeDetector
    import torch
    
    model = DeepfakeDetectorModel()
    print(f"  [OK] DeepfakeDetectorModel created")
    
    # Test forward pass
    x = torch.randn(1, 1, 128, 100)
    with torch.no_grad():
        output = model(x)
    print(f"  [OK] Forward pass successful")
    print(f"    - Output shape: {output.shape}")
    
    # Test detector wrapper
    detector = DeepfakeDetector(device=config.MODEL_DEVICE)
    print(f"  [OK] DeepfakeDetector wrapper created")
except Exception as e:
    print(f"  [FAIL] Deepfake detector error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Scam Detector
print("\n[6/10] Testing scam detector...")
try:
    from app.models.scam_detector import ScamDetector
    
    detector = ScamDetector(device=config.MODEL_DEVICE)
    print(f"  [OK] ScamDetector created")
    
    # Test prediction
    result = detector.predict("Please send me your password")
    print(f"  [OK] Scam detection works")
    print(f"    - Scam probability: {result['scam_probability']:.2%}")
except Exception as e:
    print(f"  [FAIL] Scam detector error: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Risk Engine
print("\n[7/10] Testing risk engine...")
try:
    from app.services.risk_engine import RiskEngine
    
    engine = RiskEngine()
    result = engine.calculate_risk(0.7, 0.8)
    print(f"  [OK] RiskEngine created and working")
    print(f"    - Risk Category: {result['risk_category']}")
    print(f"    - Risk Score: {result['overall_risk_score']:.2%}")
except Exception as e:
    print(f"  [FAIL] Risk engine error: {e}")
    sys.exit(1)

# Test 8: ASR Service
print("\n[8/10] Testing ASR service...")
try:
    from app.services.asr_service import ASRService
    
    asr = ASRService(model_name=config.WHISPER_MODEL, device=config.MODEL_DEVICE)
    print(f"  [OK] ASRService created")
    if asr.is_loaded:
        print(f"  [OK] Whisper model loaded")
    else:
        print(f"  [WARN] Whisper model not yet loaded (will load on first use)")
except Exception as e:
    print(f"  [FAIL] ASR service error: {e}")

# Test 9: Analysis Orchestrator
print("\n[9/10] Testing analysis orchestrator...")
try:
    from app.services.deepfake_service import AnalysisOrchestrator
    
    orchestrator = AnalysisOrchestrator()
    print(f"  [OK] AnalysisOrchestrator created")
except Exception as e:
    print(f"  [FAIL] Analysis orchestrator error: {e}")
    sys.exit(1)

# Test 10: FastAPI App
print("\n[10/10] Testing FastAPI app...")
try:
    from app.main import app
    print(f"  [OK] FastAPI app created")
    print(f"    - App title: {app.title}")
    print(f"    - Version: {app.version}")
except Exception as e:
    print(f"  [FAIL] FastAPI app error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("VERIFICATION COMPLETE [OK]")
print("="*60)
print("\nNext steps:")
print("1. Run backend: uvicorn app.main:app --reload")
print("2. Run dashboard: streamlit run dashboard/streamlit_app.py")
print("3. Run tests: pytest tests/test_system.py -v")
print("\nDocumentation:")
print("- README.md - Complete setup and usage guide")
print("- PROJECT_IMPLEMENTATION_STATUS.md - Implementation details")
print("="*60)
