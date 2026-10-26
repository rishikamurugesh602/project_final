#!/usr/bin/env python
"""
Setup script to install all project dependencies.
Run with: python setup_install.py
"""

import subprocess
import sys
import os

# List of packages to install in stages
PACKAGES = {
    "Core Web": ["fastapi==0.104.1", "uvicorn==0.24.0", "starlette"],
    "Dashboard": ["streamlit", "plotly", "pandas"],
    "Audio": ["librosa", "soundfile", "scipy"],
    "ML/AI": ["torch", "numpy", "transformers", "scikit-learn"],
    "Database": ["sqlalchemy", "pydantic"],
    "Utilities": ["python-dotenv", "requests", "pytest", "openai-whisper"],
}

def install_package_group(group_name, packages):
    """Install a group of packages with error handling."""
    print(f"\n{'='*60}")
    print(f"Installing {group_name}...")
    print(f"{'='*60}")
    
    for package in packages:
        print(f"  → {package}...", end=" ", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", package],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                print("✓")
            else:
                print("⚠ (warning, continuing...)")
                if result.stderr:
                    print(f"    Error: {result.stderr[:100]}")
        except subprocess.TimeoutExpired:
            print("✗ (timeout, skipping...)")
        except Exception as e:
            print(f"✗ ({e})")

def verify_installation():
    """Verify that core packages are installed."""
    print(f"\n{'='*60}")
    print("Verifying Installation...")
    print(f"{'='*60}\n")
    
    packages_to_check = [
        "fastapi",
        "uvicorn",
        "streamlit",
        "librosa",
        "torch",
        "transformers",
        "sqlalchemy",
        "pydantic"
    ]
    
    all_ok = True
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (NOT installed)")
            all_ok = False
    
    return all_ok

def main():
    print("\n" + "="*60)
    print("DEEPFAKE VOICE INTERCEPTOR - DEPENDENCY INSTALLER")
    print("="*60)
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    
    # Install each group
    for group_name, packages in PACKAGES.items():
        install_package_group(group_name, packages)
    
    # Verify
    success = verify_installation()
    
    print(f"\n{'='*60}")
    if success:
        print("✓ Installation Complete!")
        print("\nNext steps:")
        print("  1. Initialize database:")
        print("     python -c \"from app.database.database import init_database; init_database()\"")
        print("\n  2. Start backend (Terminal 1):")
        print("     uvicorn app.main:app --reload")
        print("\n  3. Start dashboard (Terminal 2):")
        print("     streamlit run dashboard/streamlit_app.py")
        print("\n  4. Open browser to: http://localhost:8501")
    else:
        print("⚠ Some packages failed to install.")
        print("Try running manually:")
        print("  pip install -r requirements_minimal.txt --no-cache-dir")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
