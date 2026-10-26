#!/usr/bin/env python
"""
Streamlit dashboard startup script.
Configures Streamlit to work properly with the backend.
"""

import subprocess
import sys
import os

def main():
    print("\n" + "="*60)
    print("DEEPFAKE VOICE INTERCEPTOR - DASHBOARD STARTUP")
    print("="*60)
    print(f"Python: {sys.executable}\n")
    
    # Set Streamlit configuration
    env = os.environ.copy()
    
    # Disable browser auto-open
    env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    env['STREAMLIT_CLIENT_TRACKING_URL_BASE'] = ''
    
    print("Starting Streamlit Dashboard...")
    print("-" * 60)
    print("Dashboard will be available at: http://localhost:8501")
    print("Make sure backend is running first!")
    print("-" * 60 + "\n")
    
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "dashboard/streamlit_app.py",
            "--logger.level=info"
        ]
        
        subprocess.run(cmd, env=env, check=False)
        
    except KeyboardInterrupt:
        print("\n\nDashboard stopped (Ctrl+C pressed)")
    except Exception as e:
        print(f"\nError starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
