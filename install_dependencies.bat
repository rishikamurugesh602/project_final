@echo off
REM Setup script for Deepfake Voice Interceptor project

echo.
echo ============================================================
echo Installing Python Dependencies for Deepfake Interceptor
echo ============================================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    exit /b 1
)

echo [1/4] Installing core web frameworks...
pip install -q fastapi uvicorn streamlit
if errorlevel 1 echo WARNING: Web frameworks had issues

echo [2/4] Installing audio processing libraries...
pip install -q librosa soundfile scipy
if errorlevel 1 echo WARNING: Audio libraries had issues

echo [3/4] Installing ML/AI libraries...
pip install -q torch numpy pandas transformers scikit-learn
if errorlevel 1 echo WARNING: ML libraries had issues

echo [4/4] Installing utilities...
pip install -q python-dotenv pydantic sqlalchemy requests plotly pytest openai-whisper
if errorlevel 1 echo WARNING: Utilities had issues

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo Verifying installation...
pip list | findstr "fastapi uvicorn streamlit librosa torch"
echo.
echo Next steps:
echo   1. Initialize database: python -c "from app.database.database import init_database; init_database()"
echo   2. Start backend: uvicorn app.main:app --reload
echo   3. Start dashboard: streamlit run dashboard/streamlit_app.py
echo.
pause
