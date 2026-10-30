@echo off
rem Creates .venv-transcribe in the project root and installs everything needed for transcribe_user.py
cd /d "%~dp0\.."
python -m venv .venv-transcribe
if errorlevel 1 (echo Could not create the virtual environment. Is Python 3.10-3.12 installed? & exit /b 1)
call .venv-transcribe\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r scripts\requirements-transcribe.txt
if errorlevel 1 (echo Install failed. & exit /b 1)
where ffmpeg >nul 2>nul || echo WARNING: ffmpeg not found. Install it with: winget install ffmpeg
echo.
echo Done. Next time, activate with:  .venv-transcribe\Scripts\activate
echo Then run:                        python scripts\transcribe_user.py ^<username^>
