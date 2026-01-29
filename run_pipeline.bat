@echo off
call .\.venv\Scripts\activate.bat
python ingest\fetch_and_load.py
echo Done!
pause
