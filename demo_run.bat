@echo off
echo ================================
echo Multi-Dimensional-Data-Structures - Demo Run
echo ================================
echo.

:: Ενεργοποίηση του Virtual Environment
echo Activating virtual environment...
call .venv\Scripts\activate

:: Εγκατάσταση των απαιτούμενων packages
echo Installing dependencies...
pip install -r requirements.txt

:: Εκτέλεση του κύριου κώδικα
echo Running main script...
python multi_dimensional.ipynb

pause
