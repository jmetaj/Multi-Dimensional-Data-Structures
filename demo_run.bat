@echo off
echo ================================
echo Multi-Dimensional-Data-Structures - Demo Run
echo ================================
echo.

:: Ελέγχουμε αν υπάρχει το Virtual Environment
if not exist ".venv" (
    echo No virtual environment found. Creating one...
    python -m venv .venv
)

:: Ενεργοποίηση του Virtual Environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

::Εγκατάσταση των απαιτούμενων packages
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

:: Μετατροπή του Jupyter Notebook σε Python script
echo Converting Jupyter Notebook to Python script...
jupyter nbconvert --to script multi_dimensional.ipynb

:: Εκτέλεση του κύριου κώδικα
echo Running main script...
python multi_dimensional.py

:: Αναμονή πριν το κλείσιμο
echo.
echo Execution completed! Press any key to exit.
pause
