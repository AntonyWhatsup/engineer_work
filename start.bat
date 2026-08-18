@echo off
setlocal
python "%~dp0run_app.py" %*
exit /b %errorlevel%
