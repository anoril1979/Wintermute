@echo off
cls
REM === Lancement du serveur
REM
Start "Environment" /b venv\Scripts\activate
REM
REM === Vérification de l'environnement
REM
Start "Check" /b /wait py check_setup.py
REM
REM === Lancement de Open WebUI
REM
START "Wintermute" cmd.exe /c startFastAPIServer.bat
START "Open WebUI" cmd.exe /c startOpenWebUI.bat
REM