@echo off
cls
REM === Lancement du serveur
REM
Start "Environment" /b venv\Scripts\activate
REM
REM === Lancement de Wintermute
REM
START "Wintermute" /b fastapi run --reload
REM