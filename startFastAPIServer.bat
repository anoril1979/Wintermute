@echo off
cls
REM === Lancement du serveur
REM
venv\Scripts\activate
REM
REM === Lancement de Wintermute
REM
fastapi run --reload
REM