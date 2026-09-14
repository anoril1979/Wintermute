@echo off
cls
REM === Lancement du serveur
REM
venv\Scripts\activate
REM
REM === Lancement de Wintermute
REM
START "Fast API" /b /wait fastapi run --reload
REM