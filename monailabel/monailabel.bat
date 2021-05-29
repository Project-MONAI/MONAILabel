@echo off

set DIR=%~dp0
pushd %DIR%\..
set PARENT=%cd%
popd


set BAK_PYTHONPATH=%PYTHONPATH%

set PYTHONPATH=%PARENT%;%PYTHONPATH%
python -m monailabel.main %*
set PYTHONPATH=%BAK_PYTHONPATH%
