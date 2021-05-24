@echo off

set app_dir=%1
set study_dir=%2
set user_command=%3
set train_request=%4

if exist %app_dir%\requirements.txt (
    if not exist %app_dir%\.venv (
        echo Creating VENV
        python -m venv --system-site-packages %app_dir%\.venv
    )
	
    echo "+++++++++++++++++ Installing PIP requirements"
    python -m pip install --upgrade pip
    python -m pip install -r %app_dir%\requirements.txt

    %app_dir%\.venv\Scripts\activate

    set PYTHONPATH=%PYTHONPATH%;%app_dir%
    echo "Using PYTHONPATH:: %PYTHONPATH%"
    python -m monailabel.utils.others.app_utils -a %app_dir% -s %study_dir% %user_command% -r %train_request%
	deactivate
) else (
    python -m monailabel.utils.others.app_utils -a %app_dir% -s %study_dir% %user_command% -r %train_request%
)
