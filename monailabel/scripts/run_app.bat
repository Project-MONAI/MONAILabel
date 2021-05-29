@echo off

set app_dir=%1
set study_dir=%2
set method=%3
set request=%4

set COUNT=

echo Virtual Env: %VIRTUAL_ENV%
if exist %app_dir%\requirements.txt (
    for /f "tokens=*" %%i in ('findstr /v "#" %app_dir%\requirements.txt ^| findstr "." ^| find /c /v ""') do (
        set COUNT=%%i
        goto :done
    )

    :done
    echo Total App specific Dependencies: %COUNT%

    if "%COUNT%" gtr "0" (
        if not exist %app_dir%\.venv (
            echo Creating VENV for this app
            python -m venv --system-site-packages %app_dir%\.venv
        )

        %app_dir%\.venv\Scripts\activate

        echo +++++++++++++++++ Installing PIP requirements
        python -m pip install --upgrade pip
        python -m pip install -r %app_dir%\requirements.txt

        echo App:: Virtual Env: %VIRTUAL_ENV%
        echo Using PYTHONPATH:: %PYTHONPATH%
        python -c "import os, sys; print(os.path.dirname(sys.executable))"

        python -m monailabel.utils.others.app_utils -a %app_dir% -s %study_dir% -m %method% -r %request%
        deactivate
        @exit
    ) else (
        echo Do nothing as no valid items to install
        python -m monailabel.utils.others.app_utils -a %app_dir% -s %study_dir% -m %method% -r %request%
        @exit
    )
) else (
    python -m monailabel.utils.others.app_utils -a %app_dir% -s %study_dir% -m %method% -r %request%
    @exit
)
