:: Copyright 2020 - 2021 MONAI Consortium
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::     http://www.apache.org/licenses/LICENSE-2.0
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.

@echo off

set app_dir=%1
set study_dir=%2
set method=%3
set request=%4
set multi_gpu=%5
set gpus=%6

set COUNT=
set PATH=%app_dir%\bin;%PATH%

echo Virtual Env: %VIRTUAL_ENV%
if exist %app_dir%\requirements.txt.invalid (
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

        if "%method%" == "train" if "%multi_gpu%" == "true" (
            set NVIDIA_VISIBLE_DEVICES=%gpus%
            for /f "tokens=*" %%i in ('python -c "import torch; print(torch.cuda.device_count())"') do (
              set num_gpus=%%i
              goto :train1
            )

            :train1
            if "%num_gpus%" gtr "1" (
                python -m torch.distributed.launch \
                --nproc_per_node="${num_gpus}" \
                --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=1234 \
                -m monailabel.interfaces.utils.app -a %app_dir% -s %study_dir% -m %method% -r %request% --multi_gpu
            ) else (
                python -m monailabel.interfaces.utils.app -a %app_dir% -s %study_dir% -m %method% -r %request%
            )
        ) else (
            python -m monailabel.interfaces.utils.app -a %app_dir% -s %study_dir% -m %method% -r %request%
        )

        deactivate
        @exit
    ) else (
        echo Do nothing as no valid items to install
        if "%method%" == "train" if "%multi_gpu%" == "true" (
            set NVIDIA_VISIBLE_DEVICES=%gpus%
            for /f "tokens=*" %%i in ('python -c "import torch; print(torch.cuda.device_count())"') do (
              set num_gpus=%%i
              goto :train2
            )

            :train2
            if "%num_gpus%" gtr "1" (
                python -m torch.distributed.launch \
                --nproc_per_node="${num_gpus}" \
                --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=1234 \
                -m monailabel.interfaces.utils.app -a %app_dir% -s %study_dir% -m %method% -r %request% --multi_gpu
            ) else (
                python -m monailabel.interfaces.utils.app -a %app_dir% -s %study_dir% -m %method% -r %request%
            )
        ) else (
            python -m monailabel.interfaces.utils.app -a %app_dir% -s %study_dir% -m %method% -r %request%
        )
        @exit
    )
) else (
        if "%method%" == "train" if "%multi_gpu%" == "true" (
            set NVIDIA_VISIBLE_DEVICES=%gpus%
            for /f "tokens=*" %%i in ('python -c "import torch; print(torch.cuda.device_count())"') do (
              set num_gpus=%%i
              goto :train3
            )

            :train3
            if "%num_gpus%" gtr "1" (
                python -m torch.distributed.launch \
                --nproc_per_node="${num_gpus}" \
                --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=1234 \
                -m monailabel.interfaces.utils.app -a %app_dir% -s %study_dir% -m %method% -r %request% --multi_gpu
            ) else (
                python -m monailabel.interfaces.utils.app -a %app_dir% -s %study_dir% -m %method% -r %request%
            )
        ) else (
            python -m monailabel.interfaces.utils.app -a %app_dir% -s %study_dir% -m %method% -r %request%
        )
    @exit
)
