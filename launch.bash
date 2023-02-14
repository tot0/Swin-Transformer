#!/bin/bash

set -e
set -x

env

if [[ $LOCAL_RANK == "0" ]]
then
    cd kernels/window_process/
    python setup.py install
    cd ../../
    touch "user_logs/installed_$AZUREML_CR_NODE_RANK"
    pip list
else
    while true
    do
        if [[ -e "user_logs/installed_$AZUREML_CR_NODE_RANK" ]]
        then
            echo "[`date`] Kernels have been installed."
            break
        fi
        echo "[`date`] Waiting for kernels to be installed..."
        sleep 10
    done
fi

export PYTHONUNBUFFERED=1
exec python main.py "$@"
