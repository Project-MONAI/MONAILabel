#!/bin/bash

# Starting server

echo "Starting MONAILabel server..."

PATH_TO_STUDIES=$1

echo "Using path to studies:  ${PATH_TO_STUDIES}"

../../monailabel/scripts/monailabel start_server -a ./ -s ${PATH_TO_STUDIES} --conf models segmentation_brats &

# Waiting for server to be up and running

wait_time=0
server_is_up=0
start_time_out=180

function check_server_running() {
  local code=$(curl --write-out "%{http_code}\n" -s "http://127.0.0.1:${MONAILABEL_SERVER_PORT:-8000}/" --output /dev/null)
  echo ${code}
}

while [[ $wait_time -le ${start_time_out} ]]; do
  if [ "$(check_server_running)" == "200" ]; then
    server_is_up=1
    break
  fi
  sleep 5
  wait_time=$((wait_time + 5))
  echo "Waiting for MONAILabel to be up and running..."
done

echo ""

if [ "$server_is_up" == "1" ]; then
  echo "MONAILabel server is up and running."
else
  echo "Failed to start MONAILabel server. Exiting..."
  exit 1
fi

# Running batch inference

echo "Running batch inference ..."

curl -X 'POST' \
  'http://127.0.0.1:8000/batch/infer/segmentation_brats?images=unlabeled&run_sync=false' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{}'
