#! /bin/bash

mkdir -p /workspace/Projects/MONAILabel/logs
prefix=`echo $HOSTNAME | cut -d'-' -f1,2`

#apt-get update -y
#apt-get install -y iputils-ping

echo "PING:: $prefix-0"
python -c "import socket; print(socket.gethostbyname('$prefix-0'))"
#ping -c 5 "$prefix-0"

echo "PING:: $prefix-1"
python -c "import socket; print(socket.gethostbyname('$prefix-1'))"
#ping -c 5 "$prefix-1"

echo "PING:: launcher-svc-${NGC_JOB_ID}"
python -c "import socket; print(socket.gethostbyname('launcher-svc-${NGC_JOB_ID}'))"

sh /workspace/Projects/MONAILabel/ngc.sh 2>&1 | tee /workspace/Projects/MONAILabel/logs/$HOSTNAME.log
