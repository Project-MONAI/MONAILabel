#! /bin/bash

mkdir -p /workspace/Projects/MONAILabel/logs
sh /workspace/Projects/MONAILabel/ngc.sh 2>&1 | tee /workspace/Projects/MONAILabel/logs/$RANDOM.log
