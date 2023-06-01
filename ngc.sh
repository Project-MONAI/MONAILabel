#! /bin/bash

echo $HOSTNAME
echo "----------------------------------"
echo ENV VARIABLES....
echo "----------------------------------"
printenv
echo "----------------------------------"

echo ""
echo ""

echo "----------------------------------"
echo HOST IP
echo "----------------------------------"
hostname
hostname -I
echo "----------------------------------"

cd /workspace/Projects/MONAILabel
pip install -r requirements.txt

export MONAI_ZOO_AUTH_TOKEN=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
./monailabel/scripts/monailabel start_server -a sample-apps/monaibundle/ -s /workspace/Datasets/Task09_Spleen -c models spleen_ct_segmentation
