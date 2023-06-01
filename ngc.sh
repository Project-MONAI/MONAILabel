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

./monailabel/scripts/monailabel start_server -a sample-apps/monaibundle/ -s /workspace/Datasets/Task09_Spleen -c models spleen_ct_segmentation
