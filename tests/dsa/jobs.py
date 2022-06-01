import json
import os
import subprocess
import sys
import time

import girder_client

apiUrl = 'http://localhost:8080/api/v1'
apiToken = 'mSQ5zHJk9j3FIi7MSM4LydSub0FLqm6eyqok399v'
inputImageFileFolder = '62950aadeef3938614e0ec0a'
outputFolder = '62979231f66bce86f67f359f'
monaiServerURI = 'http://10.117.23.243:8000'

nuclei_detection_path = "slicer_cli_web/dsarchive_histomicstk_latest/NucleiDetection/run"
monailabel_annotation_path = "slicer_cli_web/projectmonai_monailabel-dsa_latest/MONAILabelAnnotation/run"

gc = girder_client.GirderClient(apiUrl=apiUrl)
gc.authenticate(apiKey=apiToken)

data = gc.get("item", parameters={"folderId": inputImageFileFolder, "limit": 0})
image_ids = {d["name"]: d["largeImage"]["fileId"] for d in data if d.get("largeImage")}
print(json.dumps(image_ids, indent=2))

images = [
    'TCGA-06-0130-01Z-00-DX1.0391b65f-4e1d-4444-abb0-e5804606d461.svs',
    'TCGA-AA-3663-01Z-00-DX1.9AEDC003-2062-4876-8993-A5CEE4DDE1A9.svs',
    'TCGA-HC-7080-01Z-00-DX1.c979be6a-e7c9-4840-8555-6f34499dd2bf.svs',
    'TCGA-06-0138-01Z-00-DX3.2767efef-7d5f-40ff-9b36-5329d0fa6829.svs',
    'TCGA-28-1746-01Z-00-DX1.06f187d2-b5e8-4b37-bb23-a707d0059944.svs',
    'TCGA-DU-7010-01Z-00-DX1.542F36CC-9685-4780-94EB-B664CECFF09D.svs',
    'TCGA-EM-A2CJ-01Z-00-DX1.D6F4716C-D6C7-4087-9B17-E1D89A3EEA8F.svs',
    'TCGA-27-1831-01Z-00-DX4.b8a6fef5-9ba3-40b4-b32a-31485dbaa153.svs',
    'TCGA-AP-A0LT-01Z-00-DX1.74C269EA-3118-4E65-AAFE-C1D186EAC207.svs',
    'TCGA-HU-8244-01Z-00-DX1.EF15C805-A823-46EA-B737-2EC4A8C5C278.svs',
    'TCGA-06-0195-01Z-00-DX2.5327662a-89b0-4297-ac6e-7af80f06cb3a.svs',
    'TCGA-HT-7620-01Z-00-DX5.E88271BE-B362-4F17-96DF-31E421AA3143.svs',
    'TCGA-FF-8047-01Z-00-DX1.75aa745c-bbe3-4869-a37b-c18ee50c14d5.svs',
    'TCGA-BB-A6UO-01Z-00-DX1.11D049DC-EFC3-47EB-B390-A694BFD304A2.svs',
    'TCGA-85-8072-01Z-00-DX1.3a0ad5a6-c93e-428c-94e7-809ceaf01ef1.svs',
    'TCGA-V4-A9E7-01Z-00-DX1.465EFC95-3B6C-4836-A8BC-0A4F0BBFA601.svs',
    'TCGA-C5-A3HD-01Z-00-DX1.11EECACD-371A-4B16-A21A-8E2A2258D3A9.svs',
    'TCGA-OR-A5J6-01Z-00-DX1.C3F415F4-B679-433F-B8C2-33ED940272FB.svs',
    'TCGA-26-1799-01Z-00-DX1.630B7217-0B01-4CDD-8ABF-0EC4CF293476.svs',
    'TCGA-V1-A8WV-01Z-00-DX1.1419FDEA-BA02-42C4-9FF0-6F1F284BC6F3.svs',
    'TCGA-AX-A1C4-01Z-00-DX1.237A4C5C-E87E-4904-83F2-B76196A247F0.svs',
    'TCGA-DX-A6YR-01Z-00-DX1.8329CE17-C02B-4C56-8D02-54F40D95D624.svs',
    'TCGA-DJ-A2PP-01Z-00-DX1.5BC2A5F2-1918-44E9-9544-1972974BA7BC.svs',
    'TCGA-D8-A1JK-01Z-00-DX1.3190C919-A403-460D-9F6C-D2AB5FD3FD05.svs',
    'TCGA-50-5068-01Z-00-DX2.0492A5C6-09CB-424B-BE20-10A1CBEA2E57.svs',
    'TCGA-T7-A92I-01Z-00-DX1.3B036C1D-F8A7-475F-9830-C0972AD3889F.svs',
    'TCGA-5N-A9KM-01Z-00-DX1.5197F750-D17F-459B-B74D-846F5F50F7B7.svs',
    'TCGA-P3-A6T4-01Z-00-DX1.5DC1C4B4-7BB2-44AE-8D7A-FFFA3CB4BE63.svs',
    'TCGA-T3-A92N-01Z-00-DX2.A08786DD-AF48-4551-BF71-E41C371C97C7.svs',
    'TCGA-OL-A6VO-01Z-00-DX1.291D54D6-EBAF-4622-BD42-97AA5997F014.svs',
    'TCGA-OL-A66J-01Z-00-DX1.661F7F70-E4D4-4875-B8C4-556F7927F3BA.svs',
    # 'TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs',
]

jobs = [{
    'path': monailabel_annotation_path,
    'parameters': {
        # 'inputImageFile_folder': inputImageFileFolder,
        'inputImageFile': '.*',
        'server': monaiServerURI,
        'outputAnnotationFile_folder': outputFolder,
        'outputAnnotationFile': 'MONAILabelAnnotation.anot',
        'min_poly_area': 80,
        'analysis_tile_size': 1024,
        'extra_params': '{}',
    },
    'variations': [{
        'skip': False,
        'model_name': 'segmentation_nuclei',
        'analysis_roi': '[10000,10000,1024,1024]',
    }, {
        'skip': False,
        'model_name': 'segmentation_nuclei',
        'analysis_roi': '[10000,10000,4096,4096]',
    }, {
        'skip': False,
        'min_fgnd_frac': '0.25',
        'model_name': 'segmentation_nuclei',
        'analysis_roi': '[-1,-1,-1,-1]',
    }, {
        'skip': False,
        'model_name': 'deepedit_nuclei',
        'analysis_roi': '[10000,10000,1024,1024]',
    }, {
        'skip': False,
        'model_name': 'deepedit_nuclei',
        'analysis_roi': '[10000,10000,4096,4096]',
    }, {
        'skip': False,
        'min_fgnd_frac': '0.25',
        'model_name': 'deepedit_nuclei',
        'analysis_roi': '[-1,-1,-1,-1]',
    }],
}, {
    'path': nuclei_detection_path,
    'parameters': {
        # 'inputImageFile_folder': inputImageFileFolder,
        'inputImageFile': '.*',
        'outputNucleiAnnotationFile_folder': outputFolder,
        'outputNucleiAnnotationFile': 'NucleiDetection.anot',
        'min_fgnd_frac': '0.25',
        'min_nucleus_area': 80,
        'analysis_tile_size': 1024,
        'analysis_mag': 0,
    },
    'variations': [{
        'skip': False,
        'min_fgnd_frac': '0.25',
        'analysis_roi': '[10000,10000,1024,1024]',
    }, {
        'skip': False,
        'analysis_roi': '[10000,10000,4096,4096]',
    }, {
        'skip': False,
        'analysis_roi': '[-1,-1,-1,-1]',
    }],
}]


def run_command(command, args=None):
    cmd = [command]
    if args:
        args = [str(a) for a in args]
        cmd.extend(args)

    print("Running Command:: {}".format(" ".join(cmd)))
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        env=os.environ.copy(),
    )

    while process.poll() is None:
        line = process.stdout.readline()
        line = line.rstrip()
        if line:
            print(line)

    print(f"Return code: {process.returncode}")
    process.stdout.close()
    return process.returncode


for image in images:
    for jobRecord in jobs:
        for vari in jobRecord['variations']:
            if vari.get('skip'):
                continue

            model_name = vari.get('model_name', 'NucleiDetection')

            param = jobRecord['parameters'].copy()
            param.update({k: v for k, v in vari.items() if 'skip' not in k})
            param['inputImageFile'] = image_ids[image]
            param['outputAnnotationFile'] = f"{model_name}_{vari['analysis_roi']}_" + image.replace(".svs", ".anot")

            sys.stdout.write(f"Running {image} => {model_name} => {vari['analysis_roi']} ")

            job = gc.post(jobRecord['path'], parameters=param)
            while True:
                job = gc.get('/job/%s' % job['_id'])
                if job['status'] not in [0, 1, 2]:
                    break
                sys.stdout.write('.')
                sys.stdout.flush()
                time.sleep(1)
            sys.stdout.write('\n')
    sys.stdout.write('\n')
