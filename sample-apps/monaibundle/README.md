# Integration with MONAI Bundle

### Overview
By default models are picked from https://github.com/Project-MONAI/model-zoo/blob/dev/models/model_info.json

```commandline
monailabel start_server -a sample-apps/monaibundle/ \
  -s ~/Datasets/Radiology/ \
  -c models spleen_ct_segmentation_v0.1.0

monailabel start_server -a sample-apps/monaibundle/ \
  -s ~/Datasets/Radiology/ \
  -c models all
 
# you can pass remove zoo info
monailabel start_server -a sample-apps/monaibundle/ \
  -s ~/Datasets/Radiology/ \
  -c models all \
  -c zoo_info https://raw.githubusercontent.com/Project-MONAI/model-zoo/54-auto-update-model-info/models/model_info.json
```
