import sys


def print_config(file=sys.stdout):
    from collections import OrderedDict

    import numpy as np
    import torch

    output = OrderedDict()
    output["MONAILabel"] = "0.1"
    output["Numpy"] = np.version.full_version
    output["Pytorch"] = torch.__version__

    for k, v in output.items():
        print(f"{k} version: {v}", file=file, flush=True)
    print("MONAILabel rev id: 0.1")
