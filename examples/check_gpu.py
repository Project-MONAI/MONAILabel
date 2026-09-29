# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check the installed CUDA/PyTorch/MONAI runtime without downloading model weights."""

import platform

import monai
import torch
from monai.networks.nets import UNet


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Check the driver and the installed PyTorch build.")
    device = torch.cuda.get_device_properties(0)
    model = UNet(
        spatial_dims=2, in_channels=3, out_channels=2, channels=(4, 8, 16, 32), strides=(2, 2, 2)
    ).cuda()
    image = torch.randn(1, 3, 32, 32, device="cuda")
    result = model(image)
    result.square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(result).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(f"Architecture: {platform.machine()} · GPU: {device.name}")
    print(f"PyTorch: {torch.__version__} · CUDA: {torch.version.cuda} · MONAI: {monai.__version__}")
    print("GPU tensor operations and U-Net forward/backward passed.")


if __name__ == "__main__":
    main()
