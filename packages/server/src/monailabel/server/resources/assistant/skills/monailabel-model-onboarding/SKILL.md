---
name: monailabel-model-onboarding
description: Connect a deployed inference endpoint or explain what is needed to integrate an unsupported model architecture.
license: Apache-2.0
compatibility: Requires the MONAI Label coordinator and its registered typed tools.
allowed-tools: open_form
metadata:
  monailabel-context: workspace
---

# Model onboarding

Available: connect a deployed inference endpoint through the model form using a supported adapter; prepare U-Net scratch/fine-tune/continuation learners; derive VISTA3D CT learners. Recipes own geometry/intensity transforms and bounded settings. Credentials use protected forms/environment. An inference endpoint alone does not provide training.

New architectures, transforms and arbitrary repository code require integration work, not coordinator execution. Explain the needed framework/task/runtime, versioned weights and input/output contracts, preprocessing/inverse geometry, validation and independent evaluation. Never claim packages were installed or substitute unrelated training. Preserve defaults until explicit handoff.
