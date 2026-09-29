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

Connect a deployed inference endpoint through the model form using a supported adapter. Credentials use protected forms/environment. An inference endpoint alone does not provide training.

Creating a model from an available training recipe uses monailabel-model-training. Load that skill and create the learner; do not ask for an endpoint or credentials. The user can create a training setup before starting a run.

New architectures, transforms and arbitrary repository code require integration work, not coordinator execution. Explain the needed framework/task/runtime, versioned weights and input/output contracts, preprocessing/inverse geometry, validation and independent evaluation. Never claim packages were installed or substitute unrelated training. Preserve defaults until explicit handoff.
