# Simple checkpoint and lora Replicate model pusher

You can use the `train` feature of this model to push new Replicate models:

Visit the model then follow the instructions below:

https://replicate.com/fofr/push-checkpoint-or-lora/train

## Weights

When pushing a model with your choice of checkpoints and/or loras, you have two choices:

1. Pick from already available weights by specifying a filename. The filename must be one that’s listed in https://github.com/fofr/cog-comfyui/blob/main/weights.json in either CHECKPOINTS or LORAS.
2. Use your own checkpoint by giving a HuggingFace or CivitAI download link.

## Defaults

When ‘training’, you should also pick the best settings for the model:

- sampler
- scheduler
- cfg
- steps

For example, for a lightning model you might need to specify the sampler it was trained with, a low CFG and 4 steps.
