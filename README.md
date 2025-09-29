# Fine-Tuning Comparison Across Accelerator

This is a simple test to compare the fine-tuning time when working with two different hardware acelerators, Trainium and Nvidia GPU.

## Neuron

On Neuron, tests have been done on a Trainium [trn1.32xlarge](https://aws.amazon.com/ec2/instance-types/trn1/) instance, that has 16 chips with a total of 512 GB of accelerator memory. On this setup, the script used is a copy of a [Optimum Neuron example](https://github.com/huggingface/optimum-neuron/blob/v0.3.0-release/examples/training/qwen3/finetune_qwen3.py).

When running with the time parameter after HF cache has been deleted, results are these:

```
{'train_runtime': 5216.4665, 'train_samples_per_second': 0.456, 'train_steps_per_second': 0.057, 'train_loss': 1.4759539682857115, 'epoch': 2.97}
100%|███████████████████████████████████████████████████████████████████████████████████████████| 297/297 [1:26:57<00:00, 17.57s/it]
[2025-09-24 10:38:23.874: I neuronx_distributed/trainer/checkpoint.py:240] async saving of checkpoint adapter_shards completed
[2025-09-24 10:38:23.878: I neuronx_distributed/trainer/checkpoint.py:256] no checkpoints to remove.

real    91m41.436s
user    543m1.755s
sys     78m18.630s
```

## CUDA

For GPU, the configuration chosen to run it is a [g5.13xlarge](https://aws.amazon.com/ec2/instance-types/g5/) from AWS, that has 4X Nvidia A10G/P8 chips with 24GB of GPU RAM each.

On this setup, the example coming from Neuron has been adapted. Accelerate was used to launch the training and deepspeed's ZeRO configuration used to reduce memory usage.

When running with the time parameter after HF cache has been deleted, results are these:

```
{'train_runtime': 6135.7261, 'train_samples_per_second': 0.391, 'train_steps_per_second': 0.012, 'train_loss': 1.5560000006357828, 'entropy': 1.3681640625, 'num_tokens': 9603960.0, 'mean_token_accuracy': 0.6519232243299484, 'epoch': 3.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 75/75 [1:42:15<00:00, 81.81s/it]

real    103m31.346s
user    411m44.019s
sys     2m41.581s
```
