# Mamba-Diffusion Model with Learnable Wavelet for Controllable Symbolic Music Generation


For demonstrations, please visit our [project website](https://proffusion.github.io/).



## Training

```shell
python proffusion/main.py --model sdf_chd8bar --output_dir result/sdf_chd8bar
```

## Inference

```shell
python proffusion/inference_sdf.py --chkpt_path=/path/to/checkpoint --uncond_scale=5.
```

## Trained Checkpoints



We have released our pretrained [checkpoint](https://). Please download the checkpoint and put it under `/result/` for inference.