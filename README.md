## Prerequisites

- Linux

- Python (2.7 or later) Recommended: python 3.7

- numpy

- scipy

- NVIDIA GPU + CUDA 8.0 + CuDNN v5.1

- TensorFlow 1.0 or later Recommended: tensorflow 1.15


## steps

- clone this repo:

```
git clone https://github.com/duxingren14/DualGAN.git

cd DualGAN
```

- train the model:

```
python main.py --phase train --dataset_name sketch-photo --image_size 256 --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100
```

- test the model:

```
python main.py --phase test --dataset_name sketch-photo --image_size 256 --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100
```

## optional

Similarly, run experiments on facades dataset with the following commands:

```
python main.py --phase train --dataset_name facades --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100

python main.py --phase test --dataset_name facades --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100
```

# Acknowledgments

Codes are built on the top of pix2pix-tensorflow and DCGAN-tensorflow. Thanks for their precedent contributions!
"# DUAL_GAN" 

python -m pytorch_fid ./input/A ./input/B

FID Calculation