# Boosting Sharpness-Aware Minimization with Dominant-Eigenvector Gradient Correction (X-SAM)

This repository provides the official implementation of **X-SAM**, proposed in our paper:  
*Boosting Sharpness-Aware Minimization with Dominant-Eigenvector Gradient Correction*

---

## Introduction

**X-SAM** is an improvement over standard **Sharpness-Aware Minimization (SAM)**.  
It periodically estimates the **dominant eigenvector of the Hessian** and corrects the gradient along this direction during updates.  
This approach can:

- Suppress oscillations along sharp directions, improving training stability  
- Retain gradient sensitivity along flatter directions, enhancing generalization  
- Be used seamlessly with **SGD,ADAM**, offering flexible optimization strategies

---

## Features

- **Periodic Hessian dominant eigenvector estimation**  
- **Gradient correction** to reduce the component along the dominant eigenvector  
- Fully configurable training hyperparameters  
- Improves training stability and generalization performance  

---
## Training Example

Use the following command to train a model with **X-SAM**:

```bash
# Specify the GPU device
CUDA_VISIBLE_DEVICES=0 python trainxsam.py \
    --sam x-sam \            # Optimizer: sgd, sam, or x-sam
    --freq 3 \               # Frequency to update dominant eigenvector
    --epochs 200 \           # Total number of training epochs
    --alpha 0.2 \            # Step size for dominant-eigenvector gradient correction
    --batch_size 256 \       # Mini-batch size
    --dataset CIFAR10 \      # Dataset: CIFAR10, CIFAR100, etc.
    --num_workers 4 \        # Number of data loading workers
    --rho 0.05 \             # Perturbation magnitude for SAM/X-SAM
    --weight_decay 5e-5 \    # L2 regularization coefficient
    --seed 42 \              # Random seed for reproducibility
    --model resnet           # Model architecture


