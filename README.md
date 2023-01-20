# Measuring the Transferability of Pre-trained DNNs

This is the repository of paper "Rethinking Two Consensuses of Transferability of Pre-trained Deep Neural Networks", which is under review in ICML 2023.
In this repo, we implement the PyTorch codes and examples for measuring pre-trained DNNs' transferability on downstream tasks.

### 1. Brief Introduction to this Method

As learned knowledge, the pre-trained parameters of DNNs act as a closer initialization to the optimal point for the downstream tasks than random initialization. Based on this point of view, we quantify transferability as the extent to which pre-training helps to push the parameters closer to the optimal point for the downstream task. More transferable parameters should be closer to the optimal point of the downstream task, making the adaptation to the target domain easier. This method allows us to compare the transferabilities between different downstream tasks under the same standard, and to derive the transferabilities of different layers with precision.

Specifically, we first calculate the parameter distance $D(\theta_r, \theta_B)$ and $D(\theta_A, \theta_B)$, where $\theta_r$ is the random initialization, $\theta_A$ and $\theta_B$ are the convergence points on pre-training task $A$ and downstream task $B$. The transferability of pre-trained $\theta_A$ on task $B$ is $T_B(\theta_A) = D(\theta_r, \theta_B)/D(\theta_A, \theta_B)$ (left in the following figure). To avoid the parameter scale problem in different layers, we calculate the transferability for all layers and regard the mean value as the transferability of the whole DNN (right in the following figure).

<img src="https://github.com/Schuture/Transferability-of-Pre-trained-DNNs/blob/main/Figs/method.png" width = "800" height = "250" alt="Method" align=center />

### 2. Step by Step Implementation

(1) Pre-train a DNN (*e.g.*, resnet) on a large dataset (*e.g.*, ImageNet), and save the initialization parameters and converged parameters.

```
python pretrain_on_ImageNet.py
```

(2) Fine-tune the pre-trained DNN on a downstream task (*e.g.*, CIFAR-10), and save the converged parameters.

```
python finetune.py
```

(3) Calculate the layer-wise and overall transferability of the DNN.

```
python cal_transferability.py -r random_init_model.pth -a ImageNet_model.pth -b cifar10_model_lr001.pth
```


### 5. Environment
The code is developed with an Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz and a single Nvidia Ampere A100 GPU.

The install script *requirements.txt* has been tested on an Ubuntu 18.04 system.

:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:

### 6. License

Licensed under an MIT license.





