# Improved Training of Physics-Informed Neural Networks Using Energy-Based Priors: a Study on Electrical Impedance Tomography

## Paper: https://openreview.net/pdf?id=zqkfJA6R1-r

## Abstract:
Physics-informed neural networks (PINNs) are attracting significant attention for solving partial differential equation (PDE) based inverse problems, including electrical impedance tomography (EIT). EIT is non-linear and especially its inverse problem is highly ill-posed. Therefore, successful training of PINNs is extremely sensitive to the interplay between different loss terms and hyper-parameters, including the learning rate. In this work, we propose a Bayesian approach through a data-driven energy-based model (EBM) as a prior, to improve the overall accuracy and quality of tomographic reconstruction. In particular, the EBM is trained over the possible solutions of the PDEs with different boundary conditions. By imparting such prior onto physics-based training, PINN convergence is expedited more than ten times faster than the PDE’s solution. The evaluation outcome shows that our proposed method is more robust for solving the EIT problem.

## EIT-EBM code repository layout

## Data creation:
1. PrepareData_single.m operates in single mode and creates only 1 body with user-selected anomaly conductvity and location.
2. PrepareData_multi.m operates in multi mode and creates multiple bodies with randomly-selected anomaly conductvities and locations. This is useful for ML based training regimes.

## EBM prior training
ebm_score_matching.py trains the EBM using many phantoms created with PrepareData_multi.m.
eit_classifier.py trains a EIT classification model to determine the number of anomalies in a given phantom. This is useful for FID score computation.

## Forward problem solving
unet.py trains the forward model UNET using a single phantom created with PrepareData_single.m.

## Inverse problem solving
snet.py trains the inverse model SNET using a pre-trained UNET and corresponding sigma data.

## If our code was used in your projects, please credit our paper using below bibtext:

```
@inproceedings{
pokkunuru2023improved,
title={Improved Training of Physics-Informed Neural Networks Using Energy-Based Priors: a Study on Electrical Impedance Tomography},
author={Akarsh Pokkunuru and Pedram Rooshenas and Thilo Strauss and Anuj Abhishek and Taufiquar Khan},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=zqkfJA6R1-r}
}
```
