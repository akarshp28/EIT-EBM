# EIT-EBM

## Data creation:
1. PrepareData_single.m operates in single mode and creates only 1 body with user-selected anomaly conductvity and location.
2. PrepareData_multi.m operates in multi mode and creates multiple bodies with randomly-selected anomaly conductvities and locations. This is useful for ML based training regimes.

## EBM prior training
ebm_score_matching.py trains the EBM using many phantoms created with PrepareData_multi.m.

## Forward problem solving
unet.py trains the forward model UNET using a single phantom created with PrepareData_single.m.

## Inverse problem solving
snet.py trains the inverse model SNET using a pre-trained UNET and corresponding sigma data.
