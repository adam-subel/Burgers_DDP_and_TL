# Burgers_DDP_and_TL
This repository includes the codes to produce datasets and implement the DDP, DSMAG, and TL referenced in the 
accompanying paper.
 
## Stochastic_Burgers_DNS.m
This code creates a DNS dataset. Parameters like the Reynolds number and resolution can be altered easily to create datasets to experiment with transfer learning.

## make_training_sets.m and make_forcing.m 
This codes generate filtered and coarse grained variables for the training and $\it{a posteriori}$ testing of DDP.

### calc_bar.m
This code contains a function to take in the  DNS dataset and then calculate the filtered variables and subgrid terms $\Pi$ terms.

#### filter_bar.m

## DSMAG.py

## ddp_train_and_test.py

### Transfer_Learning.py
