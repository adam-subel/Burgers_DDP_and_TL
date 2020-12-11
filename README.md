# Burgers_DDP_and_TL
This repository includes the codes to produce datasets and implement the DDP, DSMAG, and TL referenced in the 
accompanying paper. The following links to a dataset that can be used with the given DSMAG and DDP codes, https://zenodo.org/record/4316338.
 
## Stochastic_Burgers_DNS.m
This code creates a DNS dataset. Parameters like the Reynolds number and resolution can be altered easily to create datasets to experiment with transfer learning.

## make_training_sets.m and make_forcing.m 
This codes generate filtered and coarse grained variables for the training and a posteriori testing of DDP.

### calc_bar.m
This code contains a function to take in the  DNS dataset and then calculate the filtered variables and subgrid Pi terms.

#### filter_bar.m
This code contains a function to apply the box filter.

## DSMAG.py
This code is an implementation of the Dynamic Smagorinsky LES.

## ddp_train_and_test.py
This code trains and runs a posteriori prediction for DDP.

### Transfer_Learning.py
This code takes in a set of weights for the ANN used in DDP and retrains it for a different training regime.
