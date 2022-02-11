# Burgers_DDP_and_TL
This repository includes the codes to produce datasets and implement the DDP, DSMAG, and TL referenced in the 

accompanying paper *Data-driven subgrid-scale modeling of forced Burgers turbulence using deep learning with generalization to higher Reynolds numbers via transfer learning* (https://aip.scitation.org/doi/full/10.1063/5.0040286). The following links to a dataset that can be used with the given DSMAG and DDP codes, https://zenodo.org/record/4316338.
 
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

## Citation
<!-- Links to published/arxiv work -->
Read more on [[arXiv]](https://arxiv.org/pdf/2012.06664.pdf)
<!-- Use DOI links when available -->
Read more on [[PoF]](https://doi.org/10.1063/5.0040286)
```
@article{subel2020data,
  title={Data-driven subgrid-scale modeling of forced Burgers turbulence using deep learning with generalization to higher Reynolds numbers via transfer learning},
  author={Subel, Adam and Chattopadhyay, Ashesh and Guan, Yifei and Hassanzadeh, Pedram},
  journal={arXiv e-prints},
  pages={arXiv--2012},
  year={2020}
}

@article{subel2021data,
  title={Data-driven subgrid-scale modeling of forced Burgers turbulence using deep learning with generalization to higher Reynolds numbers via transfer learning},
  author={Subel, Adam and Chattopadhyay, Ashesh and Guan, Yifei and Hassanzadeh, Pedram},
  journal={Physics of Fluids},
  volume={33},
  number={3},
  pages={031702},
  year={2021},
  publisher={AIP Publishing LLC}
}
