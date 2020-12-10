# pyCARPool

Repository that provides a custom class and functions to use the CARPool method from [Chartier et al. (2020)](https://arxiv.org/abs/2009.08970).
The 'examples/' repository contains scripts allowing to reproduce results from the paper. 

## Requirements

- Python >= 3.3
- numpy
- scipy
- [scikits-bootsrap](https://github.com/cgevans/scikits-bootstrap) : Used to compute Bias-corrected and Accelerated (BCA) bootstrap confidence intervals.
- tqdm

## Installation
In terminal, go to the project folder and run

    cd pyCARPool/
    python setup.py install

## Usage

First, it is advised to import the module as below for simplicity:

```
from pyCARPool import CARPool, confidenceInt, CARPoolSamples, CARPoolEstimator
```
The goal is to compute an unbiased estimate of a parameter vector with much lower variance than the samples mean of costly simulations. For that, we use surrogate samples paired with the simulation ones.
Beforehand, order your data so that you have two numpy arrays of shape (P,N), with P the dimension of the samples (ex: number of Power spectrum bins) and N the number of realizations, ordered in a way that each column of the surrogate and simulation data correspond to the same random seed. 
You also need to provide an estimate of the surrogate mean as an array of shape (P,) from an other set of surrogate ralizations.


### Compute single estimates


### Test mode: investigate the performance of CARPool

Put examples of functions and class usage here

## Important Remarks
