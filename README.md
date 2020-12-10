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

`cd pyCARPool`
`python setup.py install`

## Usage

First, it is advised to import the module as below for simplicity:

```
from pyCARPool import CARPool, confidenceInt, CARPoolSamples, CARPoolEstimator
```

### Description

Put some equations here to explain the method

### Examples

Put examples of functions and class usage here


## Important Remarks
