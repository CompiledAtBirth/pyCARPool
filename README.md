# FMM_3D_pybind11

A repository providing an example of pybind11 usage with the 3D Black-box Fast Multipole Method (https://github.com/ruoxi-wang/PBBFMM3D).
The black-box FMM originates from [W.Fong & E.Darve (2009)](https://mc.stanford.edu/cgi-bin/images/f/fa/Darve_bbfmm_2009.pdf).

## Requirements

- [pybind11](https://github.com/pybind/pybind11) : You either need to have the path/to/pybind11/include in your CPATH environment variable, or to include it manually in the code.
- [cppimport](https://github.com/tbenthompson/cppimport) : I use it, but it is not the only solution to compile the wrappers into standard (C)Python extension modules. See pybind11 documentation.
- Intel Math Kernel Library (MKL)

## Usage

The consistency of the binding is evaluated by comparing the output of the BBFMM calculated "directly" in C++ on the one hand, and in Python with pybind11 on the other hand. The former is calculated with :

```
make test_PBBFMM
cd exec
./test_PBBFMM
```
and the latter with `python3 test_PBBFMM.py` in the tests_PBBFMM repository.

The output is referred as the QH product, with Q the kernel matrix and H the set(s) of charges (single set ofunitary charges here for testing purposes).

## Files description

### data
- distribution_radial3D.txt is a presaved distributions of particles.
- The "casted source" file is just the distribution read in C++ and immediately written again as an output to inspect float/double consistency in the wrapper.
- The QH files are the outputs of the FMM (in C++ and in Python with pybind11)

### test_PBBFMM3D

  - *PBBFMM_shot_test.cpp* : source script to calculate the QH product directly with the PBBFMM3D submodule and a given distribution of particles
  - *PBBFMM_binding_test.cpp* and *test_PBBFMM.py* : pybind11 wrapper and python tests script, the distribution of particles being a (N,3) numpy array from an other Python application
  
### Results

Some plots are saved here (comparison between the outputs).

## Important Remarks

-  The *output* folder needs to be exactly is this place with this name ; if not, a segmentation fault will occur (hard coded path in PBBFMM3D module). It contains the precomputed files for a given kernel and interpolation order.

- I did not manage to get the $(MKLROOT) working with cppimport, so I wrote the path manually in the compiler and linker arguments (PBBFMM_binding_test.cpp).
