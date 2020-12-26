[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3743496.svg)](https://doi.org/10.5281/zenodo.3743496)

![Build Status](https://github.com/BBN-Q/MEsolve.jl/workflows/CI/badge.svg)

MESolve.jl is a Lindbald master equation solver written in Julia. It supports time-independent evolution, as well as time-dependent evolution with a time-dependent Hamiltonian and/or time-dependent Lindblad dissipative rates. The Lindblad equation is assumed to be in diagonal form.

MESolve.jl also provides support for solving covariance matrix differential equations for Gaussian continuous variable systems, as well as helper functions to create common discrete and continuous variable systems.


## Installation
This branch can now be locally installed as a package with the command:
```julia
(v1.0) pkg> add <path to this repo>
```

To run the test code do:
```julia
(v1.0) pkg> test MESolve
```

## Usage
There are 5 main functions to solve master equations:

* me_solve_time_independent() : Time-independent
* me_solve_H_time_dependent() : Time-dependent Hamiltonian
* me_solve_L_time_dependent() : Time-dependent Lindblad rates
* me_solve_full_time_dependent() : Time-dependent Hamiltonian and Lindblad rates
* me_solve_time_independent_vec() : Time-independent with a vectorized derivative operator

These output an array of density matrices (solution of the master equation as a function of time), and an array of the times of solution.

There are 2 functions to solve covaraiance matrix differential equations for Gaussian CV systems:

* CV_solve_time_independent() : Time-independent Hamiltonian, and dissipation.
* CV_solve_time_dependent() : Time-dependent Hamiltonian (both full time-dependent, and time-dependent linear terms only are possible), and time-independent dissipation.

This outputs an array of covariance matrices and an array of average value vectors (solutions as a function of time), and arrays of the times of solution. Both functions have various possible input styles, including describing the Hamiltonian in the position/momentum operators basis or the lowering/raising operator basis.

See source code for more details, including full list of input arguments.

## License
Apache License v2.0

## Contributors
Luke Govia (luke.c.govia@raytheon.com)

## Acknowledgements
This effort was partially funded by ARO under contracts W911NF-19-C-0092 and W911NF-14-C-0048.

## Copyright
Raytheon BBN Technologies
