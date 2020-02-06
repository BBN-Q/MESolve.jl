MESolve.jl is a Lindbald master equation solver written in Julia. It supports time-independent evolution, as well as time-dependent evolution with a time-dependent Hamiltonian and/or time-dependent Lindblad dissipative rates. The Lindblad equation is assumed to be in diagonal form.


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
`papa_reconstruction()` is the primary function for PAPA reconstructions.

## License
Apache License v2.0

## Contributors
Luke Govia (luke.c.govia@raytheon.com)

## Acknowledgements
This effort was partially funded by ARO under contract W911NF-14-C-0048.

## Copyright
Raytheon BBN Technologies
