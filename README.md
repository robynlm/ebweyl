The ebweyl module provides :

* A FiniteDifferencing class with 4th (default) and 6th order backward, centered, and forward schemes. Periodic boundary conditions are used by default, otherwise a combination of the 3 schemes is available.

* A Weyl class that computes for a given metric the variables of the 3+1 formalism, the spatial Christoffel symbols, spatial Ricci tensor, electric and magnetic parts of the Weyl tensor projected along the normal to the hypersurface and fluid flow, the Weyl scalars and invariant scalars.

* Functions compute the determinant and inverse of a 3x3 or 4x4 matrice in every position of a data box.

Should you use this code, please reference:
@article{R.L.Munoz_M.Bruni_2023a,
    title     = {EBWeyl: a Code to Invariantly Characterize Numerical Spacetimes},
    author    = {Munoz, Robyn L. and Bruni, Marco},
    journal   = {Classical and Quantum Gravity},
    volume    = {40},
    number    = {13},
    pages     = {135010},
    year      = {2023},
    month     = {6},
    doi       = {10.1088/1361-6382/acd6cf},
    archivePrefix = {arXiv},
    eprint    = {gr-qc/2211.08133}}
