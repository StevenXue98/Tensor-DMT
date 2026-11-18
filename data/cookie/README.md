# Cookie problem data

This directory contains unmodified input data for the parameter-dependent
elliptic PDE commonly called the Cookie problem. The Python Tucker-GMRES
experiment uses `A0.txt` through `A3.txt` and `b.txt`. The files store sparse
matrix coordinates and the spatial right-hand-side factor, respectively.

## Source

The files are byte-for-byte copies of the corresponding files in
[`SAMSI-RandTensors/randomizedTT`](https://github.com/SAMSI-RandTensors/randomizedTT/tree/main/tests/data),
which distributes them under the MIT License reproduced in [LICENSE](LICENSE).

They are also distributed with the numerical experiment archive for:

R. Smith, M. Pasha, A. Galindo-Olarte, G. Ballard, and H. Al Daas,
*Code, Data and Results for Numerical Experiments in "Efficient Sketching
Based Summation of Tucker Tensors"*, version 1.1, August 2026.
[doi:10.5281/zenodo.21753807](https://doi.org/10.5281/zenodo.21753807)

## Original paper

H. Al Daas, G. Ballard, P. Cazeaux, E. Hallman, A. Miedlar, M. Pasha,
T. W. Reid, and A. K. Saibaba, *Randomized Algorithms for Rounding in the
Tensor-Train Format*, SIAM Journal on Scientific Computing, 45(1), 2023,
pp. A74-A95. [doi:10.1137/21M1451191](https://doi.org/10.1137/21M1451191)
