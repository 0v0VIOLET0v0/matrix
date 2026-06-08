# Matrix

A small C++ matrix multiplication demo.

## Overview

This project implements a simple tiled GEMM-style matrix multiplication for
single-precision floating-point matrices. The right-hand matrix is transposed
before multiplication so the inner loop can read both input rows contiguously.
The current sample in `main()` tests the multiplication with matrix sizes
`247 x 256` and `256 x 231`.

## Build

```powershell
g++ matrix.cpp -O0 -g -std=c++17 -o matrix.exe
```

## Run

```powershell
.\matrix.exe
```

Expected output starts with the correctness result and then prints benchmark
parameters:

```text
Correct
m = ...
FLOPs = ...
avg GFLOP/s = ...
best GFLOP/s = ...
```
