# Matrix

A small C++ matrix multiplication demo.

## Overview

This project implements a simple tiled GEMM-style matrix multiplication for
single-precision floating-point matrices. The current sample in `main()` tests
the multiplication with matrix sizes `247 x 256` and `256 x 231`.

## Build

```powershell
g++ matrix.cpp -O2 -std=c++17 -o matrix.exe
```

## Run

```powershell
.\matrix.exe
```

Expected output includes the first and last result values:

```text
C[0][0] = 256
C[last] = 256
```
