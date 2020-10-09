# Deep Learning from Scratch 2 with Julia
Porting of "Deep Learning from Scratch 2" in Julia

# Diffrences from the Python version
Python（NumPy）では1次元配列と<img src="https://render.githubusercontent.com/render/math?math=1\times x">行列は同じ扱いだが，Juliaでは扱いが異なる。
これは配列の扱いがJuliaの列指向とPythonの行指向で異なるためである。
そのため，行指向の設計思想を基に実装されたアルゴリズムを列指向の設計思想で再実装することが理想ではあるが，Python版の行列やテンソルの形状と一致させるため，（あと初心者なので…）無理矢理，行指向の設計思想でコードを実装している。

> Julia arrays are column major (Fortran ordered) whereas NumPy arrays are row major (C-ordered) by default.

Source: https://docs.julialang.org/en/v1/manual/noteworthy-differences/#Noteworthy-differences-from-Python

# Prerequisites
- Plots
- PyCall
  - for `pickle` module
- HTTP
- NPZ
