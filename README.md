# Deep Learning from Scratch 2 with Julia
Porting of "Deep Learning from Scratch 2" in Julia

# Diffrences from the Python version
## No classes in Julia
Python版ではニューラルネットワークのレイヤーをクラスで定義していたが、Juliaではそれを構造体（struct）の型（type）で定義。そのため、レイヤーに持たせていたメソッドforward()とbackward()を独立させ、インスタンス変数paramsとgradsを構造体のインスタンス変数として持たせる。

## Julia arrays are column major
Python（NumPy）の配列は行指向（C由来）だが、Juliaの配列はデフォルトでは列指向（Fortran由来）となっている。  
> Julia arrays are column major (Fortran ordered) whereas NumPy arrays are row major (C-ordered) by default.

Source: https://docs.julialang.org/en/v1/manual/noteworthy-differences/#Noteworthy-differences-from-Python

# Prerequisites
- Plots
- StatsBase
- PyCall
  - for `pickle` module
- HTTP
- NPZ

# License
[MIT License](./LICENSE)
