# Deep Learning from Scratch 2 with Julia
Porting of "Deep Learning from Scratch 2" in Julia

# Diffrences from the Python version
## No classes in Julia
Python版ではレイヤをclassで定義していたが，Juliaではclassの概念が存在しない。  
モデルを構造体で定義し，それにメソッドを作用させていくように考える。そのため，レイヤに持たせていたメソッドforward()とbackward()，インスタンス変数paramsとgradsを独立させる。メソッドはfunctionで定義し，インスタンス変数はstructに持たせる。  
Juliaの多重ディスパッチの機構を利用し，同じ名前のメソッドでも型によって異なる処理を呼び出す。レイヤを型として定義する。

## Julia arrays are column major
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
