module Spiral
using Random

export load_data

function load_data(seed=1984)
    Random.seed!(seed)  # 関数直後の!は関数が破壊的変更を行うことを示す
    @show seed
    N = 100             # クラスごとのサンプルサイズ
    DIM = 2             # データの要素数
    CLS_DIM = 3         # クラス数

    x::AbstractArray{Float64} = zeros(N * CLS_DIM, DIM)
    t = zeros(Integer, N * CLS_DIM, CLS_DIM)

    for j = 0:CLS_DIM - 1 # Juliaではインデックスが1始まりのため
        for i = 0:N - 1
            rate = i / N
            radius = 1.0 * rate
            θ = j * 4.0 + 4.0 * rate + Random.randn() * 0.2

            ix = N * j + i
            x[ix + 1, :] = [radius * sin(θ) radius * cos(θ)]

            t[ix + 1, j + 1] = 1
        end
    end

    return x, t
end
end