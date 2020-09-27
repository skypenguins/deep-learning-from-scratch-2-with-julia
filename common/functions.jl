module Functions
export sigmoid, relu, softmax, cross_entropy_error

function sigmoid(x)
    return 1 ./ (1 .+ exp.(-x))
end

function relu(x)
    return max(0, x) # NOT maximum()
end

#= バッチ対応版ソフトマックス関数
softmax(a::Array{Float64,2})::Array{Float64,2} = begin
    @show size(a)
    y = []
    for row in 1:size(a, 1)
        c = maximum(a[row, :])
        exp_a = exp.(a[row, :] .- c) # オーバフロー対策
        push!(y, exp_a ./ sum(exp_a))
    end
    hcat(y...)'
end

cross_entropy_error(y::Array{Float64,2}, t::Array{Float64,2})::Float64 = -sum(t .* log.(y .+ 1e-7)) =#

softmax(x) = x
function softmax(x::Vector{<:Real})
    x = x .- maximum(x, dims=2) # Juliaではインデックスは1始まり
    x .= exp.(x) / sum(exp.(x), dims=2) # .= は高速化
end

function softmax(x::Matrix{<:Real})
    x = x .- maximum(x)
    x .= exp.(x) ./ sum(exp.(x)) 
end

function cross_entropy_error(y, t)
    # TODO:普通に数式の通りに計算するバージョンを検討
    # ミニバッチ学習の処理と共通化させるため，Vectorの場合は1xN Arrayにreshapeする
    if ndims(y) == 1
        t = reshape(t, 1, :)
        y = reshape(y, 1, :)
    end

    # 教師データがont-hotベクトルの場合，yのt列目を取得をするため正解ラベルのインデックスに変換
    if length(t) == length(y) # 要素数
        t = argmax(t, dims=2)[1][1]
    end

    batch_size = size(y, 1)

    # collect()はインデックス番号用，yのt列目を取得
    return .-(sum(log.(y[collect(1:batch_size), t] .+ 1e-7)) ./ batch_size)
end
end