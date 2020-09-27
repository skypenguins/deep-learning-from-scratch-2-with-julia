module Functions
export sigmoid, relu, softmax, cross_entropy_error

abstract type Func end

function sigmoid(x) <: Func
    return 1 ./ (1 .+ exp.(-x))
end

function relu(x) <: Func
    return max(0, x) # NOT maximum()
end

#= バッチ対応版ソフトマックス関数
softmax(a::Array{Float64,2})::Array{Float64,2} = begin
    y = []
    for row in 1:size(a, 1)
        c = maximum(a[row, :])
        exp_a = exp.(a[row, :] .- c) # オーバフロー対策
        push!(y, exp_a ./ sum(exp_a))
    end
    hcat(y...)'
end

cross_entropy_error(y::Array{Float64,2}, t::Array{Float64,2})::Float64 = -sum(t .* log.(y .+ 1e-7)) =#

function softmax(x) <: Func
    if ndims(x) == 2
        x = x .- maximum(x, dims=2) # Juliaではインデックスは1始まり
        x = exp.(x) ./ sum(x, dims=2)
    elseif ndims(x) == 1
        x = x .- maximum(x)
        x = exp.(x) ./ sum(x) 
    end
    
    return x
end 

function cross_entropy_error(y, t) <: Func
    if dims(y) == 1
        t = reshape(t, 1, :)
        y = reshape(y, 1, :)
    end

    # 教師データがont-hotベクトルの場合，正解ラベルのインデックスに変換
    if length(t) == length(y) # 要素数
        t = argmax(t, dims=2)
    end

    batch_size = size(y, 1)

    return -sum(log(y[collect(1:batch_size), t] + 1e-7)) ./ batch_size
end
end