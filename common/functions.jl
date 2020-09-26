module Functions
export sigmoid, relu, softmax, cross_entropy_error

abstract type Func end

function sigmoid(x) <: Func
    return 1 ./ (1 .+ exp.(-x))
end

function relu(x) <: Func
    return max(0, x) # NOT maximum()
end

# ifで分岐せず，多重ディスパッチを利用
function softmax(x::Array{Any,2}) <: Func
    x = x .- maximum(x, dims=2) # Juliaではインデックスは1始まり
    return exp.(x) ./ sum(x, dims=2)
end

function softmax(x::Array{Any,1}) <: Func
    x = x .- maximum(x)
    return exp.(x) ./ sum(x) 
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