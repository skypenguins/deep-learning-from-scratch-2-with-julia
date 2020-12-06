include("./functions.jl")

using .Functions: softmax, cross_entropy_error

abstract type AbstractLayer end

mutable struct MatMul <: AbstractLayer
    params
    grads
    x
    function MatMul(W)
        layer = new()
        layer.params = [W]
        layer.grads = [zero(W)]
        return layer
    end
end

function forward!(layer::MatMul, x)
    W, = layer.params
    layer.x = x
    return out = x * W
end

# Keyword Argのデフォルトが指定されていない場合，必ず呼出側で引数指定する
function backward!(layer::MatMul, dout)
    W, = layer.params
    dx = dout * W'
    dW = layer.x' * dout
    layer.grads[1] = deepcopy(dW)
    return dx
end

mutable struct Affine <: AbstractLayer
    params
    grads
    x
    function Affine(W, b)
        layer = new()
        layer.params = [W, b]
        layer.grads = [zero(W), zero(b)]
        return layer
    end
end

function forward!(layer::Affine, x)
    W, b = layer.params
    layer.x = x
    return out = x * W .+ b
end

function backward!(layer::Affine; dout)
    W, b = layer.params
    dx = dout * W'
    dW = layer.x' * dout
    db = sum(dout, dims=1)

    layer.grads[1] = dW
    layer.grads[2] = db
    return dx
end

mutable struct Softmax <: AbstractLayer
    params
    grads
    out
    function Softmax()
        layer = new()
        layer.params = []
        layer.grads = []
        return layer
    end
end

function forward!(layer::Softmax, x)
    return layer.out = softmax(x)
end

function backward!(layer::Softmax; dout)
    dx = layer.out .* dout
    sumdx = sum(dx, dims=2)
    return dx -= layer.out .* sumdx
end

mutable struct SoftmaxWithLoss <: AbstractLayer
    params
    grads
    y # SoftmaxレイヤからCrossEntropyErrorレイヤへのデータ
    t
    function SoftmaxWithLoss()
        layer = new()
        layer.params = []
        layer.grads = []
        return layer
    end
end

function forward!(layer::SoftmaxWithLoss, x, t)
    layer.t = t
    layer.y = softmax(x)
    return loss = cross_entropy_error(layer.y, layer.t)
end

function backward!(layer::SoftmaxWithLoss; dout=1.0)
    batch_size = size(layer.t, 1)
    return dx = (layer.y .- layer.t) .* dout ./ batch_size
end

mutable struct Sigmoid <: AbstractLayer
    params
    grads
    out
    function Sigmoid()
        layer = new()
        layer.params = []
        layer.grads = []
        return layer
    end
end

function forward!(layer::Sigmoid, x)
    return layer.out = 1 ./ (1 .+ exp.(-x))
end

function backward!(layer::Sigmoid; dout)
    return dx = dout .* (1.0 .- layer.out) .* layer.out
end

mutable struct Embedding <: AbstractLayer
    params
    grads
    idx
    function Embedding(W)
        layer = new()
        layer.params = [W]
        layer.grads = [zero(W)]
        layer.idx = nothing
        return layer
    end
end

function forward!(layer::Embedding, idx)
    W, = layer.params
    layer.idx = idx
    out = selectdim(W, 1, idx)
    return out
end

function backward!(layer::Embedding, dout)
    dW, = self.grads
    dW = deepcopy(zero(dW))
    selectdim(dW, 1, layer.idx) .+= dout
end

mutable struct SigmoidWithLoss <: AbstractLayer
    params
    grads
    loss
    y
    t
    function SigmoidWithLoss()
        self = new()
        self.loss = nothing
        self.y = nothing
        self.t = nothing
        return self
    end
end

function forward!(layer::SigmoidWithLoss, x, t)
    layer.t = t
    layer.y = 1 ./ (1 .+ exp.(-x))

    layer.loss = cross_entropy_error(hcat(1 - layer.y, layer.y), layer.t)
    return layer.loss
end

function backward!(layer::SigmoidWithLoss; dout=1)
    batch_size = size(layer.t, 1)
    dx = (layer.y - layer.t) .* dout ./ batch_size
    return dx
end