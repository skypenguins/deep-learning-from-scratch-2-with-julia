module AbstractLayers
export MatMul, Affine, Softmax, SoftmaxWithLoss, Sigmoid, forward, backward

include("./functions.jl")

using .Functions: softmax, cross_entropy_error

abstract type AbstractLayer end

mutable struct MatMul <: AbstractLayer
    params::AbstractArray{Float64}
    grads::AbstractArray{Float64}
    W::Float64
    x::Float64
    function MatMul(W)
        params = (W,)
        grads = (zero(W),)
        new(params, grads, W)
    end
end

function forward(self::MatMul, x)
    W, = self.params
    out = x * W
    self.x = x
    return out
end

function backward(self::MatMul, dout)
    W, = self.params
    dx = dout * W'
    dW = self.x' * dout
    self.grads[1] = deepcopy(dW)
    return dx
end

mutable struct Affine <: AbstractLayer
    params
    grads
    W
    b
    x
    function Affine(W, b)
        params = [W, b]
        grads = [zero(W), zero(b)]
        new(params, grads)
    end
end

function forward(self::Affine, x)
    W, b = self.params
    out = x * W .+ b
    self.x = x
    return out
end

function backward(self::Affine, dout)
    W, b = self.params
    dx = dout * W'
    dW = self.x' * dout
    db = sum(dout, dims=1)

    self.grads[1] = deepcopy(dW)
    self.grads[2] = deepcopy(db)
    return dx
end

mutable struct Softmax <: AbstractLayer
    params
    grads
    out
    function Softmax()
        params = []
        grads = []
        new(params, grads)
    end
end

function forward(self::Softmax, x)
    self.out = softmax(x)
    return self.out
end

function backward(self::Softmax, dout)
    dx = self.out .* dout
    sumdx = sum(dx, dims=2)
    dx -= self.out .* sumdx
    return dx
end

mutable struct SoftmaxWithLoss <: AbstractLayer
    params
    grads
    y # SoftmaxレイヤからCrossEntropyErrorレイヤへのデータ
    t
    function SoftmaxWithLoss()
        params = []
        grads = []
        new(params, grads)
    end
end

function forward(self::SoftmaxWithLoss, x, t)
    self.t = t
    self.y = softmax(x)

    # 教師ラベルがone-hotベクトルの場合，正解ラベルのインデックスに変換
    if length(self.t) == length(self.y)
        self.t = argmax(self.t, dims=2)[1][1] # Array{CartesianIndex{Int64},2}
    end

    loss = cross_entropy_error(self.y, self.t)
    return loss
end

function backward(self::SoftmaxWithLoss, dout=1)
    batch_size = size(self.t, 1)
    
    dx = deepcopy(self.y)
    dx[collect(1:batch_size), self.t] .-= 1 # 配列のスライスはコンマ必要
    dx = dx .* dout
    dx = dx ./ batch_size
    return dx
end

mutable struct Sigmoid <: AbstractLayer
    params
    grads
    out
    function Sigmoid()
        params = []
        grads = []
        new(params, grads)
    end
end

function forward(self::Sigmoid, x)
    out = 1 ./ (1 .+ exp.(-x))
    self.out = out
    return out
end

function backward(self::Sigmoid, dout)
    dx = dout .* (1.0 .- self.out) .* self.out
    return dx
end
end