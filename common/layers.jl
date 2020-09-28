module AbstractLayers
export MatMul, Affine, Softmax, SoftmaxWithLoss, Sigmoid, forward!, backward!

include("./functions.jl")

using .Functions: softmax, cross_entropy_error

abstract type AbstractLayer end

mutable struct MatMul <: AbstractLayer
    params::AbstractArray{Float64}
    grads::AbstractArray{Float64}
    W::Float64
    x::Float64
    function MatMul(W)
        self = new()
        self.params = [W]
        self.grads = [zero(W)]
        return self
    end
end

function forward!(self::MatMul, x)
    W, = self.params
    self.x = x
    return out = x * W
end

function backward!(self::MatMul, dout)
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
        self = new()
        self.params = [W, b]
        self.grads = [zero(W), zero(b)]
        return self
    end
end

function forward!(self::Affine, x)
    W, b = self.params
    self.x = x
    return out = x * W .+ b
end

function backward!(self::Affine, dout)
    W, b = self.params
    dx = dout * W'
    dW = self.x' * dout
    db = sum(dout, dims=1)

    self.grads[1] = dW
    self.grads[2] = db
    return dx
end

mutable struct Softmax <: AbstractLayer
    params
    grads
    out
    function Softmax()
        self = new()
        self.params = []
        self.grads = []
        return self
    end
end

function forward!(self::Softmax, x)
    return self.out = softmax(x)
end

function backward!(self::Softmax, dout)
    dx = self.out .* dout
    sumdx = sum(dx, dims=2)
    return dx -= self.out .* sumdx
end

mutable struct SoftmaxWithLoss <: AbstractLayer
    params
    grads
    y # SoftmaxレイヤからCrossEntropyErrorレイヤへのデータ
    t
    function SoftmaxWithLoss()
        self = new()
        self.params = []
        self.grads = []
        return self
    end
end

function forward!(self::SoftmaxWithLoss, x, t)
    self.t = t
    self.y = softmax(x)
    return loss = cross_entropy_error(self.y, self.t)
end

function backward!(self::SoftmaxWithLoss, dout=1.0)
    batch_size = size(self.t, 1)
    return dx = (self.y .- self.t) .* dout ./ batch_size
end

mutable struct Sigmoid <: AbstractLayer
    params
    grads
    out
    function Sigmoid()
        self = new()
        self.params = []
        self.grads = []
        return self
    end
end

function forward!(self::Sigmoid, x)
    return self.out = 1 ./ (1 .+ exp.(-x))
end

function backward!(self::Sigmoid, dout)
    return dx = dout .* (1.0 .- self.out) .* self.out
end
end