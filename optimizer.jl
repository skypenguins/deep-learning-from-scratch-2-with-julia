module Optimizer

abstract type Optim end

mutable struct SGD <: Optim
    #= 
    確率的勾配降下法
    Stocastic Gradient Descent =#
    lr
    function SGD(lr=0.01)
        new(lr)
    end
end

function update(params, grads, self::SGD)
    for i in 1:1:length(params)
        params[i] .-= self.lr .* grads[i]
    end
end
end