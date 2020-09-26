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

function update(self::SGD, params, grads)
    for i in 1:1:length(params)
        params[i] -= self.lr .* grads[i] # .-=は破壊的変更
    end
end
end