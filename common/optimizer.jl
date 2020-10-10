abstract type Optim end

mutable struct SGD <: Optim
    #= 
    確率的勾配降下法
    Stocastic Gradient Descent =#
    lr
    function SGD(;lr=0.01)
        new(lr)
    end
end

function update!(self::SGD, params, grads)
    for i = 1:lastindex(params)
        for j = 1:lastindex(params[i])
            params[i][j] = params[i][j] .- self.lr .* grads[i][j] # .-=は破壊的変更
        end
    end
end
