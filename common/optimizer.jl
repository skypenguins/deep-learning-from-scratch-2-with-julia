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

mutable struct Adam <: Optim
    #= Adam
    (http://arxiv.org/abs/1412.6980v8) =#
    lr
    β_1
    β_2
    iter
    m
    v
    function Adam(;lr=0.001, β_1=0.9, β_2=0.999)
        self = new()
        self.lr = lr
        self.β_1 = β_1
        self.β_2 = β_2
        self.iter = 0
        self.m = nothing
        self.v = nothing
        return self
    end
end

function update!(self::Adam, params, grads)
    if self.m === nothing
        self.m, self.v = [], []
        for param = params
            push!(m, zeros(size(param)...))
            push!(v, zeros(size(param)...))
        end
    end
    self.iter += 1
    lr_t = self.lr .* sqrt(1.0 - self.β_2^self.iter) / (1.0 - self.β_1^self.iter)

    for i = 1:length(params)
        self.m[i] .+= (1 - self.β_1) * (grads[i] - self.m[i])
        self.v[i] .+= (1 - self.β_2) * (grads[i]^2 - self.v[i])

        params[i] .-= lr_t .* self.m[i] / (sqrt(self.v[i]) .+ 1e-7)
    end
end