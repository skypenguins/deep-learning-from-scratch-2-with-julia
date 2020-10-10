# レイヤ定義ファイルの読み込み
include("../common/layers.jl")

using Random

# TwoLayerNetの定義
mutable struct TwoLayerNet
    layers
    loss_layer
    params
    grads
    
    function TwoLayerNet(;input_size, hidden_size, output_size, weight_init=0.01) # Keyword Argumentのみの場合はセミコロンが必要 cf. https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments
        self = new()
        I_s, H_s, O_s = input_size, hidden_size, output_size
        # 重みとバイアスの初期化
        W_1 = weight_init .* Random.randn(I_s, H_s)
        b_1 = zeros(H_s)' # JuliaのブロードキャストはNumPyと挙動が異なり，次元を追加しないため，予め転置して次元数を揃える
        W_2 = weight_init .* Random.randn(H_s, O_s)
        b_2 = zeros(O_s)'
        
        # レイヤの生成
        self.layers = [
            Affine(W_1, b_1),
            Sigmoid(),
            Affine(W_2, b_2)
        ]

        # すべての重みと勾配をまとめる
        self.params, self.grads = [], []
        for layer in self.layers
            push!(self.params, layer.params)
            push!(self.grads, layer.grads)
        end
        self.loss_layer = SoftmaxWithLoss()
        return self
    end
end

# Python版のTwoLayerNetのpredict()
function predict!(model::TwoLayerNet, x)
    for layer in model.layers
        x = forward!(layer, x) # Todo:同名モジュールをusingした場合でもモジュール名を省略したい
    end
    return x
end

# Python版のTwoLayersNetのforward()
function forward!(model::TwoLayerNet, x, t)
    score = predict!(model, x)
    loss = forward!(model.loss_layer, score, t) # SoftmaxWithLoss()のforward()
    return loss
end

# Python版のTwoLayersNetのbackward()
function backward!(model::TwoLayerNet; dout=1)
    dout = backward!(model.loss_layer, dout=dout) # SoftmaxWithLoss()のbackward()
    for layer in reverse(model.layers)
        dout = backward!(layer, dout=dout)
    end
    return dout
end