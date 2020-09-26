module Two_layer_net
export TwoLayerNet, predict, forward, backward
# レイヤ定義ファイルの読み込み
include("../common/layers.jl")

using .Layers
using Random

# TwoLayerNetの定義
mutable struct TwoLayerNet
    I_s # input_size (Iは単位行列の変数名と衝突するので変更)
    H_s # hidden_size
    O_s # output_size
    W_1
    b_1
    W_2
    b_2
    layers
    loss_layer
    params
    grads
    
    function TwoLayerNet(;input_size, hidden_size, output_size) # Keyword Argumentsのみの場合はセミコロンが必要 cf. https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments
        I_s, H_s, O_s = input_size, hidden_size, output_size
        # 重みとバイアスの初期化
        W_1 = 0.01 .* Random.randn(I_s, H_s) 
        b_1 = zeros(H_s)
        W_2 = 0.01 .* Random.randn(H_s, O_s)
        b_2 = zeros(O_s)
        
        # レイヤの生成
        layers = [
            Affine(W_1, b_1),
            Sigmoid(),
            Affine(W_2, b_2)
        ]

        # すべての重みと勾配をまとめる
        params, grads = [], []
        for layer in layers
            params += layer.params
            grads += layer.grads
        end
        loss_layer = SoftmaxWithLoss()
        new(I_s, H_s, O_s, W_1, b_1, W_2, b_2, layers, loss_layer, params, grads)
    end
end

# Python版のTwoLayerNetのpredict()
function predict(self::TwoLayerNet, x)
    for layer in self.layers
        x = forward(layer, x)
    end
    return x
end

# Python版のTwoLayersNetのforward()
function forward(self::TwoLayerNet, x, t)
    score = predict(self, x)
    loss = forward(self.loss_layer, score, t) # SoftmaxWithLoss()のforward()
    return loss
end

# Python版のTwoLayersNetのbackward()
function backward(self::TwoLayerNet, dout=1)
    dout = backward(self.loss_layer, dout) # SoftmaxWithLoss()のbackward()
    for layer in reverse(self.layers)
        dout = backward(layer, dout)
    end
    return dout
end
end