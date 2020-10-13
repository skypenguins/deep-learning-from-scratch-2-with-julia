include("../common/layers.jl")

using Random

mutable struct SimpleCBOW
    in_layer0
    in_layer1
    out_layer
    loss_layer
    params
    grads
    word_vecs

    function SimpleCBOW(vocab_size, hidden_size)
        self = new()
        V, H = vocab_size, hidden_size

        # 重みの初期化
        W_in = 0.01 .* Random.randn(V, H) # Float32の指定は特にしない
        W_out = 0.01 .* Random.randn(H, V) # Float32の指定は特にしない

        # レイヤの生成
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 全ての重みと勾配を配列にまとめる
        layers = [self.in_layer0, self.in_layer1, self.out_layer, self.loss_layer]
        self.params, self.grads = [], []
        for layer in layers
            push!(self.params, layer.params)
            push!(self.grads, layer.grads)
        end

        # 構造体の変数に単語の分散表現を設定
        self.word_vecs = W_in
        return self
    end
end

function forward!(self::SimpleCBOW, contexts, target)
    # contextsの形状は3次元決め打ち
    h0 = forward!(self.in_layer0, contexts[:, 1, :]) # MatMulのforward!
    h1 = forward!(self.in_layer1, contexts[:, 2, :]) # MatMulのforward!
    h = (h0 .+ h1) .* 0.5
    score = forward!(self.out_layer, h) # MatMulのforward!
    loss = forward!(self.loss_layer, score, target) # SoftmaxWithLossのforward!
    return loss
end

function backward!(self::SimpleCBOW; dout=1)
    ds = backward!(self.loss_layer; dout=dout) # SoftmaxWithLossのbackward!
    da = backward!(self.out_layer, ds) # MatMulのbackward!
    da .*= 0.5
    backward!(self.in_layer1, da) # MatMulのbackward!
    backward!(self.in_layer0, da) # MatMulのbackward!
end