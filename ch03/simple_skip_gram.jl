include("../common/layers.jl")

using Random

mutable struct SimpleSkipGram
    in_layer
    out_layer
    loss_layer1
    loss_layer2
    params
    grads
    word_vecs

    function SimpleSkipGram(vocab_size, hidden_size)
        self = new()
        V, H = vocab_size, hidden_size

        # 重みの初期化
        W_in = 0.01 .* Random.randn(V, H) # Float32の指定は特にしない
        W_out = 0.01 .* Random.randn(H, V) # Float32の指定は特にしない

        # レイヤの生成
        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer1 = SoftmaxWithLoss()
        self.loss_layer2 = SoftmaxWithLoss()

        # 全ての重みと勾配を配列にまとめる
        layers = [self.in_layer, self.out_layer]
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

function forward!(self::SimpleSkipGram, contexts, target)
    # contextsの形状は3次元決め打ち
    h = forward!(self.in_layer, target)
    s = forward!(self.out_layer, h)
    l1 = forward!(self.loss_layer1, s, contexts[:, 1, :])
    l2 = forward!(self.loss_layer2, s, contexts[:, 2, :])
    loss = l1 + l2
    return loss
end

function backward!(self::SimpleSkipGram; dout = 1)
    dl1 = backward!(self.loss_layer1, dout = dout)
    dl2 = backward!(self.loss_layer2, dout = dout)
    ds = dl1 + dl2
    dh = backward!(self.out_layer, ds)
    backward!(self.in_layer, dh)
end