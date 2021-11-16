include("../common/layers.jl")
include("../ch04/negative_sampling_layer.jl")

using Random

mutable struct CBOW
    in_layers
    ns_loss
    params
    grads
    word_vecs
    function CBOW(vocab_size, hidden_size, windows_size, corpus)
        self = new()
        V, H = vocab_size, hidden_size

        # 重みの初期化
        W_in = 0.01 .* randn(V, H)
        W_out = 0.01 .* randn(V, H)

        # レイヤの生成
        self.in_layers = []
        for i = 1:(2 .* windows_size)
            layer = Embedding(W_in)
            push!(self.in_layers, layer)
        end
        self.ns_loss = NegativeSamplingLoss(W_out, corpus; power = 0.75, sample_size = 5)

        # 全ての重みと勾配を配列にまとめる
        layers = vcat(self.in_layers, self.ns_loss)

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

function forward!(self::CBOW, contexts, target)
    h = 0
    for (i, layer) in enumerate(self.in_layers)
        h .+= forward!(layer, contexts[:, i])
    end
    h .*= 1 ./ length(self.in_layers)
    loss = forward!(self.ns_loss, h, target)
    return loss
end

function backward!(self::CBOW; dout = 1)
    dout = backward!(self.ns_loss, dout = dout)
    dout .*= 1 ./ length(self.in_layers)
    for layer in self.in_layers
        backward!(layer, dout = dout)
    end
end