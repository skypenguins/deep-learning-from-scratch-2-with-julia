include("../common/layers.jl")

using StatsBase

mutable struct EmbeddingDot
    embed
    params
    grads
    cache
    function EmbeddingDot(W)
        self = new()
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = nothing
        return self
    end
end

function forward!(self::EmbeddingDot, h, idx)
    target_W = forward!(self.embed, idx)
    out = sum(target_W .* h, dims = 2)
    self.cache = (h, target_W)
    return out
end

function backward!(self::EmbeddingDot, dout)
    h, target_W = self.cache
    dout = reshape(dout, size(dout, 1), 1)

    dtarget_W = dout .* h
    backward!(self.embed, dtarget_W)
    dh = dout .* target_W
    return dh
end

mutable struct UnigramSampler
    sample_size
    vocab_size
    word_p
    function UnigramSampler(corpus, power, sample_size)
        self = new()
        self.sample_size = sample_size
        self.vocab_size = nothing
        self.word_p = nothing

        counts = Dict()
        for word_id in corpus
            if nothing == get(counts, word_id, nothing)
                get!(counts, word_id, 1)
            else
                counts[word_id] += 1
            end
        end

        vocab_size = length(counts)
        self.vocab_size = vocab_size

        self.word_p = zeros(vocab_size)
        for i = 1:vocab_size
            self.word_p[i] = counts[i]
        end

        self.word_p = self.word_p .^ power
        self.word_p ./= sum(self.word_p)

        return self
    end
end

function get_negative_sample(self::UnigramSampler, target)
    batch_size = size(target, 1)

    negative_sample = zeros(Int32, batch_size, self.sample_size)

    for i = 1:batch_size
        p = copy(self.word_p)
        target_idx = target[i]
        p[target_idx] = 0
        p ./= sum(p)
        negative_sample[i, :] = sample(self.vocab_size, ProbabilityWeights(p), self.sample_size, replace = false)
    end

    # TODO: GPUで処理する際の演算

    return negative_sample
end

mutable struct NegativeSamplingLoss <: AbstractLayer
    sample_size
    sampler
    loss_layers
    embed_dot_layers
    params
    grads
    function NegativeSamplingLoss(W, corpus; power = 0.75, sample_size = 5)
        self = new()
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ = 1:(sample_size+1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ = 1:(sample_size+1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers
            push!(self.params, layer.params)
            push!(self.grads, layer.grads)
        end
        return self
    end
end

function forward!(layer::NegativeSamplingLoss, h, target)
    batch_size = size(target, 1)
    negative_sample = get_negative_sample(layer.sampler, target)

    # 正例の順伝播
    score = forward!(layer.embed_dot_layers[1], target)
    correct_label = ones(Int32, batch_size)
    loss = forward!(layer.loss_layers[1], score, correct_label)

    # 負例の順伝播
    negative_label = zeros(Int32, batch_size)
    for i = 1:sample_size
        negative_target = negative_sample[:, i]
        score = forward!(layer.embed_dot_layers[1+i], h, negative_target)
        loss .+= forward!(layer.loss_layers[1+i], score, negative_label)
    end

    return loss
end

function backward!(layer::NegativeSamplingLoss; dout = 1)
    dh = 0
    for (l0, l1) in zip(layer.loss_layers, layer.embed_dot_layers)
        dscore = backward!(l0, dout)
        hf .+= backward!(l1, dscore)
    end

    return dh
end