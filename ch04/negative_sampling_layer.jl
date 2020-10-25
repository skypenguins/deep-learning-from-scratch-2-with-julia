include("../common/layers.jl")

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
    out = sum(target_W .* h, dims=2)
    self.cache = (h, target_W)
    return out
end

function backward!(self::EmbeddingDot, dout)
    h, target_W = self.cache
    dout = reshape(dout, size(dout, 1))

    dtarget_W = dout .* h
    backward!(self.embed, dtarget_W)
    dh = dout .* target_W
    return dh
end
