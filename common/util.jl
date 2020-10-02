function preprocess(text)
    text = lowercase(text)
    text = replace(text, "." => " .")
    words = split(text, " ")

    word_to_id = Dict()
    id_to_word = Dict()

    for word = words
        if false == get(word_to_id, word, false)
            new_id = length(word_to_id)
            push!(word_to_id, word => new_id) # word_to_id = Dict(word_to_id..., word=>new_id) より高速(アロケーションは大きい)
            push!(id_to_word, new_id => word)
        end
    end

    corpus = [word_to_id[w] for w = words]

    return corpus, word_to_id, id_to_word
end