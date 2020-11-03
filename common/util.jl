using LinearAlgebra

function preprocess(text)
    text = lowercase(text)
    text = replace(text, "." => " .")
    words = split(text, " ")

    word_to_id = Dict()
    id_to_word = Dict()

    for word = words
        if false == get(word_to_id, word, false)
            new_id = length(word_to_id) + 1     # 単語IDは配列のインデックス番号として使用するため0を含まないようにする
            push!(word_to_id, word => new_id)   # word_to_id = Dict(word_to_id..., word=>new_id) より高速(アロケーションは大きい)
            push!(id_to_word, new_id => word)
        end
    end

    corpus = [word_to_id[w] for w = words]
    return corpus, word_to_id, id_to_word
end

function create_co_matrix(corpus, vocab_size; windows_size=1)
    """
    共起行列の作成

    パラメータ:
    corpus コーパス（単語IDのリスト）
    vocab_size 語彙の大きさ
    windows_size 窓の大きさ（これが1の場合，単語の左右1単語がコンテキスト）
    
    返り値:
    co_matrix 共起行列
    """

    corpus_size = length(corpus)
    co_matrix = zeros(Int32, (vocab_size, vocab_size)...)

    for (idx, word_id) = enumerate(corpus)
        for i = 1:windows_size
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 1
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            end

            if right_idx <= corpus_size
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
            end
        end
    end

    return co_matrix
end

function cos_similarity(x, y; ϵ=1e-8)
    """
    コサイン類似度の算出
    パラメータ:
    x ベクトル
    y ベクトル
    ϵ ゼロ除算の防止のための定数

    返り値:
    内積
    """
    nx = x ./ (sqrt(sum(x.^2)) .+ ϵ)
    ny = y ./ (sqrt(sum(y.^2)) .+ ϵ)
    return dot(nx, ny)
end

function most_similar(query, word_to_id, id_to_word, word_matrix; top=5)
    """
    類似単語の検索
    パラメータ:
    query クエリ（テキスト）
    word_to_id 単語から単語IDへのディクショナリ
    id_to_word 単語IDから単語へのディクショナリ
    word_matrix 単語ベクトルをまとめた行列。各行に対応する単語のベクトルが格納されていることを想定
    top 上位何位まで表示するか
    """

    # 1.クエリを取り出す
    if false == get(word_to_id, query, false)
        println("$(query) is not found")
        return
    end

    println("[query] $(query)")
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id, :]

    # 2.コサイン類似度の算出
    vocab_size = length(id_to_word)
    similarity = zeros(vocab_size)
    for i = 1:vocab_size
        similarity[i] = cos_similarity(word_matrix[i, :], query_vec)
    end

    # 3.コサイン類似度の結果から，その値を降順で出力
    count = 0
    for i =  sortperm(-1 .* similarity)
        if id_to_word[i] == query
            continue
        end

        println(" $(id_to_word[i]), $(similarity[i])")

        count += 1
        if count >= top
            return
        end
    end
end

function ppmi(C; verbose=false, ϵ=1e-8)
    """
    正の相互情報量（PPMI）の作成
    パラメータ:
    C 共起行列
    verbose 進捗を表示するか
    ϵ ゼロ除算回避のための定数

    返り値:
    M PPMI行列
    """
    M = zero(C)
    N = sum(C)
    S = sum(C, dims=1)
    total = size(C, 1) .* size(C, 2)
    cnt = 0

    for i = 1:size(C, 1)
        for j = 1:size(C, 2)
            pmi = log2(C[i, j] .* N ./ (S[j] .* S[i]) + ϵ)
            M[i, j] = max(0, pmi...)

            if verbose == true
                cnt += 1
                if cnt % (total ÷ 100) == 0
                    println("$(round((100 .* cnt / total); digits=1))% done")
                end
            end
        end
    end
    return M
end

function create_contexts_target(corpus; window_size=1)
    """
    コンテキストとターゲットの作成

    パラメータ:
    corpus コーパス（単語IDのリスト）
    windows_size 窓の大きさ（これが1の場合，単語の左右1単語がコンテキスト）

    返り値:
    contexts コーパス中の全単語IDのコンテキスト
    target ターゲットとする単語ID
    """
    target = corpus[window_size + 1:end - window_size]
    cs = []

    for idx = (window_size + 1):(length(corpus) - window_size) # window_size=1の場合，開始のインデックス番号は2
        # 各単語のコンテキストを取り出し，csに追加
        push!(cs, corpus[filter(!iszero, -window_size:window_size) .+ idx]')
    end
    # 全単語のコンテキストを作成
    contexts = vcat(cs...)
    return contexts, target
end

function convert_one_hot(corpus, vocab_size)
    """
    one-hot表現への変換

    パラメータ:
    corpus 単語IDのリスト（1次元もしくは2次元のNumPy配列）
    vocab_size 語彙の大きさ

    返り値
    one-hot表現（1次元もしくは2次元のNumPy配列）
    """
    N = size(corpus, 1)

    if ndims(corpus) == 1
        one_hot = zeros(Int32, N, vocab_size)
        for (idx, word_id) = enumerate(corpus)
            one_hot[idx, word_id] = 1
        end
    elseif ndims(corpus) == 2
        C = size(corpus, 2)
        one_hot = zeros(Int32, N, C, vocab_size)
        for i = 1:N
            for j = 1:C
                one_hot[i, j, corpus[i, j]] = 1
            end
        end
    end
    return one_hot
end