using HTTP
using PyCall
using NPZ
# pikcleを扱うためのPythonのモジュールの読み込み
pickle = pyimport("pickle")

url_base = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/"
key_file = Dict(
    "train" => "ptb.train.txt",
    "test" => "ptb.test.txt",
    "valid" => "ptb.valid.txt"
)
save_file = Dict(
    "train" => "ptb.train.npy",
    "test" => "ptb.test.npy",
    "valid" => "ptb.valid.npy"
)
vocab_file = "ptb.vocab.pkl"

dataset_dir = dirname(@__FILE__)

function _download(file_name)
    file_path = dataset_dir * "/" * file_name
    if isfile(file_path)
        return
    end

    println("Downloading " * file_name * " ...")

    try
        HTTP.download(url_base * file_name, file_path)
    catch e
        println(e)
        return
    end

    println("Done!")
end

function load_vocab()
    vocab_path = dataset_dir * "/" * vocab_file

    # 既にツリーバンクがダウンロードされている場合
    if isfile(vocab_path)
        @pywith pybuiltin("open")(vocab_path, "rb") as f begin
            word_to_id, id_to_word = pickle.load(f)
            return word_to_id, id_to_word
        end
    end

    word_to_id = Dict()
    id_to_word = Dict()
    data_type = "train"
    file_name = key_file[data_type]
    file_path = dataset_dir * "/" * file_name

    _download(file_name)

    f = open(file_path)
    words = split(strip(replace(read(f, String), "\n" => "<eos>")))
    close(f)

    for (i, word) = enumerate(words)
        if false == get(word_to_id, word, false)
            tmp_id = length(word_to_id) + 1
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word
        end
    end

    f = open(vocab_path, "w")
    pickle.dump((word_to_id, id_to_word), f)
    close(f)

    return word_to_id, id_to_word
end

function load_data(;data_type="train")
    #= 
    パラメータ:
    data_type データの種類（"train" or "test" or "valid (val)"）

    返り値:
    なし =#
    if data_type == "val" data_type = "valid" end
    save_path = dataset_dir * "/" * save_file[data_type]

    word_to_id, id_to_word = load_vocab()

    if isfile(save_path)
        corpus = npzread(save_path)
        return corpus, word_to_id, id_to_word
    end

    file_name = key_file[data_type]
    file_path = dataset_dir * "/" * file_name
    _download(file_name)

    f = open(file_path)
    words = split(strip(replace(read(f, String), "\n" => "<eos>")))
    close(f)
    corpus = [word_to_id[w] for w = words]

    npzwrite(save_path, corpus)
    return corpus, word_to_id, id_to_word
end

for data_type in ("train", "val", "test")
    load_data(data_type=data_type)
end