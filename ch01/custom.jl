# モデル定義ファイルの読み込み
@show include("../common/optimizer.jl")
@show include("../dataset/spiral.jl")
@show include("./two_layer_net.jl")

# モジュールの読み込み
using .Optimizer
using .Spiral
using .Two_layer_net
using Random
# using Plots

# ハイパーパラメータの設定
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

# 学習で使用する変数
data_size = size(x, 1)
max_iters = data_size ÷ batch_size # 整数除算「÷」は「\div」で入力 cf. https://docs.julialang.org/en/v1/manual/unicode-input/
total_loss = 0
loss_count = 0
loss_list = []

# Jupyter(IJulia)上では，ローカルスコープ内からグローバルスコープの変数を変更できる(v1.5以降も同様)
for epoch = 1:max_epoch
    # データのシャッフル
    idx = shuffle(1:data_size) # 1～data_sizeの範囲の一意な数値を要素に持つ要素数data_sizeのVectorを返す
    x = x[idx, :]
    t = t[idx, :]
    
    for iters = 0:max_iters - 1
        batch_x = x[(iters * batch_size) + 1:(iters + 1) * batch_size, :]
        batch_t = t[(iters * batch_size) + 1:(iters + 1) * batch_size, :]
        
        # 勾配を求め，パラメータを更新
        loss = forward(model, batch_x, batch_t) # model.forward()
        backward(model) # model.backward()
        update(optimizer, model.params, model.grads)
        
        total_loss += loss
        loss_count += 1
        
        # 定期的に学習経過を出力
        if iters % 10 == 0
            avg_loss = total_loss / loss_count
            println("| epoch $epoch | iter $(iters + 1) / $max_iters | loss $(round(avg_loss, 2))")
            vcat(loss_list, avg_loss)
            total_loss, loss_count = 0, 0
        end
    end
end