module Functions
export sigmoid, relu, softmax, cross_entropy_error

# TODO:関数の内部にdotを付けるのではなく，関数の呼び出し元にdotを付けるように変更
function sigmoid(x)
    return 1 ./ (1 .+ exp.(-x))
end

function relu(x)
    return max(0, x) # NOT maximum()
end


function softmax(x)
    y = []
    for row_i in 1:size(x, 1)
        c = maximum(x[row_i, :])
        exp_a = exp.(x[row_i, :] .- c) # オーバフロー対策
        push!(y, exp_a ./ sum(exp_a))
    end
    return hcat(y...)' # cf. https://docs.julialang.org/en/v1/manual/faq/?highlight=splat#What-does-the-...-operator-do?
end

function cross_entropy_error(y, t)
    return -sum(t .* log.(y .+ 1e-7)) # 0除算を回避するために微小な値を加算
end
end