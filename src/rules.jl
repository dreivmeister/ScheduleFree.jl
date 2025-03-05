using Zygote


function optimize(::Val{:SF_SGD}, f, x0, max_iters, γ=1.0, β=0.9)
    grad = x -> Zygote.gradient(f, x)[1]
    t = 1
    z = x = x0

    for _ in 1:max_iters
      y = (1 - β) .* z + β .* x
      z = z - γ .* grad(y)
      c = 1 / (t + 1)
      x = (1 - c) .* x + c .* z
      t += 1
    end

    return x
end

function optimize(op::Symbol, args...)
  optimize(Val(op), args...)
end