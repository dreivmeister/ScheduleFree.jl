using Zygote


function optimize(::Val{:SF_SGD}, f, x0, max_iters, γ=1.0, β=0.9)
    grad = x -> Zygote.gradient(f, x)[1]
    z = x = x0

    for t in 1:max_iters
      yt = (1 - β) .* z + β .* x
      z = z - γ .* grad(yt)
      ct = 1 / (t + 1)
      x = (1 - ct) .* x + ct .* z
    end

    return x
end

function optimize(::Val{:SF_ADAMW}, f, x0, max_iters, γ=1.0, λ=0.9, β1=0.9, β2=0.999, ϵ=1e-8)
  grad = x -> Zygote.gradient(f, x)[1]
  z = x = x0
  v = zeros(size(x0))
  γs = 0.0

  for t in 1:max_iters
    yt = (1 - β1) .* z + β1 .* x
    gt = grad(yt)
    v = β2 .* v + (1 - β2) .* gt.^2
    γt = γ * sqrt(1 - β2^t)
    z = z - γt .* gt ./ (sqrt.(v) .+ ϵ) - γt * λ * yt
    γs += γt^2
    ct = γt^2 / γs
    x = (1 - ct) .* x + ct .* z
  end

  return x
end

function optimize(op::Symbol, args...)
  optimize(Val(op), args...)
end