using Zygote: gradient
using Statistics: norm, mean

function optimize(::Val{:SF_SGD}, f, x0, maxiters; γ=1.0, β=0.9)
    grad = x -> gradient(f, x)[1]
    z = x = x0

    for t in 1:maxiters
      yt = (1 - β) .* z + β .* x
      z = z - γ .* grad(yt)
      ct = 1 / (t + 1)
      x = (1 - ct) .* x + ct .* z
    end

    return x
end

function optimize(::Val{:PRODIGY_SGD}, f, x0, maxiters; d0, G)
  @assert d0 > 0 "d0 must be >0"
  @assert G >= 0 "G must be >=0"
  grad = x -> gradient(f, x)[1]

  xs = [x0]
  ds = [d0]
  gs = []
  μs = []
  λs = []

  for k in 1:maxiters
    gk = grad(xs[end])
    push!(gs, gk)
    λk = 1.0
    push!(λs, λk)
    μk = (ds[end]^2 * λk) / sqrt(ds[end]^2 * G^2 + sum(ds.^2 .* λs.^2 .* norm.(gs).^2))
    push!(μs, μk)
    xk = xs[end] - μk * gk
    push!(xs, xk)
    dk_hat = sum(μs[i] * gs[i]' * (x0 - xs[i]) for i in 1:k) / norm(xk - x0)
    dk = max(dk_hat, ds[end])
    push!(ds, dk)
  end

  return mean(μs .* xs[2:end])
end

function optimize(::Val{:SF_ADAMW}, f, x0, maxiters; γ=0.0025, λ=0, β1=0.9, β2=0.999, ϵ=1e-8)
  grad = x -> gradient(f, x)[1]
  z = x = x0
  v = zeros(size(x0))
  γs = 0.0

  for t in 1:maxiters
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

function optimize(rule::Symbol, args...)
  optimize(Val(rule), args...)
end