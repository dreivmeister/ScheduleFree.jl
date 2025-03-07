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

function optimize(::Val{:PRODIGY_SGD}, f, x0, maxiters; d0=1e-6, G=1.0)
  @assert d0 > 0 "d0 must be >0"
  @assert G >= 0 "G must be >=0"
  grad = x -> gradient(f, x)[1]

  xs = [x0]
  ds = [d0]
  gs = []
  μs = [1.0]
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

  return mean(μs .* xs)
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

function optimize(::Val{:PRODIGY_ADAM}, f, x0, maxiters; γ=1.0, d0=1e-6, β1=0.9, β2=0.999, ϵ=1e-8)
  @assert d0 > 0 "d0 must be >0"
  grad = x -> gradient(f, x)[1]

  x = copy(x0)
  d = d0
  r = 0.0
  s = zeros(size(x0))
  m = zeros(size(x0))
  v = zeros(size(x0))

  for k in 1:maxiters
    gk = grad(x)
    m = β1 * m + (1 - β1) * d * gk
    v = β2 * v + (1 - β2) * d^2 * gk.^2
    r = sqrt(β2) * r + (1 - sqrt(β2)) * γ * d^2 * gk' * (x0 - x)
    s = sqrt(β2) * s + (1 - sqrt(β2)) * γ * d^2 * gk
    dk_hat = r / norm(s, 1)
    dk = max(d, dk_hat)
    x = x - γ * d * m ./ (sqrt.(v) .+ d * ϵ)
    d = dk
  end

  return x
end