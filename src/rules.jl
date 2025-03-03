using Random, LinearAlgebra

# Schedule-Free SGD implementation
function schedule_free_sgd(grad, x0; γ=0.01, β=0.9, max_iters=100)
    t = 1
    z = x = x0

    for _ in 1:max_iters
        y = (1 - β) * z + β * x
        z = z - γ * grad(y)
        c = 1 / (t + 1)
        x = (1 - c) * x + c * z
        t += 1
    end

    return x
end

# Example usage with a quadratic loss function
function quadratic_grad(x)
    return 2 * x  # Gradient of f(x) = x^2
end

# Initial point
x0 = randn(2)

# Run Schedule-Free SGD
x_opt = schedule_free_sgd(quadratic_grad, x0, γ=0.1, β=0.9, max_iters=100)
println("Optimized x: ", x_opt)


@def struct ScheduleFreeSGD <: AbstractRule 
  γ = 0.01 # Learning rate for SGD updates 
  β = 0.9 # Interpolation factor 
end

function init(o::ScheduleFreeSGD, x::AbstractArray) 
  return (z=copy(x), t=1) 
end

function apply!(o::ScheduleFreeSGD, state, x::AbstractArray{T}, dx) where T
  γ, β = T(o.γ), T(o.β)
  # Unpack state
  z, t = state.z, state.t

  # Compute update steps
  y = (1 - β) * z + β * x
  z -= γ * dx  # SGD step
  c = 1 / (t + 1)
  x .= (1 - c) * x + c * z  # Weighted averaging

  # Update state
  state.z .= z
  state.t += 1

  return state, x
end
