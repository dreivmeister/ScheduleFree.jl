using ScheduleFree: optimize, hyperparametersweep

function f(x)
    return x[1]^2 + 2*x[2]^2
end
x0 = randn(2)

x_opt = optimize(:SF_SGD, f, x0, 1000)
@assert f(x_opt) < 1e-1 "SF_SGD error is too large: $(f(x_opt))"

x_opt = optimize(:SF_ADAMW, f, x0, 1000)
@assert f(x_opt) < 1e-1 "SF_ADAMW error is too large: $(f(x_opt))"

best_hyperparams, best_x, best_x_value = hyperparametersweep(:SF_SGD, f, x0, 100, Dict(:γ => [0.1, 0.01], :β => [0.9, 0.99]))
@assert best_x_value < 1e-3 "SF_SGD hyperparametersweep error is too large: $(best_x_value)"