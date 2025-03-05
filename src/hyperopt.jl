using ScheduleFree: optimize


function hyperparametersweep(rule::Symbol, f, x0, maxiters, hyperparams)
    # e.g. hyperparams = Dict(:γ => [0.1, 0.01], :β => [0.9, 0.99])
    param_combinations = collect(Iterators.product(values(hyperparams)...))
    hyperparam_keys = keys(hyperparams)
    named_param_combinations = [Dict(zip(hyperparam_keys,combo)) for combo in param_combinations]
    best_hyperparams = Dict()
    best_params = Inf
    best_params_value = Inf
    for params in named_param_combinations
        final_params = optimize(Val(rule), f, x0, maxiters; params...)
        final_params_value = f(final_params)
        if final_params_value < best_params_value
            best_params = final_params
            best_params_value = final_params_value
            best_hyperparams = params
        end
    end
    return best_hyperparams, best_params, best_params_value
end