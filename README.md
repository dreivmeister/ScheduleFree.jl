# ScheduleFree.jl
Implementation of Schedule-Free optimization algorithms for ML in Julia.
The official PyTorch version is here: https://github.com/facebookresearch/schedule_free/tree/main


The idea is simple. Provide a differentiable $f: R^n \to R$ to the `optimize(...)` function with some additional hyperparameters and let it run.
For ML $f$ would need to be a function which takes the parameters of the model and returns a loss value based on data. This would need to be handled in the function.
For a standard optimization task this is just the objective function which returns a scalar value, possibly based on some other function like: $f(x) = sum(g(x))$ with $g: R^n \to R^m$.