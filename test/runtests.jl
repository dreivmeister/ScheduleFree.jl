# import the package
using ScheduleFree


function f(x)
    return x[1]^2 + 2*x[2]^2
end

x0 = randn(2)
x_opt = optimize(:SF_SGD, f, x0, 1000)
@assert f(x_opt) < 1e-1