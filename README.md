# Periodic Circular Convolution
A collection of functions to calculate the periodic circular convolution

Authors: Fernando González del Cueto and Fernando Guevara Vasquez

## Tests
To run tests, in Julia and from the directory `PCC.jl`
```
] activate .
] test
```

## Benchmarks
The benchmarks require 28GB just for Julia. They were run on a 32GB machine. To run the benchmarks, in Julia cd to the directory `PCC.j/benchmark`
```
]activate .
include("run_benchmarks.jl")
res = bench_all()
```

To reproduce the plots:
```
plot_all(res...)
```

The results can be saved using JLD2, which HDF5 based. For example we are using the file `bench.jld2`
```
using JLD2
jldsave("bench.jld2";res)
```

Results can be loaded as follows. The instructions are written assuming this is a new Julia session. First cd to `PCC.jl/benchmark`
```
]activate .
include("run_benchmarks.jl")
using JLD2
res=load("bench.jld2","res")
plot_all(res...)
```

The benchmarks folder also contains the following code:
* `simple1d.jl`: simple 1D example. this is a script, `include("simple1d.jl")
  to run
* `simple2d.jl`: simple 1D example. this is a script, `include("simple2d.jl")
  to run
* `deconv.jl`: deconvolution example. include it and use `run_deconv()` to run
* `greenper.jl`: example of Green function for a periodic network. use
  `run_greenper2d()` to run
* `digits.jl`: this file is included in `simple2d.jl` and `deconv.jl`. It
  defines a (3,3)-periodic kernel with digits from 0 to 9
