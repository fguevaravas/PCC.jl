# Fast circular convolution with periodic kernels

This code reproduces the figures in __Fast circular convolution with periodic kernels__ by
Fernando González del Cueto and Fernando Guevara Vasquez.

For the associated preprint see: TBA.

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


## Funding
The authors received funding from Lumos Imaging Inc. FGV acknowledges funding from the National Science
Foundation DMS-2136198.

## License
Unless noted otherwise, the code is released under the following license (see also `LICENSE` file)
```
BSD 3-Clause License

Copyright (c) 2022, Trent DeGiovanni and Fernando Guevara Vasquez
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
