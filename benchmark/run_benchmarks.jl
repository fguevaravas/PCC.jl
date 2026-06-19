using BenchmarkTools, DataFrames, PyPlot, PCC

function bench_pcc1d_pfixed()
  results = DataFrame(p=Any[],q=Any[],algo=String[],bench=Any[])
  for p ∈ [8], q ∈ 2 .^ (1:10)
    n = p*q
    k = randn(n,p)
    k̂ = pcc1d_cc_prep(k)
    x = randn(n)
    t = @benchmark pcc1d_cc($(randn(n)),$k̂) samples=1000
    push!(results,(p,q,"cc",t))

    k̂ = pcc1d_improved_prep(k)
    t = @benchmark pcc1d_improved($(randn(n)),$k̂) samples=1000
    push!(results,(p,q,"improved",t))
  end
  return results
end

function bench_pcc1d_nfixed()
  results = DataFrame(p=Any[],q=Any[],algo=String[],bench=Any[])
  n = 1024
  for p ∈ 2 .^ (1:10)
    q = n ÷ p
    k = randn(n,p)
    k̂ = pcc1d_cc_prep(k)
    x = randn(n)
    t = @benchmark pcc1d_cc($(randn(n)),$k̂) samples=1000
    push!(results,(p,q,"cc",t))

    k̂ = pcc1d_improved_prep(k)
    t = @benchmark pcc1d_improved($(randn(n)),$k̂) samples=1000
    push!(results,(p,q,"improved",t))
  end
  return results
end

function bench_pcc2d_pfixed()
  results = DataFrame(p=Any[],q=Any[],algo=String[],bench=Any[])
  for p ∈ [(8,8)], q ∈ 2 .^ (1:7)
    qq = (q,q)
    n = p.*qq
    k = randn(n...,p...)
    k̂ = pcc2d_cc_prep(k)
    x = randn(n...)
    t = @benchmark pcc2d_cc($(randn(n...)),$k̂) samples=1000
    push!(results,(p,qq,"cc",t))

    k̂ = pcc2d_improved_prep(k)
    t = @benchmark pcc2d_improved($(randn(n...)),$k̂) samples=1000
    push!(results,(p,qq,"improved",t))
  end
  return results
end

function bench_pcc2d_nfixed()
  results = DataFrame(p=Any[],q=Any[],algo=String[],bench=Any[])
  n = (2^10,2^10)
  for p ∈ 2 .^ (1:5)
    pp = (p,p)
    q = n .÷ pp
    k = randn(n...,pp...)
    k̂ = pcc2d_cc_prep(k)
    x = randn(n...)
    t = @benchmark pcc2d_cc($(randn(n...)),$k̂) samples=1000
    push!(results,(pp,q,"cc",t))

    k̂ = pcc2d_improved_prep(k)
    t = @benchmark pcc2d_improved($(randn(n...)),$k̂) samples=1000
    push!(results,(pp,q,"improved",t))
  end
  return results
end

algos = [ "cc","improved"]
algo_legend = Dict( "cc" => "circular conv.", "improved"=>"improved")

# extract time (in nanoseconds) and memory usage (in bytes)
times(z) = [ mean(z[:,:bench][j].times) for j ∈ 1:size(z,1) ]
memory(z) = [ z[:,:bench][j].memory for j ∈ 1:size(z,1) ]
# extract meaningful dimensions from dataframe
ns(z) = [ z[j,:p][1]*z[j,:q][1] for j ∈ 1:size(z,1) ]
periods(z) = [ z[j,:p][1] for j ∈ 1:size(z,1) ]

# extract speedup for two algorithms alg1/alg2
function speedups(z,alg1,alg2)
 z1 = z[z.algo .== alg1,:]
 z2 = z[z.algo .== alg2,:]
 return times(z1)./times(z2)
end

# to put any savefig options
mysavefig(x) = savefig("$x.pdf", bbox_inches="tight")
figsize = (4,3)

function plot_pcc1d_pfixed(res)
  figure(1,figsize=figsize); clf()
  for algo ∈ algos
    z = res[res.algo .== algo,:]
    loglog(z[:,:p].*z[:,:q], times(z)/1e3, label=algo_legend[algo])
  end
  legend()
  xlabel("n")
  ylabel("time (μs)")
  mysavefig("pfixed_time_1d")

  figure(2,figsize=figsize); clf()
  for algo ∈ algos 
    z = res[res.algo .== algo,:]
    loglog(z[:,:p].*z[:,:q], memory(z), label=algo_legend[algo])
  end
  legend()
  xlabel("n")
  ylabel("memory (bytes)")
  mysavefig("pfixed_mem_1d")
end

function plot_pcc1d_nfixed(res)
  figure(1,figsize=figsize); clf()
  for algo ∈ algos 
    z = res[res.algo .== algo,:]
    loglog(z[:,:p], times(z)/1e3, label=algo_legend[algo])
  end

  legend()
  xlabel("p")
  ylabel("time (ms)")
  mysavefig("nfixed_time_1d")

  figure(2,figsize=figsize); clf()
  for algo ∈ algos 
    z = res[res.algo .== algo,:]
    loglog(z[:,:p], memory(z)/1e3, label=algo_legend[algo])
  end
  legend()
  xlabel("p")
  ylabel("memory (bytes)")
  mysavefig("nfixed_mem_1d")
end


function plot_pcc_speedups(res1p,res1n,res2p,res2n)
  figure(1,figsize=figsize); clf()
  z = res1p[res1p.algo .== "cc",:]
  semilogx(z[:,:p].*z[:,:q],speedups(res1p,"cc","improved"),label="1D")
  z = res2p[res2p.algo .== "cc",:]
  semilogx(ns(z),speedups(res2p,"cc","improved"),label="2D")
  xlabel("n"); ylim(bottom=0)
  legend()
  mysavefig("pfixed_speedup")

  figure(2,figsize=figsize); clf()
  z = res1n[res1n.algo .== "cc",:]
  semilogx(z[:,:p],speedups(res1n,"cc","improved"),label="1D")
  z = res2n[res2n.algo .== "cc",:]
  semilogx(periods(z),speedups(res2n,"cc","improved"),label="2D")
  xlabel("p"); ylim(bottom=0)
  legend()
  mysavefig("nfixed_speedup")

end
function plot_pcc2d_pfixed(res)
  figure(1,figsize=figsize); clf()
  for algo ∈ algos 
    z = res[res.algo .== algo,:]
    loglog(ns(z), times(z)/1e6, label=algo_legend[algo])
  end
  legend()
  xlabel("n")
  ylabel("time (ms)")
  mysavefig("pfixed_time_2d")

  figure(2,figsize=figsize); clf()
  for algo ∈ algos 
    z = res[res.algo .== algo,:]
    loglog(ns(z), memory(z), label=algo_legend[algo])
  end
  legend()
  xlabel("n")
  ylabel("memory (bytes)")
  mysavefig("pfixed_mem_2d")
end

function plot_pcc2d_nfixed(res)
  figure(1,figsize=figsize); clf()
  for algo ∈ algos 
    z = res[res.algo .== algo,:]
    loglog(periods(z), times(z)/1e6, label=algo_legend[algo])
  end
  legend()
  xlabel("n")
  ylabel("time (ms)")
  mysavefig("nfixed_time_2d")

  figure(2,figsize=figsize); clf()
  for algo ∈ algos 
    z = res[res.algo .== algo,:]
    loglog(periods(z),memory(z), label=algo_legend[algo])
  end
  legend()
  xlabel("n")
  ylabel("memory (bytes)")
  mysavefig("nfixed_mem_2d")
end

function bench_all()
 res1p=bench_pcc1d_pfixed()
 res1n=bench_pcc1d_nfixed()
 res2p=bench_pcc2d_pfixed()
 res2n=bench_pcc2d_nfixed()
 return (res1p,res1n,res2p,res2n)
end

function plot_all(res1p,res1n,res2p,res2n)
 plot_pcc1d_pfixed(res1p)
 plot_pcc1d_nfixed(res1n)
 plot_pcc2d_pfixed(res2p)
 plot_pcc2d_nfixed(res2n)
 plot_pcc_speedups(res1p,res1n,res2p,res2n)
end
