using BenchmarkTools, DataFrames, PyPlot, PCC

function bench_pcc1d_pfixed()
  results = DataFrame(p=Any[],q=Any[],algo=String[],bench=Any[])
  for p ∈ [8], q ∈ 2 .^ (1:10)
    n = p*q
    k = randn(n,p)
    k̂ = pcc1d_fft_prep(k)
    x = randn(n)
    t = @benchmark pcc1d_fft($(randn(n)),$k̂) samples=1000
    push!(results,(p,q,"fft",t))

    k̂ = pcc1d_fft_blk_v1_prep(k)
    t = @benchmark pcc1d_fft_blk_v1($(randn(n)),$k̂) samples=1000
    push!(results,(p,q,"blk_v1",t))
  end
  return results
end

function bench_pcc1d_nfixed()
  results = DataFrame(p=Any[],q=Any[],algo=String[],bench=Any[])
  n = 1024
  for p ∈ 2 .^ (1:10)
    q = n ÷ p
    k = randn(n,p)
    k̂ = pcc1d_fft_prep(k)
    x = randn(n)
    t = @benchmark pcc1d_fft($(randn(n)),$k̂) samples=1000
    push!(results,(p,q,"fft",t))

    k̂ = pcc1d_fft_blk_v1_prep(k)
    t = @benchmark pcc1d_fft_blk_v1($(randn(n)),$k̂) samples=1000
    push!(results,(p,q,"blk_v1",t))
  end
  return results
end

function bench_pcc2d_pfixed()
  results = DataFrame(p=Any[],q=Any[],algo=String[],bench=Any[])
  for p ∈ [(8,8)], q ∈ 2 .^ (1:7)
    qq = (q,q)
    n = p.*qq
    k = randn(n...,p...)
    k̂ = pcc2d_fft_prep(k)
    x = randn(n...)
    t = @benchmark pcc2d_fft($(randn(n...)),$k̂) samples=1000
    push!(results,(p,qq,"fft",t))

    k̂ = pcc2d_fft_blk_v1_prep(k)
    t = @benchmark pcc2d_fft_blk_v1($(randn(n...)),$k̂) samples=1000
    push!(results,(p,qq,"blk_v1",t))
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
    k̂ = pcc2d_fft_prep(k)
    x = randn(n...)
    t = @benchmark pcc2d_fft($(randn(n...)),$k̂) samples=1000
    push!(results,(pp,q,"fft",t))

    k̂ = pcc2d_fft_blk_v1_prep(k)
    t = @benchmark pcc2d_fft_blk_v1($(randn(n...)),$k̂) samples=1000
    push!(results,(pp,q,"blk_v1",t))
  end
  return results
end

# BenchmarkTools stores results in nanoseconds
function plot_pcc1d_pfixed(res)
  alg1 = res[res.algo .== "fft",:]
  alg2 = res[res.algo .== "blk_v1",:]

  figure(1); clf()
  loglog(alg1[:,:p].*alg1[:,:q],[ mean(alg1[:,:bench][j].times) for j ∈ 1:size(alg1,1) ]/1e3,label="fft")
  loglog(alg2[:,:p].*alg2[:,:q],[ mean(alg2[:,:bench][j].times) for j ∈ 1:size(alg2,1) ]/1e3,label="improved")
  legend()
  xlabel("n")
  ylabel("time (μs)")
  savefig("pfixed_time_1d.png")

  figure(2); clf()
  loglog(alg1[:,:p].*alg1[:,:q],[ alg1[:,:bench][j].memory for j ∈ 1:size(alg1,1) ],label="fft")
  loglog(alg2[:,:p].*alg2[:,:q],[ alg2[:,:bench][j].memory for j ∈ 1:size(alg2,1) ],label="improved")
  legend()
  xlabel("n")
  ylabel("memory (bytes)")
  savefig("pfixed_mem_1d.png")
end

function plot_pcc1d_nfixed(res)
  alg1 = res[res.algo .== "fft",:]
  alg2 = res[res.algo .== "blk_v1",:]

  figure(1); clf()
  loglog(alg1[:,:p],[ mean(alg1[:,:bench][j].times) for j ∈ 1:size(alg1,1) ]/1e6,label="fft")
  loglog(alg2[:,:p],[ mean(alg2[:,:bench][j].times) for j ∈ 1:size(alg2,1) ]/1e6,label="improved")
  legend()
  xlabel("p")
  ylabel("time (ms)")
  savefig("nfixed_time_1d.png")

  figure(2); clf()
  loglog(alg1[:,:p],[ alg1[:,:bench][j].memory for j ∈ 1:size(alg1,1) ],label="fft")
  loglog(alg2[:,:p],[ alg2[:,:bench][j].memory for j ∈ 1:size(alg2,1) ],label="improved")
  legend()
  xlabel("p")
  ylabel("memory (bytes)")
  savefig("nfixed_mem_1d.png")
end

function plot_pcc2d_pfixed(res)
  alg1 = res[res.algo .== "fft",:]
  alg2 = res[res.algo .== "blk_v1",:]

  figure(1); clf()
  loglog([ alg1[j,:p][1]*alg1[j,:q][1] for j ∈ 1:size(alg1,1) ] ,
         [ mean(alg1[:,:bench][j].times) for j ∈ 1:size(alg1,1) ]/1e6,
         label="fft")
  loglog([ alg2[j,:p][1]*alg2[j,:q][1] for j ∈ 1:size(alg2,1) ] ,
         [ mean(alg2[:,:bench][j].times) for j ∈ 1:size(alg2,1) ]/1e6,
         label="improved")
  legend()
  xlabel("n")
  ylabel("time (ms)")
  savefig("pfixed_time_2d.png")

  figure(2); clf()
  loglog([ alg1[j,:p][1]*alg1[j,:q][1] for j ∈ 1:size(alg1,1) ] ,
         [ alg1[:,:bench][j].memory for j ∈ 1:size(alg1,1) ],
         label="fft")
  loglog([ alg2[j,:p][1]*alg2[j,:q][1] for j ∈ 1:size(alg2,1) ] ,
         [ alg2[:,:bench][j].memory for j ∈ 1:size(alg2,1) ],
         label="improved")
  legend()
  xlabel("n")
  ylabel("memory (bytes)")
  savefig("pfixed_mem_2d.png")
end

function plot_pcc2d_nfixed(res)
  alg1 = res[res.algo .== "fft",:]
  alg2 = res[res.algo .== "blk_v1",:]

  figure(1); clf()
  loglog([ alg1[j,:p][1] for j ∈ 1:size(alg1,1) ] ,
         [ mean(alg1[:,:bench][j].times) for j ∈ 1:size(alg1,1) ]/1e6,
         label="fft")
  loglog([ alg2[j,:p][1] for j ∈ 1:size(alg2,1) ] ,
         [ mean(alg2[:,:bench][j].times) for j ∈ 1:size(alg2,1) ]/1e6,
         label="improved")
  legend()
  xlabel("n")
  ylabel("time (ms)")
  savefig("nfixed_time_2d.png")

  figure(2); clf()
  loglog([ alg1[j,:p][1] for j ∈ 1:size(alg1,1) ] ,
         [ alg1[:,:bench][j].memory for j ∈ 1:size(alg1,1) ],
         label="fft")
  loglog([ alg2[j,:p][1] for j ∈ 1:size(alg2,1) ] ,
         [ alg2[:,:bench][j].memory for j ∈ 1:size(alg2,1) ],
         label="improved")
  legend()
  xlabel("n")
  ylabel("memory (bytes)")
  savefig("nfixed_mem_2d.png")
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
end
