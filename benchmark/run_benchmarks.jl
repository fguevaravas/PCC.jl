using BenchmarkTools, PCC 

function bench_pcc1d()
  results = Dict()
  for p ∈ [4,8,16], q ∈ [2,4,8,16,32,64]
   n = p*q
   k = randn(n,p)
   k̂ = pcc1d_fft_prep(k)
   x = randn(n)
   t1 = @benchmark pcc1d_fft($x,$k̂) 
   results[(p=p,q=q,alg="fft")] = t1

   k̂ = pcc1d_fft_blk_v2_prep(k)
   t2 = @benchmark pcc1d_fft_blk_v2($x,$k̂)
   results[(p=p,q=q,alg="blk_v2")] = t2
  end
  return results
end
