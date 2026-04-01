using PCC, Test, LinearAlgebra

relerr(a,b) = norm(a-b)/norm(a)

function test_cc1d(n=11)
  x = randn(n); k = randn(n)
  y1,y2 = cc1d(x,k)
  println(norm(y1-y2))
end

pcc1d_functions = 
         [ (apply=pcc1d_fft, prep=pcc1d_fft_prep),
           (apply=pcc1d_fft_blk_v0, prep=pcc1d_fft_blk_v0_prep),
           (apply=pcc1d_fft_blk_v1, prep=pcc1d_fft_blk_v1_prep),
           (apply=pcc1d_fft_blk_v2, prep=pcc1d_fft_blk_v2_prep) ]

pcc2d_functions = 
         [ (apply=pcc2d_fft, prep=pcc2d_fft_prep),
           (apply=pcc2d_fft_blk_v0, prep=pcc2d_fft_blk_v0_prep),
           (apply=pcc2d_fft_blk_v1, prep=pcc2d_fft_blk_v1_prep),
           (apply=pcc2d_fft_blk_v2, prep=pcc2d_fft_blk_v2_prep) ]


# Tests all implementations against naive implementation
function test_pcc(naive=pcc1d_naive,other=pcc1d_functions,n=12,p=3)
  k = randn(n...,p...)
  x = randn(n...)
  yref = naive(x,k)
  @testset "pcc n=$n, p=$p" begin
    for f ∈ other 
      k̂ = f.prep(k)
      y = f.apply(x,k̂)
      @test relerr(yref,y) ≈ 0 atol=1e-10
    end
  end
end

function test_pcc_all()
 # 1D tests
 test_pcc(pcc1d_naive,pcc1d_functions,12,3)
 test_pcc(pcc1d_naive,pcc1d_functions,12,12) # p = n
 test_pcc(pcc1d_naive,pcc1d_functions,100,5)
 test_pcc(pcc1d_naive,pcc1d_functions,64,8)

 # 2D tests
 test_pcc(pcc2d_naive,pcc2d_functions,(27,32),(3,4))
 test_pcc(pcc2d_naive,pcc2d_functions,(32,27),(4,3))
 test_pcc(pcc2d_naive,pcc2d_functions,(32,32),(32,32)) # p =n
 test_pcc(pcc2d_naive,pcc2d_functions,(32,27),(32,27)) # p =n
end

test_pcc_all()
