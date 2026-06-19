using PCC, Test, LinearAlgebra, FFTW

relerr(a,b) = norm(a-b)/norm(a)

pcc1d_mv_functions = 
         [ (apply=pcc1d_gemv_cc, prep=pcc1d_gemv_cc_prep),
           (apply=pcc1d_gemv_improved, prep=pcc1d_gemv_improved_prep),
           (apply=pcc1d_gemv_blkm, prep=pcc1d_gemv_blkm_prep)
         ]

pcc2d_mv_functions = 
         [ (apply=pcc2d_gemv_cc, prep=pcc2d_gemv_cc_prep),
           (apply=pcc2d_gemv_improved, prep=pcc2d_gemv_improved_prep),
           (apply=pcc2d_gemv_blkm, prep=pcc2d_gemv_blkm_prep)
         ]


# Tests all implementations against naive implementation
function test_pcc(naive=pcc1d_gemv_naive,other=pcc1d_mv_functions,n=12,p=3)
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

# makes sure that the pcc operations corresponds to the 
# circular convolution operation in the special case where p=1 or (1,1)
function test_pcc_and_cc(pcc=pcc1d_gemv_naive,n=12)
 k = randn(n...)
 x = randn(n...)
 kt = length(n) == 1 ? reshape(k,n...,1) : reshape(k,n...,1,1)
 y1 = pcc(x,kt)
 y2 = ifft(fft(x) .* fft(k))
 @test relerr(y2,y1) ≈ 0 atol=1e-10
end

# TODO: add test for consistency with circular convolution (p=1)

function test_pcc_all()
 # 1D tests
 test_pcc_and_cc(pcc1d_gemv_naive,12)
 test_pcc_and_cc(pcc1d_gemv_naive,15)
 test_pcc(pcc1d_gemv_naive,pcc1d_mv_functions,12,1)
 test_pcc(pcc1d_gemv_naive,pcc1d_mv_functions,12,3)
 test_pcc(pcc1d_gemv_naive,pcc1d_mv_functions,12,12) # p = n
 test_pcc(pcc1d_gemv_naive,pcc1d_mv_functions,100,5)
 test_pcc(pcc1d_gemv_naive,pcc1d_mv_functions,64,8)

 # 2D tests
 test_pcc_and_cc(pcc2d_gemv_naive,(12,15))
 test_pcc_and_cc(pcc2d_gemv_naive,(15,12))
 test_pcc(pcc2d_gemv_naive,pcc2d_mv_functions,(27,32),(1,1))
 test_pcc(pcc2d_gemv_naive,pcc2d_mv_functions,(32,27),(1,1))
 test_pcc(pcc2d_gemv_naive,pcc2d_mv_functions,(27,32),(3,4))
 test_pcc(pcc2d_gemv_naive,pcc2d_mv_functions,(32,27),(4,3))
 test_pcc(pcc2d_gemv_naive,pcc2d_mv_functions,(32,32),(32,32)) # p =n
 test_pcc(pcc2d_gemv_naive,pcc2d_mv_functions,(32,27),(32,27)) # p =n
end

test_pcc_all()
