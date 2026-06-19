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

# tests basic matrix matrix product
function test_gemm(gemv=pcc1d_gemv_blkm,
                   gemv_prep=pcc1d_gemv_blkm_prep,
                   gemm=pcc1d_gemm,n=12,p=3)
 a = randn(n...,p...)
 b = randn(n...,p...)
 x = randn(n...)
 â = gemv_prep(a)
 b̂ = gemv_prep(b)
 ĉ = gemm(â,b̂)

 cx1 = gemv(gemv(x,b̂),â)
 cx2 = gemv(x,ĉ)

 @test relerr(cx1,cx2) ≈ 0 atol=1e-10 
end

# tests basic solve
function test_solve(gemv=pcc1d_gemv_blkm,
                   gemv_prep=pcc1d_gemv_blkm_prep,
                   solve=pcc1d_solve,n=12,p=3)
 a = randn(n...,p...)
 x = randn(n...)
 â = gemv_prep(a)
 y = gemv(x,â)

 xx = solve(y,â)
 @test relerr(x,xx) ≈ 0 atol=1e-5
end

# tests adjoint
function test_adjoint(gemv=pcc1d_gemv_blkm,
                      gemv_prep=pcc1d_gemv_blkm_prep,
                      adjoint=pcc1d_adjoint,n=12,p=3)
 a = randn(n...,p...)
 â = gemv_prep(a)
 ât = adjoint(â)
 
 x = randn(n...)
 y = randn(n...)
 z1 = vec(y)'*vec(gemv(x,â))
 z2 = vec(gemv(y,ât))'*vec(x)
 @test relerr(z1,z2) ≈ 0 atol=1e-10
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

 # tests matrix matrix
 @testset "gemm" begin
 test_gemm(pcc1d_gemv_blkm,pcc1d_gemv_blkm_prep,pcc1d_gemm,12,1)
 test_gemm(pcc1d_gemv_blkm,pcc1d_gemv_blkm_prep,pcc1d_gemm,12,4)
 test_gemm(pcc1d_gemv_blkm,pcc1d_gemv_blkm_prep,pcc1d_gemm,12,12)
 test_gemm(pcc1d_gemv_blkm,pcc1d_gemv_blkm_prep,pcc1d_gemm,15,5)
 test_gemm(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_gemm,(27,32),(1,1))
 test_gemm(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_gemm,(32,27),(1,1))
 test_gemm(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_gemm,(27,32),(3,4))
 test_gemm(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_gemm,(32,27),(4,3))
 test_gemm(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_gemm,(27,32),(27,32))
 test_gemm(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_gemm,(32,27),(32,27))
 end

 # tests adjoint
 @testset "adjoint" begin
 test_adjoint(pcc1d_gemv_blkm,pcc1d_gemv_blkm_prep,pcc1d_adjoint,12,1)
 test_adjoint(pcc1d_gemv_blkm,pcc1d_gemv_blkm_prep,pcc1d_adjoint,12,4)
 test_adjoint(pcc1d_gemv_blkm,pcc1d_gemv_blkm_prep,pcc1d_adjoint,12,12)
 test_adjoint(pcc1d_gemv_blkm,pcc1d_gemv_blkm_prep,pcc1d_adjoint,15,5)
 test_adjoint(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_adjoint,(27,32),(1,1))
 test_adjoint(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_adjoint,(32,27),(1,1))
 test_adjoint(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_adjoint,(27,32),(3,4))
 test_adjoint(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_adjoint,(32,27),(4,3))
 test_adjoint(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_adjoint,(32,32),(32,32))
 test_adjoint(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_adjoint,(32,27),(32,27))
 end
 
 # tests solve
 @testset "solve" begin
 test_solve(pcc1d_gemv_blkm,pcc1d_gemv_blkm_prep,pcc1d_solve,12,1)
 test_solve(pcc1d_gemv_blkm,pcc1d_gemv_blkm_prep,pcc1d_solve,12,3)
 test_solve(pcc1d_gemv_blkm,pcc1d_gemv_blkm_prep,pcc1d_solve,12,12) # p = n
 test_solve(pcc1d_gemv_blkm,pcc1d_gemv_blkm_prep,pcc1d_solve,100,5)
 test_solve(pcc1d_gemv_blkm,pcc1d_gemv_blkm_prep,pcc1d_solve,64,8)
 test_solve(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_solve,(27,32),(1,1))
 test_solve(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_solve,(32,27),(1,1))
 test_solve(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_solve,(27,32),(3,4))
 test_solve(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_solve,(32,27),(4,3))
 test_solve(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_solve,(32,32),(32,32)) # p =n
 test_solve(pcc2d_gemv_blkm,pcc2d_gemv_blkm_prep,pcc2d_solve,(32,27),(32,27)) # p =n
 end
end

test_pcc_all()
