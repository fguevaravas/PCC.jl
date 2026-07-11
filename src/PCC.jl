module PCC
 using LinearAlgebra, FFTW

 export pcc1d_gemv_naive_prep, pcc1d_gemv_naive, 
  pcc1d_gemv_cc_prep, pcc1d_gemv_cc, 
  pcc1d_gemv_improved_prep, pcc1d_gemv_improved, 
  pcc1d_gemv_blkm_prep, pcc1d_gemv_blkm, 
  pcc1d_cond, pcc1d_svdvals, pcc1d_eye, pcc1d_eigvals,
  pcc2d_gemv_naive_prep, pcc2d_gemv_naive, 
  pcc2d_gemv_cc_prep, pcc2d_gemv_cc, 
  pcc2d_gemv_improved_prep, pcc2d_gemv_improved,
  pcc2d_gemv_blkm_prep, pcc2d_gemv_blkm, 
  pcc2d_cond, pcc2d_svdvals, pcc2d_eye, pcc2d_eigvals

 export pcc1d_gemm, pcc2d_gemm, pcc1d_adjoint, pcc2d_adjoint,
  pcc1d_solve, pcc2d_solve

 """
  Naive implementation of 1D periodic circular convolution
  Inputs:
   x (size n) signal 
   k (size n x p) first p columns of kernel matrix
 """
 function pcc1d_gemv_naive(x,k)
  n, = size(x); n_,p = size(k)
  @assert (n%p == 0) && (n==n_)
  T = promote_type(eltype(x),eltype(k))
  y = zeros(T,n)
  for i=1:n,j=1:n
   jmp = mod1(j,p)
   y[i] += x[j] * k[mod1(i-j+jmp,n),jmp]
  end
  return y
 end

 """
 preparation phase for naive implementation
 (not needed)
 """
 pcc1d_gemv_naive_prep(k) = k

 """
  PCC 1D in terms of n circular convolutions. Preparation phase.
  Inputs:
   k (size n x p) first p columns of kernel matrix

   Note: n % p == 0 is required
   
  Outputs:
   k̂ (size n x p, complex float) precomputed DFTs of kernel
 """
 function pcc1d_gemv_cc_prep(k)
  n,p = size(k); @assert n%p == 0
  k̂ = zeros(complex(float(eltype(k))),n,p)
  for i=1:p
   k̂[:,i] = circshift(k[:,i],-i+1)
  end
  fft!(k̂,1) # FFT in place
  return k̂
 end

 """
  PCC 1D in terms of n circular convolutions. Application phase.
  Inputs:
   x (size n) signal 
   k̂ (size n x p) output of pcc1d_gemv_cc_prep

   Note: n % p == 0 is required

  Outputs:
   y (size n) result of PCC K *ₚ x
 """ 
 function pcc1d_gemv_cc(x,k̂)
  n,p = size(k̂); n_, = size(x)
  @assert (n%p == 0) && (n==n_)
  T = promote_type(eltype(x),eltype(k̂))
  y = zeros(T,n)
  xi = zeros(T,n)
  for i=1:p
   xi .=0; xi[i:p:n]=x[i:p:n]
   fft!(xi)
   y += k̂[:,i] .* xi
  end
  return ifft!(y)
 end 

 """
  Improved PCC 1D implementation. Preparation phase.
  Inputs:
   k (size n x p) first p columns of kernel matrix

   Note: n % p == 0 is required

  Outputs:
   k̂ (size p x q x p, complex float) precomputed DFTs of kernel
 """
 function pcc1d_gemv_improved_prep(k)
  n,p = size(k)
  @assert (n%p == 0)
  q = n÷p
  return fft(reshape(k,p,q,p),2)
 end

 """
  Improved PCC 1D implementation. Application phase.
  Inputs:
   x (size n) signal 
   k̂ (size p x q x p) output of pcc1d_gemv_improved_prep

   Note: n == p*q is required

  Outputs:
   y (size n) result of PCC K *ₚ x
 """
 function pcc1d_gemv_improved(x,k̂)
  n, = size(x); p,q,p_ = size(k̂)
  @assert (n%p == 0) && (p==p_) && (p*q==n)
  x̂ = fft(reshape(x,p,q),2)
  y  = zeros(eltype(x̂),p,q)
  for l=1:p,j=1:q,i=1:p
   @inbounds y[i,j] += k̂[i,j,l]*x̂[l,j]
  end
  ifft!(y,2)
  return reshape(y,n)
 end

 """
  PCC 1D implementation as block matrix. Preparation phase.
  Inputs:
   k (size n x p) first p columns of kernel matrix

   Note: n % p == 0 is required

  Outputs:
   k̂ (size p x q x p, complex float) precomputed DFTs of kernel
 """
 pcc1d_gemv_blkm_prep(k) = permutedims(pcc1d_gemv_improved_prep(k),(1,3,2))

 """
  Improved PCC 1D implementation as block matrix. Application phase.
  Inputs:
   x (size n) signal 
   k̂ (size p x p x q) output of pcc1d_gemv_blkm_prep

   Note: n == p*q is required

  Outputs:
   y (size n) result of PCC K *ₚ x
 """
 function pcc1d_gemv_blkm(x,k̂)
  n, = size(x); p,p_,q = size(k̂)
  @assert (n%p == 0) && (p==p_) && (p*q==n)
  x̂ = fft(reshape(x,p,q),2)
  y  = zeros(eltype(x̂),p,q)
  for j=1:q
   @inbounds y[:,j] .= k̂[:,:,j]*x̂[:,j]
  end
  ifft!(y,2)
  return reshape(y,n)
 end

 
 """
  multiplication of two PCCs with kernel matrices A and B
  Inputs:
   â (size p x p x q) blk matrix representation for A
   b̂ (size p x p x q) blk matrix representation for B

  Note: the tensor representations are obtained with pcc1d_gemv_blkm_prep

  Outputs:
   ĉ (size p x q x p) the prepared kernel to apply the operation A*B
 """
 function pcc1d_gemm(â,b̂)
  p,p_,q = size(â)
  @assert (p==p_) && all([p,p_,q] .== size(b̂))
  ĉ = zeros(promote_type(eltype(â),eltype(b̂)),p,p,q)
  for j=1:q
   @inbounds ĉ[:,:,j] = reshape(â[:,:,j],p,p) * reshape(b̂[:,:,j],p,p)
  end
  return ĉ
 end

 """
  adjoint of a matrix kernel A via its blk matrix representation
  Inputs: 
   â  (size p x p x q) blk matrix representation for kernel matrix A
  Outputs:
   ât (size p x p x q) blk matrix representation for kernel matrix A'
 """
 function pcc1d_adjoint(â)
  p,p_,q = size(â)
  @assert (p==p_)
  ât = permutedims(â,(2,1,3))
  conj!(ât)
  return ât
 end

 """
  solves A x = y  via blk matrix representation of A
  Inputs: 
   y  (size n) input signal
   â  (size p x p x q) blk matrix representation for kernel matrix A
  Outputs:
   x (size n) output signal
 """
 function pcc1d_solve(y,â)
  n, = size(y); p,p_,q = size(â)
  @assert (n%p == 0) && (p==p_) && (p*q==n)
  ŷ = fft(reshape(y,p,q),2)
  x  = zeros(eltype(ŷ),p,q)
  for j=1:q
   @inbounds x[:,j] .= â[:,:,j]\ŷ[:,j]
  end
  ifft!(x,2)
  return reshape(x,n)
 end

 """
  computes the singular values via blk matrix representation of A
  Inputs: 
   â  (size p x p x q) blk matrix representation for kernel matrix A
  Outputs:
   s (size  p x q) singular values of each each of the blocks
 """
 function pcc1d_svdvals(â)
  p,p_,q = size(â)
  @assert (p==p_)
  s = zeros(p,q)
  for j=1:q
   @inbounds s[:,j] .= svdvals(â[:,:,j])
  end
  return s
 end

 """
  computes the eigenvalues via blk matrix representation of A
  Inputs: 
   â  (size p x p x q) blk matrix representation for kernel matrix A
  Outputs:
   s (size  p x q) eigen values of each each of the blocks

   pcc1d_eigvals(â,b̂) does the same for the matrix pencil A x = λ B x

 """
 function pcc1d_eigvals(â)
  p,p_,q = size(â)
  @assert (p==p_)
  s = zeros(ComplexF64,p,q)
  for j=1:q
   @inbounds s[:,j] .= eigvals(â[:,:,j])
  end
  return s
 end

 function pcc1d_eigvals(â,b̂)
  p,p_,q = size(â)
  @assert all(size(â) .== size(b̂) )
  @assert (p==p_)
  s = zeros(ComplexF64,p,q)
  for j=1:q
   @inbounds s[:,j] .= eigvals(â[:,:,j],b̂[:,:,j])
  end
  return s
 end

 """
  computes the (2-norm) condition number via blk matrix representation of A
  Inputs: 
   â  (size p x p x q) blk matrix representation for kernel matrix A
  Outputs:
   cond(A)
 """
 function pcc1d_cond(â)
  s = pcc1d_svdvals(â)
  return maximum(s)/minimum(s)
 end

 """ 
 the identity compatible with blk matrix representation
 Inputs:
  n,p (Integers) to specify the kernel size and period
 Outputs:
  Id (p x p x q) block diagonal representation of identity
 """
 function pcc1d_eye(n,p)
  @assert (n%p == 0)
  q = n ÷ p
  Id = [ float(i==j) for i=1:p, j=1:p, k=1:q ]
  return Id
 end

 """
  Naive implementation of 2D periodic circular convolution
  Inputs:
   x (size n1 x n2) signal 
   k (size n1 x n2 x p1 x p2) k = K[:,:,1:p1,1:p2] for kernel K

   Note: n1 % p1 == 0 and n2 % p2 == 0 are required
 """
 function pcc2d_gemv_naive(x,k)
  n1,n2 = size(x); n1_,n2_,p1,p2 = size(k)
  @assert (n1%p1==0) && (n2%p2==0) 
  @assert (n1==n1_) && (n2==n2_)
  T = promote_type(eltype(x),eltype(k))
  y = zeros(T,n1,n2)
  for i1=1:n1,j1=1:n1,i2=1:n2,j2=1:n2
      jmp1 = mod1(j1,p1)
      jmp2 = mod1(j2,p2)
      y[i1,i2] += x[j1,j2] * k[mod1(i1-j1+jmp1,n1),mod1(i2-j2+jmp2,n2),jmp1,jmp2]
  end
  return y
 end

 """
 preparation phase for naive implementation
 (not needed)
 """
 pcc2d_gemv_naive_prep(k) = k

 """
  PCC 2D implementation with n circular convolutions. Preparation phase.
  Inputs:
   k (size n1 x n2 x p1 x p2) k = K[:,:,1:p1,1:p2] for kernel K

   Note: n1 % p1 == 0 and n2 % p2 == 0 are required

  Outputs:
   k̂ (size n1 x n2 x p1 x p2, complex float) precomputed DFTs of kernel
 """
 function pcc2d_gemv_cc_prep(k)
  n1,n2,p1,p2 = size(k) 
  @assert n1%p1 == 0 && n2%p2 == 0
  k̂ = zeros(complex(float(eltype(k))),n1,n2,p1,p2)
  for i1=1:p1,i2=1:p2
   k̂[:,:,i1,i2] = circshift(k[:,:,i1,i2],(-i1+1,-i2+1))
  end
  fft!(k̂,(1,2)) # FFT in place
  return k̂
 end

 """
  PCC 2D implementation with n circular convolutions. Application phase

  Inputs:
   x (size n1 x n2) signal
   k̂ (size n1 x n2 x p1 x p2) output of pcc2d_gemv_cc_prep

   Note: n1 % p1 == 0 and n2 % p2 == 0 are required

  Outputs:
   y (size n1 x n2) result of PCC K *ₚ x
 """
 function pcc2d_gemv_cc(x,k̂)
  n1,n2 = size(x); n1_,n2_,p1,p2 = size(k̂)
  @assert (n1%p1==0) && (n2%p2==0) 
  @assert (n1==n1_) && (n2==n2_)
  T = promote_type(eltype(x),eltype(k̂))
  y = zeros(T,n1,n2)
  xi = zeros(T,n1,n2)
  for i1=1:p1, i2=1:p2
   xi .=0; xi[i1:p1:n1,i2:p2:n2]=x[i1:p1:n1,i2:p2:n2]
   fft!(xi) # FFT in place
   y += k̂[:,:,i1,i2] .* xi
  end
  return ifft!(y)
 end 

 """
  Improved PCC 2D implementation. Preparation phase.
  Inputs:
   k (size n1 x n2 x p1 x p2) k = K[:,:,1:p1,1:p2] for kernel K

   Note: n1 % p1 == 0 and n2 % p2 == 0 are required. Also
   n1 == p1*q1 and n2 == p2*q2

  Outputs:
   k̂ (size p1 x q1 x p2 x q2 x p1 x p2, complex float) precomputed DFTs of kernel
 """
 function pcc2d_gemv_improved_prep(k)
  n1,n2,p1,p2 = size(k)
  @assert (n1%p1==0) && (n2%p2==0)
  q1 = n1÷p1; q2 = n2÷p2;
  return fft(reshape(k,p1,q1,p2,q2,p1,p2),(2,4))
 end

 """
  Improved PCC 2D implementation. Application phase.
  Inputs:
   x (size n1 x n2) signal 
   k̂ (size p1 x q1 x p2 x q2 x p1 x p2) output of pcc2d_improved_prep

   Note: n1 == p1*q1 and n2 == p2*q2 are required

  Outputs:
   y (size n1 x n2) result of PCC K *ₚ x
  """
 function pcc2d_gemv_improved(x,k̂)
  n1,n2 = size(x); p1,q1,p2,q2,p1_,p2_ = size(k̂)
  @assert (n1==p1*q1) && (p1==p1_)
  @assert (n2==p2*q2) && (p2==p2_)
  x̂ = fft(reshape(x,p1,q1,p2,q2),(2,4))
  y = zeros(eltype(x̂),p1,q1,p2,q2)
  for l2=1:p2,l1=1:p1,j2=1:q2,i2=1:p2,j1=1:q1,i1=1:p1
   @inbounds y[i1,j1,i2,j2] += k̂[i1,j1,i2,j2,l1,l2]*x̂[l1,j1,l2,j2]
  end
  ifft!(y,(2,4))
  return reshape(y,n1,n2)
 end

  """
  PCC 2D implementation as block matrix. Preparation phase.
  Inputs:
   k (size n1 x n2 x p1 x p2) k = K[:,:,1:p1,1:p2] for kernel K

  Outputs:
   k̂ (size p1 x p2 x p1 x p2 x q1 x q2, complex float) precomputed DFTs of kernel
   ordered in a way to highlight block matrix structure
 """
 pcc2d_gemv_blkm_prep(k) = permutedims(pcc2d_gemv_improved_prep(k),(1,3,5,6,2,4))

 """
  Improved PCC 2D implementation as block matrix. Application phase.
  Inputs:
   x (size n1 x n2) signal 
   k̂ (size p1 x p2 x p1 x p2 x q1 x q2, complex float) output of pcc2d_gemv_blkm_prep

   Note: n == p*q is required

  Outputs:
   y (size n1 x n2) result of PCC K *ₚ x
 """
 function pcc2d_gemv_blkm(x,k̂)
  n1,n2 = size(x); p1,p2,p1_,p2_,q1,q2 = size(k̂)
  @assert (n1==p1*q1) && (p1==p1_)
  @assert (n2==p2*q2) && (p2==p2_)
  x̂ = fft(reshape(x,p1,q1,p2,q2),(2,4))
  x̂ = permutedims(x̂,(1,3,2,4))
  y = zeros(eltype(x̂),p1*p2,q1,q2)
  for j2=1:q2,j1=1:q1
   @inbounds y[:,j1,j2] .= reshape(k̂[:,:,:,:,j1,j2],p1*p2,p1*p2)*vec(x̂[:,:,j1,j2])
  end
  y = reshape(y,p1,p2,q1,q2)
  y = permutedims(y,(1,3,2,4))
  ifft!(y,(2,4))
  return reshape(y,n1,n2)
 end

 """
  multiplication of two PCCs with kernel matrices A and B
  Inputs:
   â (size p1 x p2 x p1 x p2 x q1 x q2) blk matrix representation for kernel A
   b̂ (size p1 x p2 x p1 x p2 x q1 x q2) blk matrix representation for kernel B

  Note: the tensor representations are obtained with pcc1d_gemv_blkm_prep

  Outputs:
   ĉ (size p1 x p2 x p1 x p2 x q1 x q2) the prepared kernel to apply the operation A*B
 """
 function pcc2d_gemm(â,b̂)
  p1,p2,p1_,p2_,q1,q2 = size(â)
  @assert (p1==p1_) && (p2==p2_)
  @assert all([p1,p2,p1,p2,q1,q2] .== size(b̂))
  ĉ = zeros(promote_type(eltype(â),eltype(b̂)),p1*p2,p1*p2,q1,q2)
  for j2=1:q2,j1=1:q1
   @inbounds ĉ[:,:,j1,j2] = reshape(â[:,:,:,:,j1,j2],p1*p2,p1*p2)*reshape(b̂[:,:,:,:,j1,j2],p1*p2,p1*p2)
  end
  return reshape(ĉ,p1,p2,p1,p2,q1,q2)
 end

 """
  adjoint of a matrix kernel A via its tensor representation
  Inputs: 
   â  (size p1 x p2 x p1 x p2 x q1 x q2) blk matrix representation for A
  Outputs:
   ât (size p1 x p2 x p1 x p2 x q1 x q2) blk matrix representation for A'
 """
 function pcc2d_adjoint(â)
  p1,p2,p1_,p2_,q1,q2 = size(â)
  @assert (p1==p1_) && (p2==p2_)
  ât = permutedims(â,(3,4,1,2,5,6))
  conj!(ât)
  return ât
 end

 """
  solves A x = y  via blk matrix representation of A
  Inputs: 
   y  (size n1 x n2) input signal
   â  (size p1 x p2 x p1 x p2 x q1 x q2) blk matrix representation for kernel matrix A
  Outputs:
   x (size n1 x n2) output signal
 """
 function pcc2d_solve(y,â)
  n1,n2 = size(y); p1,p2,p1_,p2_,q1,q2 = size(â)
  @assert (n1==p1*q1) && (p1==p1_)
  @assert (n2==p2*q2) && (p2==p2_)
  ŷ = fft(reshape(y,p1,q1,p2,q2),(2,4))
  ŷ = permutedims(ŷ,(1,3,2,4))
  x = zeros(eltype(ŷ),p1*p2,q1,q2)
  for j2=1:q2,j1=1:q1
   @inbounds x[:,j1,j2] .= reshape(â[:,:,:,:,j1,j2],p1*p2,p1*p2)\vec(ŷ[:,:,j1,j2])
  end
  x = reshape(x,p1,p2,q1,q2)
  x = permutedims(x,(1,3,2,4))
  ifft!(x,(2,4))
  return reshape(x,n1,n2)
 end

 """
  computes the singular values via blk matrix representation of A
  Inputs: 
   â  (size p1 x p2 x p1 x p2 x q1 x q2) blk matrix representation for kernel matrix A
  Outputs:
   s (size p1*p2 x q1 x q2) singular values of each each of the blocks
 """
 function pcc2d_svdvals(â)
  p1,p2,p1_,p2_,q1,q2 = size(â)
  @assert (p1==p1_) && (p2==p2_)
  s = zeros(p1*p2,q1,q2)
  for j2=1:q2,j1=1:q1
   @inbounds s[:,j1,j2] .= svdvals(reshape(â[:,:,:,:,j1,j2],p1*p2,p1*p2))
  end
  return s
 end

  """
  computes the eigenvalues via blk matrix representation of A
  Inputs: 
   â  (size p1 x p2 x p1 x p2 x q1 x q2) blk matrix representation for kernel matrix A
  Outputs:
   s (size p1*p2 x q1 x q2) eigenvalues of each each of the blocks

   pcc2d_eigvals(â,b̂) does the same for the matrix pencil A x = λ B x
 """
 function pcc2d_eigvals(â)
  p1,p2,p1_,p2_,q1,q2 = size(â)
  @assert (p1==p1_) && (p2==p2_)
  s = zeros(ComplexF64,p1*p2,q1,q2)
  for j2=1:q2,j1=1:q1
   @inbounds s[:,j1,j2] .= eigvals(reshape(â[:,:,:,:,j1,j2],p1*p2,p1*p2))
  end
  return s
 end

 function pcc2d_eigvals(â,b̂)
  p1,p2,p1_,p2_,q1,q2 = size(â)
  @assert all( size(â) .== size(b̂) )
  @assert (p1==p1_) && (p2==p2_)
  s = zeros(ComplexF64,p1*p2,q1,q2)
  for j2=1:q2,j1=1:q1
   @inbounds s[:,j1,j2] .= eigvals(reshape(â[:,:,:,:,j1,j2],p1*p2,p1*p2),
                                   reshape(b̂[:,:,:,:,j1,j2],p1*p2,p1*p2))
  end
  return s
 end

  """
  computes the (2-norm) condition number via blk matrix representation of A
  Inputs: 
   â  (size p1 x p2 x p1 x p2 x q1 x q2) blk matrix representation for kernel matrix A
  Outputs:
   cond(A)
 """
 function pcc2d_cond(â)
  s =  pcc2d_svdvals(â)
  return maximum(s)/minimum(s)
 end

 """ 
 the identity compatible with blk matrix representation
 Inputs:
  n1, n2, p1, p2 (Integers) dimension and period
 Outputs:
  Id (p1 x p2 x p1 x p2 x q1 x q1) Identity representation
 """
 function pcc2d_eye(n1,n2,p1,p2)
  @assert (n1%p1 == 0) && (n2%p2 == 0)
  q1 = n1 ÷ p1; q2 = n2 ÷ p2
  Id = [ float((i1==j1) && (i2==j2)) for i1=1:p1,i2=1:p2,j1=1:p1,j2=1:p2,k1=1:q1,k2=1:q2 ]
  return Id
 end


 
end
