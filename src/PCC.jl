module PCC
 # TODO: add @inbounds to inner loops before benchmarks
 using LinearAlgebra, FFTW

 export pcc1d_naive_prep, pcc1d_naive, 
  pcc1d_cc_prep, pcc1d_cc, 
  pcc1d_improved_prep, pcc1d_improved, 
  pcc2d_naive_prep, pcc2d_naive, 
  pcc2d_cc_prep, pcc2d_cc, 
  pcc2d_improved_prep, pcc2d_improved

 """
  Naive implementation of 1D periodic circular convolution
  Inputs:
   x (size n) signal 
   k (size n x p) first p columns of kernel matrix
 """
 function pcc1d_naive(x,k)
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
 pcc1d_naive_prep(k) = k

 """
  PCC 1D in terms of n circular convolutions. Preparation phase.
  Inputs:
   k (size n x p) first p columns of kernel matrix

   Note: n % p == 0 is required
   
  Outputs:
   k̂ (size n x p, complex float) precomputed DFTs of kernel
 """
 function pcc1d_cc_prep(k)
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
   k̂ (size n x p) output of pcc1d_cc_prep

   Note: n % p == 0 is required

  Outputs:
   y (size n) result of PCC K *ₚ x
 """ 
 function pcc1d_cc(x,k̂)
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
 function pcc1d_improved_prep(k)
  n,p = size(k)
  @assert (n%p == 0)
  q = n÷p
  return fft(reshape(k,p,q,p),2)
 end

 """
  Improved PCC 1D implementation. Application phase.
  Inputs:
   x (size n) signal 
   k̂ (size p x q x p) output of pcc1d_improved_prep

   Note: n == p*q is required

  Outputs:
   y (size n) result of PCC K *ₚ x
 """
 function pcc1d_improved(x,k̂)
  n, = size(x); p,q,p_ = size(k̂)
  @assert (n%p == 0) && (p==p_) && (p*q==n)
  x̂ = fft(reshape(x,p,q),2)
  y  = zeros(eltype(x̂),p,q)
  for l=1:p,j=1:q,i=1:p
   y[i,j] += k̂[i,j,l]*x̂[l,j]
  end
  ifft!(y,2)
  return reshape(y,n)
 end

 """
  Naive implementation of 2D periodic circular convolution
  Inputs:
   x (size n1 x n2) signal 
   k (size n1 x n2 x p1 x p2) k = K[:,:,1:p1,1:p2] for kernel K

   Note: n1 % p1 == 0 and n2 % p2 == 0 are required
 """
 function pcc2d_naive(x,k)
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
 pcc2d_naive_prep(k) = k

 """
  Improved PCC 2D implementation. Preparation phase.
  Inputs:
   k (size n1 x n2 x p1 x p2) k = K[:,:,1:p1,1:p2] for kernel K

   Note: n1 % p1 == 0 and n2 % p2 == 0 are required. Also
   n1 == p1*q1 and n2 == p2*q2

  Outputs:
   k̂ (size p1 x q1 x p2 x q2 x p1 x p2, complex float) precomputed DFTs of kernel
 """
 function pcc2d_improved_prep(k)
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
 function pcc2d_improved(x,k̂)
  n1,n2 = size(x); p1,q1,p2,q2,p1_,p2_ = size(k̂)
  @assert (n1==p1*q1) && (p1==p1_)
  @assert (n2==p2*q2) && (p2==p2_)
  x̂ = fft(reshape(x,p1,q1,p2,q2),(2,4))
  y = zeros(eltype(x̂),p1,q1,p2,q2)
  for l2=1:p2,l1=1:p1,j2=1:q2,i2=1:p2,j1=1:q1,i1=1:p1
   y[i1,j1,i2,j2] += k̂[i1,j1,i2,j2,l1,l2]*x̂[l1,j1,l2,j2]
  end
  ifft!(y,(2,4))
  return reshape(y,n1,n2)
 end
 """
  PCC 2D implementation with n circular convolutions. Preparation phase.
  Inputs:
   k (size n1 x n2 x p1 x p2) k = K[:,:,1:p1,1:p2] for kernel K

   Note: n1 % p1 == 0 and n2 % p2 == 0 are required

  Outputs:
   k̂ (size n1 x n2 x p1 x p2, complex float) precomputed DFTs of kernel
 """
 function pcc2d_cc_prep(k)
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
   k̂ (size n1 x n2 x p1 x p2) output of pcc2d_cc_prep

   Note: n1 % p1 == 0 and n2 % p2 == 0 are required

  Outputs:
   y (size n1 x n2) result of PCC K *ₚ x
 """
 function pcc2d_cc(x,k̂)
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
 
end
