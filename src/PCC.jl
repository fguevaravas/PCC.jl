module PCC
 using LinearAlgebra, FFTW
 using BenchmarkTools

 export cc1d, pcc1d_naive_prep, pcc1d_naive, 
  pcc1d_fft_prep, pcc1d_fft, 
  pcc1d_fft_blk_v0_prep, pcc1d_fft_blk_v0,
  pcc1d_fft_blk_v1_prep, pcc1d_fft_blk_v1,
  pcc1d_fft_blk_v2_prep, pcc1d_fft_blk_v2

 export  pcc2d_naive_prep, pcc2d_naive, 
  pcc2d_fft_prep, pcc2d_fft, 
  pcc2d_fft_blk_v0_prep, pcc2d_fft_blk_v0,
  pcc2d_fft_blk_v1_prep, pcc2d_fft_blk_v1,
  pcc2d_fft_blk_v2_prep, pcc2d_fft_blk_v2

 ⊗ = kron


 function cc1d(x,k)
  n = length(x)
  y1=ifft(fft(x).*fft(k))
  y2=zeros(n)
  for i=1:n,j=1:n
      y2[i] += x[j] * k[mod1(i-j+1,n)]
  end
  return y1,y2
 end

 """
  Naive implementation of periodic circular convolution
 """
 function pcc1d_naive(x,k)
  n, = size(x); n_,p = size(k)
  @assert (n%p == 0) && (n==n_)
  T = promote_type(eltype(x),eltype(k))
  y = zeros(T,n)
  for i=1:n,j=1:n
      y[i] += x[j] * k[mod1(i-j+1,n),mod1(j,p)]
  end
  return y
 end
 pcc1d_naive_prep(k) = k

 function pcc1d_fft_prep(k)
  n,p = size(k); @assert n%p == 0
  return fft(k,1)
 end

 function pcc1d_fft(x,k̂)
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


 function pcc1d_fft_blk_v0_prep(k)
  n,p = size(k); @assert n%p == 0
  k̂ = zeros(complex(eltype(k)),size(k))
  for i=1:p
   k̂[:,i] = fft(circshift(k[:,i],i-1))
  end
  return k̂
 end
 
 function pcc1d_fft_blk_v0(x,k̂)
  n,p = size(k̂); n_, = size(x); q = n ÷ p
  @assert (n%p == 0) && (n==n_)
  T = promote_type(eltype(x),eltype(k̂))
  y = zeros(T,n)
  x̂ = fft(reshape(x,p,q),2) 
  for j=1:p
   y += k̂[:,j] .* (ones(p,1) ⊗ x̂[j:p:n])
  end
  return ifft!(y)
 end
 
 # multi-index version
 pcc1d_fft_blk_v1_prep = pcc1d_fft_blk_v0_prep
 function pcc1d_fft_blk_v1(x,k̂)
  n,p = size(k̂); n_, = size(x); q = n ÷ p
  @assert (n%p == 0) && (n==n_)
  T = promote_type(eltype(x),eltype(k̂))
  y = zeros(T,q,p)
  x̂ = fft(reshape(x,p,q),2) 
  k̂ = reshape(k̂,q,p,p)
  for j=1:p, i=1:p, l=1:q
   y[l,i] += k̂[l,i,j] * x̂[j,l]
  end
  return ifft!(y[:])
 end

 # MV version
 function pcc1d_fft_blk_v2_prep(k)
  n,p = size(k); @assert n%p == 0; q = n ÷ p
  k̂ = zeros(complex(eltype(k)),size(k))
  for i=1:p
   k̂[:,i] = fft(circshift(k[:,i],i-1))
  end
  k̂ = reshape(k̂,q,p,p)
  return permutedims(k̂,(2,3,1))
 end

 function pcc1d_fft_blk_v2(x,k̂)
  p,p_,q = size(k̂); n, = size(x); 
  @assert (n == p*q) && (p == p_)
  T = promote_type(eltype(x),eltype(k̂))
  y = zeros(T,q,p)
  x̂ = fft(reshape(x,p,q),2) 
  for l=1:q
   y[l,:] += k̂[:,:,l] * x̂[:,l]
  end
  return ifft!(y[:])
 end

 
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

 function pcc2d_naive(x,k)
  n1,n2 = size(x); n1_,n2_,p1,p2 = size(k)
  @assert (n1%p1==0) && (n2%p2==0) 
  @assert (n1==n1_) && (n2==n2_)
  T = promote_type(eltype(x),eltype(k))
  y = zeros(T,n1,n2)
  for i1=1:n1,j1=1:n1,i2=1:n2,j2=1:n2
      y[i1,i2] += x[j1,j2] * k[mod1(i1-j1+1,n1),mod1(i2-j2+1,n2),mod1(j1,p1),mod1(j2,p2)]
  end
  return y
 end
 pcc2d_naive_prep(k) = k

 function pcc2d_fft_prep(k)
  n1,n2,p1,p2 = size(k) 
  @assert n1%p1 == 0 && n2%p2 == 0
  return fft(k,(1,2))
 end

 function pcc2d_fft(x,k̂)
  n1,n2 = size(x); n1_,n2_,p1,p2 = size(k̂)
  @assert (n1%p1==0) && (n2%p2==0) 
  @assert (n1==n1_) && (n2==n2_)
  T = promote_type(eltype(x),eltype(k̂))
  y = zeros(T,n1,n2)
  xi = zeros(T,n1,n2)
  for i1=1:p1, i2=1:p2
   xi .=0; xi[i1:p1:n1,i2:p2:n2]=x[i1:p1:n1,i2:p2:n2]
   fft!(xi)
   y += k̂[:,:,i1,i2] .* xi
  end
  return ifft!(y)
 end 

 function pcc2d_fft_blk_v0_prep(k)
  n1,n2,p1,p2 = size(k); @assert n1%p1==0 && n2%p2==0
  k̂ = zeros(complex(eltype(k)),size(k))
  for i1=1:p1,i2=1:p2
   k̂[:,:,i1,i2] = fft(circshift(k[:,:,i1,i2],(i1-1,i2-1)))
  end
  return k̂
 end
 
 function pcc2d_fft_blk_v0(x,k̂)
  n1,n2,p1,p2 = size(k̂); n1_,n2_ = size(x)
  q1 = n1 ÷ p1; q2 = n2 ÷ p2
  @assert (n1%p1==0) && (n2%p2==0) 
  @assert (n1==n1_) && (n2==n2_)
  T = promote_type(eltype(x),eltype(k̂))
  y = zeros(T,n1,n2)
  x̂ = reshape(fft(reshape(x,p1,q1,p2,q2),(2,4)),n1,n2)
  for j1=1:p1,j2=1:p2
   y += k̂[:,:,j1,j2] .* (ones(p1,p2) ⊗ x̂[j1:p1:n1,j2:p2:n2])
  end
  return ifft!(y)
 end

 # multi-index version 
 pcc2d_fft_blk_v1_prep = pcc2d_fft_blk_v0_prep
 function pcc2d_fft_blk_v1(x,k̂)
  n1,n2,p1,p2 = size(k̂); n1_,n2_ = size(x)
  q1 = n1 ÷ p1; q2 = n2 ÷ p2
  @assert (n1%p1==0) && (n2%p2==0) 
  @assert (n1==n1_) && (n2==n2_)
  T = promote_type(eltype(x),eltype(k̂))
  y = zeros(T,q1,p1,q2,p2)
  x̂ = fft(reshape(x,p1,q1,p2,q2),(2,4))
  k̂ = reshape(k̂,q1,p1,q2,p2,p1,p2)
  for j1=1:p1, j2=1:p2, i1=1:p1, i2=1:p2, l1=1:q1, l2=1:q2
   y[l1,i1,l2,i2] += k̂[l1,i1,l2,i2,j1,j2] * x̂[j1,l1,j2,l2]
  end
  return ifft!(reshape(y,n1,n2))
 end

  # MV version, all dimensions are reordered to make the
  # p1p2 x p1p2 matrix vector product contiguous in memory
 function pcc2d_fft_blk_v2_prep(k)
  n1,n2,p1,p2 = size(k); @assert n1%p1==0 && n2%p2==0
  q1 = n1 ÷ p1; q2 = n2 ÷ p2
  k̂ = zeros(complex(eltype(k)),size(k))
  for i1=1:p1,i2=1:p2
   k̂[:,:,i1,i2] = fft(circshift(k[:,:,i1,i2],(i1-1,i2-1)))
  end
  k̂ = reshape(k̂,q1,p1,q2,p2,p1,p2)
  k̂ = permutedims(k̂,(2,4,5,6,1,3))

 end

 function pcc2d_fft_blk_v2(x,k̂)
  p1,p2,p1_,p2_,q1,q2 = size(k̂); n1,n2 = size(x)
  @assert (n1==p1*q1) && (n2==p2*q2) 
  @assert (p1==p1_) && (p2==p2_)
  T = promote_type(eltype(x),eltype(k̂))
  y = zeros(T,p1*p2,q1,q2)
  x̂ = fft(reshape(x,p1,q1,p2,q2),(2,4))
  x̂ = permutedims(x̂,(1,3,2,4))
  for l1=1:q1, l2=1:q2
   D = reshape(k̂[:,:,:,:,l1,l2],p1*p2,p1*p2)
   y[:,l1,l2] += D * vec(x̂[:,:,l1,l2])
  end
  y = reshape(y,p1,p2,q1,q2)
  y = permutedims(y,(3,1,4,2))
  y = reshape(y,n1,n2)
  return ifft!(y)
 end
 
end
