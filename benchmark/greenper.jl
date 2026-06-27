using PCC, LinearAlgebra, PyPlot
# a script to calculate a periodic Green function

# 1D case examples
# n = 6, p = 1
# . -1- . -1- . -1- . -1- . -1- . -1- 
# n = 6, p = 2
# . -1- . -2- . -1- . -2- . -1- . -2- 
# n = 6, p = 3
# . -1- . -2- . -3- . -1- . -2- . -3- 
#
# convention is that σ[j] is conductivity between nodes j and j+1
function periodic_laplacian_1d(n,p,σ)
 @assert (n % p == 0) && (n>2)
 @assert (length(σ) == p)
 q = n ÷ p
 L = zeros(n,p)
 for j=1:p
  L[mod1(j+1,n),j] = -σ[j]
  L[mod1(j-1,n),j] = -σ[mod1(j-1,p)]
  L[j,j] += σ[j] + σ[mod1(j-1,p)]
 end
 return L
end


# convention:
#   - σ1 are edges in direction 1 so of size p1 x p2
#   - σ2 are edges in direction 2 so of size p1 x p2
function periodic_laplacian_2d(n1,n2,p1,p2,σ1,σ2)
 @assert (n1 % p1 == 0) && (n2 % p2 == 0)
 @assert (n1>2) && (n2>2)
 @assert all(size(σ1) .== [p1,p2]) && all(size(σ2) .== [p1,p2])
 q1 = n1 ÷ p1; q2 = n2 ÷ p2
 L = zeros(n1,n2,p1,p2)
 for j1=1:p1,j2=1:p2
  L[mod1(j1+1,n1),j2,j1,j2] = -σ1[j1,j2]
  L[mod1(j1-1,n1),j2,j1,j2] = -σ1[mod1(j1-1,p1),j2]
  L[j1,mod1(j2+1,n2),j1,j2] = -σ2[j1,j2]
  L[j1,mod1(j2-1,n2),j1,j2] = -σ2[j1,mod1(j2-1,p2)]
  L[j1,j2,j1,j2] += σ1[j1,j2] + σ1[mod1(j1-1,p1),j2] + σ2[j1,j2] + σ2[j1,mod1(j2-1,p2)]
 end
 return L
end

# we test by calculating this in a different way
function test_periodic_laplacian_1d(n=20,p=4)
 x = randn(n)
 σ = rand(p)
 L = periodic_laplacian_1d(n,p,σ)
 L̂ = pcc1d_gemv_blkm_prep(L)
 Lx = zeros(n)
 for i=1:n
  Lx[i] += σ[mod1(i,p)]  *x[i]   - σ[mod1(i,p)]*x[mod1(i+1,n)]
  Lx[i] += σ[mod1(i-1,p)]*x[i] - σ[mod1(i-1,p)]*x[mod1(i-1,n)]
 end
 println("should be small = ",norm(Lx - pcc1d_gemv_blkm(x,L̂)))
end

# we test by calculating this in a different way
function test_periodic_laplacian_2d(n=(20,20),p=(4,5))
 x = randn(n...)
 σ1 = rand(p...)
 σ2 = rand(p...)
 L = periodic_laplacian_2d(n...,p...,σ1,σ2)
 L̂ = pcc2d_gemv_blkm_prep(L)
 Lx = zeros(n...)
 n1,n2=n; p1,p2=p
 for i1=1:n1, i2=1:n2
  Lx[i1,i2] += σ1[mod1(i1,p1),mod1(i2,p2)]  *x[i1,i2] - σ1[mod1(i1,p1),mod1(i2,p2)]  *x[mod1(i1+1,n1),i2]
  Lx[i1,i2] += σ1[mod1(i1-1,p1),mod1(i2,p2)]*x[i1,i2] - σ1[mod1(i1-1,p1),mod1(i2,p2)]*x[mod1(i1-1,n1),i2]
  Lx[i1,i2] += σ2[mod1(i1,p1),mod1(i2,p2)]  *x[i1,i2] - σ2[mod1(i1,p1),mod1(i2,p2)]  *x[i1,mod1(i2+1,n2)]
  Lx[i1,i2] += σ2[mod1(i1,p1),mod1(i2-1,p2)]*x[i1,i2] - σ2[mod1(i1,p1),mod1(i2-1,p2)]*x[i1,mod1(i2-1,n2)]
 end
 println("should be small = ",norm(Lx - pcc2d_gemv_blkm(x,L̂)))
end

function greenper1d(n=20,p=4,k=1+1e-2im)
 σ = 1:4
 L = periodic_laplacian_1d(n,p,σ)
 L̂ = pcc1d_gemv_blkm_prep(L)
 L̂ -= k*2*pcc1d_eye(n,p)
 G = zeros(n,p)
 for j=1:p
  G[:,j] = real(pcc1d_solve([i==j for i=1:n],L̂))
 end
 figure(1); clf()
 plot(G)
end

function greenper2d(n=(81,81),p=(3,3),k=0.1+1e-2im)
 #σ1 = reshape(1:prod(p),p...)
 #σ2 = reshape(1:prod(p),p...)
 σ1 = rand(p...)
 σ2 = rand(p...)
 L = periodic_laplacian_2d(n...,p...,σ1,σ2)
 L̂ = pcc2d_gemv_blkm_prep(L)
 L̂ -= k*2*pcc2d_eye(n...,p...)
 G = zeros(n...,p...)
 for j1=1:p[1],j2=1:p[2]
  G[:,:,j1,j2] = real(pcc2d_solve([(i1==j1)*(i2==j2) for i1=1:n[1],i2=1:n[2]],L̂))
 end
 figure(1); clf()
 for j1=1:p[1],j2=1:p[2]
  subplot(p...,j2 + (j1-1)*p[2]);
  imshow(circshift(G[:,:,j1,j2],n.÷2));
  axis("equal"); axis("off")
 end
end
