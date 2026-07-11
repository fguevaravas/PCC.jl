using PCC, LinearAlgebra, PyPlot, FFTW
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
  L[j1,j2,j1,j2] += σ1[j1,j2] + σ1[mod1(j1-1,p1),j2] 
  L[j1,j2,j1,j2] += σ2[j1,j2] + σ2[j1,mod1(j2-1,p2)]
 end
 return L
end

# A periodic diagonal operator
function periodic_diag_1d(n,p,d)
 @assert (n % p == 0)
 @assert (length(d) == p)
 q = n ÷ p 
 D = zeros(n,p)
 for j=1:p
  D[j,j] = d[j]
 end
 return D
end

# A periodic diagonal operator
function periodic_diag_2d(n1,n2,p1,p2,d)
 @assert (n1 % p1 == 0) && (n2 % p2 == 0)
 @assert all(size(d) .== [p1,p2]) 
 q1 = n1 ÷ p1; q2 = n2 ÷ p2
 D = zeros(n1,n2,p1,p2)
 for j1=1:p1,j2=1:p2
  D[j1,j2,j1,j2] = d[j1,j2]
 end
 return D
end

# we test by calculating Laplacian in a different way
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

# we test by calculating Laplacian in a different way
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

function test_laplacian()
 test_periodic_laplacian_1d(20,1)
 test_periodic_laplacian_1d(20,4)
 test_periodic_laplacian_1d(81,3)
 test_periodic_laplacian_1d(20,20)

 test_periodic_laplacian_2d((20,20),(1,1))
 test_periodic_laplacian_2d((20,20),(5,4))
 test_periodic_laplacian_2d((20,20),(4,5))
 test_periodic_laplacian_2d((20,20),(20,20))
 test_periodic_laplacian_2d((20,81),(4,3))
 test_periodic_laplacian_2d((81,20),(3,4))

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

# note the explicit formulas are only valid for p=2
function dispersion1d(n=30,p=2)
 σ = ones(p) 
 m = 1:p
 #m = ones(p)
 L = periodic_laplacian_1d(n,p,σ)
 M = periodic_diag_1d(n,p,m)
 L̂ = pcc1d_gemv_blkm_prep(L)
 M̂ = pcc1d_gemv_blkm_prep(M)
 λ = pcc1d_eigvals(L̂,M̂)
 _,q = size(λ)
 ks = 2π.*(0:q-1)./q


 #ks = fftshift(ks); λ = fftshift(λ,2)



 # explicit formula for dispersion relation from
 #
 # https://solidstate.quantumtinkerer.tudelft.nl/8_many_atoms/
 #
 # nice interactive animations:
 #
 # https://lampz.tugraz.at/~hadley/ss1/phonons/1d/1d2m.php
 M1=1; M2=2
 MM = 1/M1 + 1/M2
 Q = 200; KS = 2π.*(0:Q-1)./Q # use finer grid
 ωp(k) = sqrt.(MM + sqrt( MM^2 - 4sin(k/2)^2/M1/M2))
 ωm(k) = sqrt.(MM - sqrt( MM^2 - 4sin(k/2)^2/M1/M2))
 figure(1,figsize=(2,1.5))
 plot(KS,ωp.(KS),"C1") # optical
 plot(KS,ωm.(KS),"C1") # acoustic

 # plots dispersion relation from block matrix rep.
 for j=1:q
  plot(ks[j]*ones(p),sqrt.(max.(real.(λ[:,j]),0)),"C0.")
 end

 xticks([0,π,2π],[0,L"$\pi$",L"$2\pi$"])

 savefig("dispersion.pgf",bbox_inches="tight")


 # find minimum distance from each of the dispersion curves to 
 # the eigenvalues
 errp = zeros(q)
 errm = zeros(q)
 for (j,k) ∈ enumerate(ks)
  c = sqrt.(max.(real.(λ[:,j]),0))
  errp[j] = minimum(abs.(ωp(k) .- c))
  errm[j] = minimum(abs.(ωm(k) .- c))
 end
 println("errorp = ",maximum(errp),", errorm = ",maximum(errm))
end

# an example with random conductivity
function greenper2d(n=(81,81),p=(3,3),k=0.3+5e-3im)
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

# a checkerboard pattern
function run_greenper2d()
 p = (4,4)
end
