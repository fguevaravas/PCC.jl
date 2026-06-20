# cameraman image deconvolution
using PCC, LinearAlgebra, PyPlot, TestImages, Random

x = float(testimage("cameraman"))
# zero pad by adding removing 2 pixels so that the image has size
# divisible by p
p = 3
q = size(x,1) ÷ p
n = p*q 
x = x[1:n,1:n]

# use digits as PSFs and correct for shifts so they don't dance around
include("digits.jl")
k = zeros(n,n,p^2)
for i=1:9
 #k[1:5,1:5,i] = digits[:,:,i]
 #Z = rand(5,5)
 Z = digits[:,:,i]
 k[1:5,1:5,i] = Z/sum(vec(Z))
end
k = circshift(k,(-2,-2,0))
k = reshape(k,n,n,p,p)
k̂ = pcc2d_gemv_blkm_prep(k)

figure(1)
imshow(x,cmap="gray")


println("conditioning = ",pcc2d_cond(k̂))
s = pcc2d_svdvals(k̂)
s = sort(vec(s),rev=true)
figure(2)
semilogy(s)

y = real(pcc2d_gemv_blkm(x,k̂))
figure(3);
imshow(y,cmap="gray")

# now add noise
Random.seed!(33)
nlvl = 1e-2
y_noisy = y + nlvl*randn(n,n);
figure(4);
imshow(y_noisy,cmap="gray")



"""
periodic difference operator (block matrix kernel representation)
"""
function gradient(n1,n2,p1,p2)
 @assert (n1%p1 == 0) && (n2%p2 == 0)
 q1 = n1 ÷ p1; q2 = n2 ÷ p2
 D1 = zeros(n1,n2,p1,p2)
 D2 = zeros(n1,n2,p1,p2)
 for i1=1:p1,i2=1:p2
  D1[mod1.(i1.+[0,1],n1),i2,i1,i2] .= [-1, 1]
  D2[i1,mod1.(i2.+[0,1],n2),i1,i2] .= [-1, 1]
 end
 D̂1 = pcc2d_gemv_blkm_prep(D1)
 D̂2 = pcc2d_gemv_blkm_prep(D2)
 return D̂1,D̂2
end



# L-curve
αs = 2.0 .^(-10:10);
k̂t = pcc2d_adjoint(k̂)
k̂ty = pcc2d_gemv_blkm(y_noisy,k̂t);
eye = pcc2d_eye(n,n,p,p)
resnrm = zeros(length(αs))
xnrm = zeros(length(αs))
# calculate discrete Laplacian
D̂1,D̂2 = gradient(n,n,p,p)
L̂ = pcc2d_gemm(pcc2d_adjoint(D̂1),D̂1) + pcc2d_gemm(pcc2d_adjoint(D̂2),D̂2)
for (iα,α) ∈ enumerate(αs)
 xα = pcc2d_solve(k̂ty,pcc2d_gemm(k̂t,k̂) + α^2*L̂)
 xnrm[iα] = sqrt(norm(pcc2d_gemv_blkm(xα,D̂1))^2 + norm(pcc2d_gemv_blkm(xα,D̂2))^2) 
 resnrm[iα] = norm(pcc2d_gemv_blkm(xα,k̂) - y_noisy)
end

figure(5)
loglog(resnrm,xnrm)
xlabel("residual norm")
ylabel("regularization norm")

# do normal equations + Tikhonov
α = 1/4
@time x_noisy = pcc2d_solve(k̂ty,pcc2d_gemm(k̂t,k̂) + α^2*L̂)

# plot reg parameter choice on L curve
nrmx = sqrt(norm(pcc2d_gemv_blkm(x_noisy,D̂1))^2 + norm(pcc2d_gemv_blkm(x_noisy,D̂2))^2) 
nrmres = norm(pcc2d_gemv_blkm(x_noisy,k̂) - y_noisy)
loglog(nrmres,nrmx,"r*")


figure(6)
imshow(real(x_noisy),cmap="gray")

println("system size |n| = ",n*n)
println("relative error with ",nlvl*100,"% noise = ",100*norm(x-x_noisy)/norm(x),"%")


