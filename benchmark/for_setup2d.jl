# a simple 2D image for the sample
using PCC, LinearAlgebra, PyPlot, TestImages, Images, ImageTransformations, Random, FFTW

function run_setup()
 p = 3; n = 129
 mysavefig(x) = savefig("$x.png", bbox_inches="tight")

 include("digits.jl")
 k = zeros(n,n,p^2)
 for i=1:9
  k[1:5,1:5,i] = digits[:,:,i]
 end
 k = circshift(k,(-2,-2,0))
 k = reshape(k,n,n,p,p)
 k̂ = pcc2d_gemv_blkm_prep(k)

 x = Float32.(imresize(testimage("cameraman"),(n,n)))
 imsave("for_setup_x.png",x,cmap="gray")
 y = real.(pcc2d_gemv_blkm(x,k̂))
 imsave("for_setup_y.png",y,cmap="gray")

 # generate a cartoon PSF
 i0 = j0 = n÷2 + 1
 x = [ (i-i0)^2 + (j-j0)^2 <= 4^2 for i=1:n, j=1:n]

 mask1 = [ exp( -((i-i0)^2 + (j-j0)^2)/ 16^2 ) for i=1:n, j=1:n]
 mask2 = [ exp( -((i-i0)^2 + (j-j0)^2)/ 16^2 ) for i=1:n, j=1:n]
 yhat = (randn(n,n) + 1im*randn(n,n)).*  mask1
 y = abs.(ifft(fftshift(yhat))) .* mask2

 imsave("cartoonpsf_x.png",x,cmap="gray")
 imsave("cartoonpsf_y.png",y,cmap="gray")

end



