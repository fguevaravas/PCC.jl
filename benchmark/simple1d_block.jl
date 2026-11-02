# visualization of the block matrix approach in a simple case
using PCC, LinearAlgebra, PyPlot
# a simple 1D example
v = imread("tree32.png")
p = size(v,2); n=p*5
k = zeros(n,p);
k[1:size(v,1),:]=1 .-v

function complex_to_rgb(X)
 hue = (π.+angle.(X))/(2π)
 mag = abs.(X); mag = mag/maximum(mag);
 mag = mag.^0.4# use gamma corretion 
 #Xhsv = cat(hue,ones(size(X)),mag,dims=3) # black is zero
 Xhsv = cat(hue,mag,ones(size(X)),dims=3) # white is zero
 Xrgb = PyPlot.matplotlib.colors.hsv_to_rgb(Xhsv)
end

# the actual matrix
q = n ÷ p
A = hcat([circshift(k,(i*p,0)) for i=0:(q-1)]...)
imsave("example_A.png",A,cmap="gray_r")

# the reshaped version
kk = reshape(k,p,q,p)
imsave("example_kk.png",vcat([kk[:,i,:] for i=1:q]...),cmap="gray_r")

# the transformed version
fkk = fft(kk,2)
imsave("example_fkk.png",complex_to_rgb(vcat([fkk[:,i,:] for i=1:q]...)))

# the block diagonal
Ahat = zeros(ComplexF64,n,n)
for i=0:q-1
 Ahat[i*p .+ (1:p), i*p .+ (1:p) ] = fkk[:,i+1,:];
end

# the matrix Q
xhat = x->fft(reshape(x,p,q),2)/sqrt(q)
Q = zeros(ComplexF64,n,n)
for i=1:n
 Q[:,i] = vec(xhat([i==j for j=1:n]))
end

imsave("example_Ahat.png",complex_to_rgb(Ahat))
imsave("example_QH.png",complex_to_rgb(Q))
imsave("example_Q.png",complex_to_rgb(Q'))

println("Should be small = ",norm(A - Q'*Ahat*Q));
dx = 1/256
R = [ sqrt(x^2 + y^2) for x ∈ -1:dx:1, y ∈ -1:dx:1]
R .*= (R.<=1)
P = [ angle(x+im*y) for x ∈ -1:dx:1, y ∈ -1:dx:1]
Z = R.*exp.(im*P)
ZZ = complex_to_rgb(Z)
imsave("example_ref.png",ZZ)

