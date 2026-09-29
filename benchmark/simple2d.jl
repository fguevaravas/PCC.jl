# simple two dimensional examples
using PCC, LinearAlgebra, PyPlot
p = 3; n = 15

mysavefig(x) = savefig("$x.png", bbox_inches="tight")

include("digits.jl")

# use digits as PSFs and correct for shifts so they don't dance around
k = zeros(n,n,p^2)
for i=1:9
 k[1:5,1:5,i] = digits[:,:,i]
end
k=circshift(k,(-2,-2,0))
k = reshape(k,n,n,p,p)
#for i1=1:p,i2=1:p
# k[:,:,i1,i2] = circshift(k[:,:,i1,i2],(1-i1,1-i2))
#end

s1 = [
0 0 0 0 0 0 0 0 0;
0 0 0 0 0 1 0 0 0;
0 0 0 0 0 0 0 0 0;
0 0 0 0 0 0 0 0 0;
0 0 1 0 0 0 0 0 1;
0 0 0 0 0 0 0 0 0;
0 0 0 0 0 0 0 0 0;
0 0 0 0 0 0 0 0 0;
0 0 0 0 0 0 0 0 0;
]

s2 = [
0 0 0 0 0 0 0 0 0;
0 0 0 0 0 1 0 0 0;
0 0 0 0 0 0 0 0 0;
0 0 0 0 0 0 0 0 0;
0 0 0 0 0 0 0 0 0;
0 1 0 0 0 0 0 0 1;
0 0 0 0 0 0 0 0 0;
0 0 0 0 0 0 0 0 0;
0 0 0 0 0 0 0 0 0;
]



function extend_by_zeros(x,n1,n2)
 m1,m2=size(x)
 @assert n1 >= m1 && n2 >= m2
 s1 = (n1-m1) ÷ 2
 s2 = (n2-m2) ÷ 2
 y = zeros(n1,n2)
 y[ s1 .+ (1:m1), s2 .+ (1:m2) ] = x
 return y
end

s1 = extend_by_zeros(s1,n,n)
s2 = extend_by_zeros(s2,n,n)

# canonical basis vectors
signals = vec([ [Float64(j1==(2p+i1) && j2==(2p+i2)) for j1=1:n,j2=1:n] for i1=1:p, i2=1:p ])

push!(signals,s1)
push!(signals,circshift(s1,(-1,0)))
push!(signals,circshift(s1,( 1,0)))
push!(signals,circshift(s1,(0,-1)))
push!(signals,circshift(s1,(0, 1)))
push!(signals,s2)
push!(signals,circshift(s2,(-1,0)))
push!(signals,circshift(s2,( 1,0)))
push!(signals,circshift(s2,(0,-1)))
push!(signals,circshift(s2,(0, 1)))

# transform all signals
ys = [ pcc2d_gemv_naive(signals[i],k) for i=1:length(signals) ]

# figure out consistent scales
s1 =  maximum(maximum.(ys[1:p^2]))
s2 =  maximum(maximum.(ys[p^2+1:end]))


for (ix,x) ∈ enumerate(signals)
 # the transformed signal
 y = ys[ix]
 figure(1,figsize=(2,2),dpi=360); clf()
 imshow(x,cmap="gray_r")
 xticks((0:p:n).-0.5,[]); yticks((0:p:n).-0.5,[])
 xticks((0:n).-0.5,[],minor=true); yticks((0:n).-0.5,[],minor=true)
 tick_params(axis="both", which="both", length=0)
 box(false); 
 grid(true,which="major",color="black") 
 grid(true,which="minor",color="black",linestyle=":") 
 mysavefig("x2d_$ix")

 figure(2,figsize=(2,2),dpi=360); clf()
 imshow(y,cmap="gray_r");
 # use same color scale for kernels
 # and a different one for the other signals
 clim(0,ix <= p^2 ? s1 : s2)
 xticks((0:p:n).-0.5,[]); yticks((0:p:n).-0.5,[])
 xticks((0:n).-0.5,[],minor=true); yticks((0:n).-0.5,[],minor=true)
 tick_params(axis="both", which="both", length=0)
 box(false);
 grid(true,which="major",color="black") 
 grid(true,which="minor",color="black",linestyle=":")
 mysavefig("y2d_$ix")
end
