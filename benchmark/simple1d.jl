using PCC, LinearAlgebra, PyPlot
# a simple 1D example
p = 3; n=12
k = [ 1 2    2 
      0 1    0
      0 0    0 
      0 0    0 
      0 0    0
      0 0    0
      0 0    0 
      0 0    0 
      0 0    0 
      0 0    0 
      0 0    0
      0 0    1]

k = k * diagm(vec(1 ./sum(k,dims=1)))
colors = ["blue","red","orange"]

mysavefig(x) = savefig("$x.pdf", bbox_inches="tight")

# the kernel
figure(1,figsize=(3,9/4)); clf();
for i=1:p
  subplot(p,1,i)
  step(0:n,[k[:,i];k[end,i]],color=colors[i],where="post")
  ylim(-0.05,1.05); xlim(0,n)
  grid(true,axis="x")
  xticks(0:n,[])
  yticks([0,1],[])
  tick_params(axis="both", which="both", length=0)
  box(false)
end
mysavefig("kernel1d")

# the signal
u = [0, 0, 0, -1, -0.5, -1, 1, 1, 0, 0, 0, 0 ]
for (ix,x) ∈ enumerate([ u , circshift(u,-1) ])
  figure(2,figsize=(3,1.5)); clf()
  step(0:n,[x;x[end]],color="black",where="post")
  xticks(0:n,[])
  grid(true)
  xlim(0,n); ylim(-1.05,1.05)
  yticks([-1,0,1])
  box("off")
  mysavefig("x1d_$ix")
  
  # the transformed signal
  y = pcc1d_naive(x,k)
  figure(3,figsize=(3,1.5)); clf()
  step(0:n,[y;y[end]],color="black",where="post")
  xticks(0:n,[])
  grid(true)
  xlim(0,n); ylim(-1.05,1.05)
  yticks([-1,0,1])
  box("off")
  mysavefig("y1d_$ix")
  
  # the signal components
  figure(4,figsize=(3,9)); clf()
  for j ∈ 1:n
    yj = [x[j]* k[mod1(i-j+1,n),mod1(j,p)] for i=1:n]
    subplot(n,1,j)
    step(0:n,[yj;yj[end]],color=colors[mod1(j,p)],where="post")
    ylim(-1.05,1.05); xlim(0,n)
    grid(true,axis="x")
    xticks(0:n,[])
    yticks([-1,0,1],[])
    tick_params(axis="both", which="both", length=0)
    box(false)
  end
  mysavefig("c1d_$ix")
end;

