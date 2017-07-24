function [normalize_net]=jb_normalizeNet(net,ntype)
%clear all
%net=[ 1 2 5 ; 4 2 1;3 1 6];
if (nargin<2)
    ntype=0;
end
[m,n,t]=size(net);
for i=1:t
    aa(:,:)=net(:,:,i);
    rP=reshape(aa,m*n,1)
    switch(ntype)
    case 0
        dt=jb_normalize(rP);
    case 1
        dt=jb_scaling(rP,-1,1);
    case 2
        dt=jb_scaling(rP,0,1);
    end
    cc=reshape(dt,m,n);
    norm_net(:,:,i)=cc;
end
normalize_net=norm_net;