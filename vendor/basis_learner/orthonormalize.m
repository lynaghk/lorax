function [W] = orthonormalize(G,width)
% function [W] = orthonormalize_unsup(G,width)
% Orthonormalizes G and return weight matrix W such that 
% G*W is is orthonormal, corresponding to largest 'width' singular values

[~, D,W] = svd(G,'econ');
W = W(:,diag(D)>0.0001);
W = W(:,1:min(width,end));
B = G*W;
W = W./repmat(sqrt(sum(B.^2,1)),size(W,1),1);