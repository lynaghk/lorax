function [W] = orthonormalize_approx(G,width)
% function [W] = orthonormalize_unsup_approx(G,width)
% Orthonormalizes G and return weight matrix W such that 
% G*W is is orthonormal, corresponding to largest 'width' singular values
% Uses approximate SVD: G*W is still orthonormal, but only approximately
% corresponds to the `width' largest singular values.

% Internal parameters
p = width; % oversampling factor
tol = 10^(-9);

% Perform approximate SVD with a single quality improvement iteration
% (multiplying by G*G' once)
Y = G*(G'*(G*randn(size(G,2),width+p)));
[~,D,W] = svd(orth(Y)'*G,'econ');
W = W(:,diag(D)>tol);
W = W(:,1:min(width,end));

% Correct so that G*W is indeed orthonormal (G*W not exactly orthonormal,
% since U is only approximate singular vectors of G). 
B = G*W; 
[~,~,W2] = svd(B,'econ');
B = B*W2;
W = W*W2;
norms = sqrt(sum(B.^2,1));
W = W./repmat(norms,size(W,1),1);
