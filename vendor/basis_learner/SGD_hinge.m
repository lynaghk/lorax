function [w] = SGD_hinge(X,Y,lam)
% function [w] = SGD_hinge(X,Y,lam)
% Performs stochastic gradient descent to solve regularized
% hinge loss over data X,Y (with L2 regularization parameter lam).

X=X'; % Store examples column-wise to speed training
factor = sqrt(2/lam);

num_epochs = 50; % How many times to go over dataset.

m = size(X,2);
n = size(X,1);
w = zeros(1,n);
inds = randperm(m);

t = 1;
%fprintf('Running SGD, epoch ');
for epoch=1:num_epochs
    %if (mod(epoch,ceil(num_epochs/10))==0) 
    %    fprintf(' %d ',epoch); 
    %end;
    for ind = inds
        if (1>Y(ind)*w*X(:,ind))
            w = (1-1/t)*w+Y(ind)*X(:,ind)' / (lam*t);
        else
            w = (1-1/t)*w;
        end;
        w = min(1,factor/norm(w))*w;
        t = t+1;
    end;
end;
%fprintf('\n');
w=w';