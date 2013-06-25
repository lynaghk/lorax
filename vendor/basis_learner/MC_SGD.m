function w = MC_SGD(X,Y,lam)
% solve multiclass SVM using SGD
% Assume that the instances are columns of X

k = length(unique(Y));
Y = Y-min(Y)+1;

%factor = sqrt(2/lam);

num_epochs = 100; % How many times to go over dataset.
m = size(X,2);
n = size(X,1);
w = zeros(k,n);


t = 1;
%fprintf('Running SGD, epoch ');
for epoch=1:num_epochs
    inds = randperm(m);
    for ind = inds
        pred = w*X(:,ind);
        [val,j] = max(((1:k)~=Y(ind))' + pred - pred(Y(ind)));
        w = (1-1/t)*w;
        if val > 0
            w(Y(ind),:) = w(Y(ind),:) + X(:,ind)'/(lam*t);
            w(j,:) = w(j,:) - X(:,ind)'/(lam*t);
        end       
        t = t+1;
    end;
end;
%fprintf('\n');
w=w';