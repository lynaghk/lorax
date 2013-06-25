% Copyright (c) 2013 Ohad Shamir
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met: 
% 
% 1. Redistributions of source code must retain the above copyright notice, this
%    list of conditions and the following disclaimer. 
% 2. Redistributions in binary form must reproduce the above copyright
%    notice, this list of conditions and the following disclaimer in the documentation
%    and/or other materials provided with the distribution. 
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


function F = BuildNetwork(X,Y,widths,trainend)
% function F = BuildNetwork(X,Y,widths,trainend)
% Build network for classification.
% Input:
% - Instances X (each row is an instance);
% - Target values vector Y (either binary, e.g. {-1,+1}, or multiclass, e.g. {1,2,3,..})
% - widths vector - width(t) is desired width of layer (t)
% - trainend - X(1:trainend,:) is assumed to be training instances, and
% X(trainend+1:end,:) is assumed to be test instances
% Output: A matrix F, so that F(i,j) is the response of element j on
% instance X(i,:)

% Internal parameters
BuildMethodFirstLayer = 'exact';
% {'exact','approx'}: Can be either using exact SVD or approximate SVD (less precise but
% faster for large problems)
batchSize = 50;
tol = 10^(-9); % Coarse tolerance bound to avoid rounding errors

fprintf('Building F for widths: [');    fprintf('%d ', widths);    fprintf(']\n');
X = X(:,sum(abs(X),1)>0); % eliminate zero coordinates
Ytrain = Y(1:trainend); clear Y; % Just need the training coordinates
classLabels = unique(Ytrain);
F = zeros(size(X,1),sum(widths));
OF = zeros(trainend,size(F,2)); % Maintain orthonormal basis of F.

% Create input layer
switch BuildMethodFirstLayer
    case 'exact'
        W = orthonormalize([ones(trainend,1) X(1:trainend,:)],widths(1));
    case 'approx'
        W = orthonormalize_approx([ones(trainend,1) X(1:trainend,:)],widths(1));
    otherwise
        error('Unknown BuildMethodFirstLayer');
end;
F(:,1:widths(1)) = [ones(size(X,1),1) X]*W;
OF(:,1:widths(1)) = F(1:trainend,1:widths(1));
F(:,1:widths(1)) = F(:,1:widths(1))./repmat(sqrt(mean(F(1:trainend,1:widths(1)).^2,1)),size(F,1),1); % Normalize
clear X;

% Create intermediate layers
for t=2:length(widths)
    beginThis = sum(widths(1:t-1))+1; % Column of F where this layer should go to.
    beginLast = sum(widths(1:t-2))+1;
    if (length(classLabels)>2) % Multiclass: create indicator matrix Vfor class types
        V = zeros(trainend,length(classLabels));
        for i=1:length(classLabels)
            V(:,i) = (Ytrain==i);
        end;
    else % Binary: create sign vector V representing class
        V = sign((Ytrain==classLabels(1))-0.1);
    end;
    V = V - OF(:,1:beginThis-1)*(OF(:,1:beginThis-1)'*V);
    r = beginThis;
    while r <= beginThis+widths(t)-1 % Begin mini-batch constructions
        scores = zeros(widths(1),widths(t-1)); %Score for each product of elements in first and last constructed layers
        OV = orth(V);
        for i=1:widths(1) % Compute scores - one row of the scores matrix at a time
            Ci = repmat(F(1:trainend,i),1,widths(t-1)).*F(1:trainend,beginLast:beginThis-1);
            Ci = Ci-OF(:,1:r-1)*(OF(:,1:r-1)'*Ci);
            normCi = sqrt(sum(Ci.^2,1));
            Ci = Ci./repmat(normCi,trainend,1);
            scores(i,:) = sum((OV'*Ci).^2,1);
            scores(i,normCi<tol) = -inf; % Exclude vectors in (or almost in) the span of OF
        end;
        [~,inds] = sort(reshape(scores,1,numel(scores)),'descend');
        [I J] = ind2sub(size(scores),inds);
        numNewColumns = min(batchSize,beginThis+widths(t)-r);
        Cchosen = zeros(trainend,numNewColumns);
        OC = zeros(size(Cchosen));
        l=1;
        ind = 1;
        while (l<=numNewColumns)
            F(:,r-1+l) = F(:,I(ind)).*F(:,beginLast+J(ind)-1);
            Cchosen(:,l) = F(1:trainend,r-1+l);
            OC(:,l) = Cchosen(:,l)-OF(:,1:r-1)*(OF(:,1:r-1)'*Cchosen(:,l));
            OC(:,l) = OC(:,l)-OC(:,1:l-1)*(OC(:,1:l-1)'*OC(:,l));
            normOCl = norm(OC(:,l));
            if (normOCl > tol) % Accept new vector if it's linearly independent from previous ones
                OC(:,l) = OC(:,l)/normOCl;
                F(:,r-1+l) = F(:,r-1+l)/sqrt(mean(F(1:trainend,r-1+l).^2));
                l = l+1;
            end;
            ind = ind + 1;
            if (ind > length(I)) % Exhausted all candidate vectors
                if (l==1) % No candidates at all found
                    error('No linearly-independent candidates were found in mini-batch');
                else % Candidate vectors were found, just less than batchSize
                    fprintf('Warning: Only %d (out of mini-batch size %d) linearly-independent candidate vectors were found and added.\n',l-1,batchSize);
                end;
                break;
            end;
        end;
        if (normOCl > tol) % If l was again incremented in last loop previously
            l = l-1;
        end;
        OF(:,r:r-1+l) = OC(:,1:l);
        V = V-OC(:,1:l)*(OC(:,1:l)'*V);
        r = r + l;
        %OF(:,1:r-1) = orth(OF(:,1:r-1)); % possibly re-orthogonalize for numerical stability
        fprintf('Built %d out of %d elements in layer %d\n',min(widths(t),r-beginThis),widths(t),t);
    end;
end;