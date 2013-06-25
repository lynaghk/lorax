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

function Results = BuildOutputLayers(F,Y,trainend,widths,lambdaRange)

Results = zeros(length(widths),length(lambdaRange),2);
% last index is train or test error
fprintf('Building Output Layer\n trainend=%d. widths = [',trainend);
fprintf('%d ',widths);
fprintf('], lambdaRange = [');
fprintf('%g ',lambdaRange);
fprintf(']\n');

if (length(unique(Y))>2)
    lossType = 'multiclass_hinge';
else
    lossType = 'hinge';
end;

szCounter=1;
for sz=cumsum(widths)
    lambdaCounter = 1;
    for lambda=lambdaRange
        fprintf('Training depth %d, lambda %g\n',szCounter+1,lambda);
        switch lossType
            case 'hinge'
                w = SGD_hinge(F(1:trainend,1:sz),Y(1:trainend),lambda);
                preds = sign(F(:,1:sz)*w);
            case 'multiclass_hinge'
                minY = min(Y);
                Y = Y - minY+1;
                w = MC_SGD(F(1:trainend,1:sz)',Y(1:trainend),lambda);
                [~,preds] = max(F(:,1:sz)*w,[],2);
                preds = preds+minY-1;
        end;
        Results(szCounter,lambdaCounter,1) = mean(preds(1:trainend)~=Y(1:trainend));
        Results(szCounter,lambdaCounter,2) = mean(preds(trainend+1:end)~=Y(trainend+1:end));
        fprintf('Train Error: %g, Test Error: %g\n',Results(szCounter,lambdaCounter,1),Results(szCounter,lambdaCounter,2));
        lambdaCounter = lambdaCounter+1;
    end;
    szCounter = szCounter + 1;
end;
bestTestError = min(min(Results(:,:,2)));
[i,j] = find(Results(:,:,2) == bestTestError);
fprintf('Best Test Error Result:%g\n - Architecture [',bestTestError);
fprintf('%d ',widths(1:i));
fprintf(']\n - lambda %g (no. %d in lambdaRange)\n',lambdaRange(j),j);
