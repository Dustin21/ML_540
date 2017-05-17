function [model] = leastSquaresRBFL2(X,y, sig,lambda)

model.X_rbfcenters  = X;
%change basis
X = rbfBasis(X,model.X_rbfcenters,sig);

% Compute sizes
[n,d] = size(X);
% Add bias variable
Z = [ones(n,1) X];

% Solve least squares problem with L2 regularization
w = (Z'*Z + lambda*(eye(n+1)))\Z'*y;

model.rbfBasis = X;
model.sig = sig;
model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
Xhat = rbfBasis(Xhat, model.X_rbfcenters, model.sig); %change basis before prediction
[t,d_t] = size(Xhat);
Zhat = [ones(t,1) Xhat]; 
yhat = Zhat*model.w;
end

%converts matrix X to rbf basis using training data (X_rbfcenters)
function [Xrbf] = rbfBasis(X,X_rbfcenters,sig)
[n,~] = size(X); 
[d_rbf, ~] = size(X_rbfcenters);
Xrbf = ones(n,d_rbf);
for i = 1:n
    for j = 1:d_rbf
        Xrbf(i,j) = normpdf(sum((X(i,:)-X_rbfcenters(j,:)).^2),  0.0, sig);        
    end
end
end