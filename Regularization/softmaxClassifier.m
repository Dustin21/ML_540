function [model] = softmaxClassifier(X,y)

% Compute sizes
[n,d] = size(X);
k = max(y);

W = zeros(d,k);     

maxFunEvals = 500;  
verbose = 1;        

% This is how you compute the function and gradient:
[f,g] = softMax(W(:),X,y,k);

% Derivative check that the gradient code is correct:
[f2,g2] = autoGrad(W(:),1,@softMax,X,y,k);

fprintf('f : %d\n', f);
fprintf('f2: %d\n', f2);

if max(abs(g-g2) > 1e-4)
    fprintf('User and numerical derivatives differ:\n');
    [g g2]
else
    fprintf('User and numerical derivatives agree.\n');
end

W(:) = findMin(@softMax,W(:),maxFunEvals,verbose,X,y,k);

model.W = W;
model.predict = @predict;
end

function [f,g] = softMax(w,X,y,k)
[n,d] = size(X);
k = max(y);

% transform to d x k matrix
W = reshape(w, [d k]);

% Loss function
f = 0;
for i = 1:n
    yi = y(i);      
    Xi = X(i,:);    
    f = f - W(:,yi)' * Xi' + log(sum(exp(Xi * W)));
end

% Compute gradient
g = zeros(d,k);
for c = 1:k
    eachK = zeros(n,d);
    for i = 1:n
        Xi = X(i,:);
        
        numerator = exp(W(:,c)' * Xi') * Xi;
        denominator = sum(exp(Xi * W));

        eachK(i,:) = - Xi * (y(i) == c) + numerator / denominator;
    end
    g(:,c) = sum(eachK);
end

% coerce to 1D vector
g = g(:);
end

function [yhat] = predict(model,X)
W = model.W;
[~,yhat] = max(X*W,[],2);
end