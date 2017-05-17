function [ model ] = generativeGaussianSSL( Xtrain, Ytrain, Xtilde, k )

if nargin < 3,
    k = 1;
end

model.Xtrain = Xtrain;
model.Ytrain = Ytrain;
model.Xtilde = Xtilde;
model.K = k;
model.predict = @(model, Xtest) predict(model, Xtest);
end

function yhat = predict(model, Xtest)
%initialize parameters
w = zeros(1,k);
mu = zeros(1,k);
cov = zeros(k,k);


%E-step
for i=1:k
    f = mvnpdf(model.Xtrain, mu(k), cov) * w(k) /Z;

%M-step
mu_k = (1/(nc + rc)) * (sum((y == x) * x) + sum(r * x));
cov_k = 

nTest = size(Xtest, 1);
yhat = zeros(nTest, 1);
dist = pdist2(Xtest, model.Xtrain);
for i=1:nTest,
    [~, perm] = sort(dist(i, :));
    s_perm = perm(1:model.K);
    s_label = model.Ytrain(s_perm);
    yhat(i)=mode(s_label);
end
end