function [ model ] = svRegression( X, y, epsilon )
%SVREGRESSION Summary of this function goes here
%   Detailed explanation goes here

[n,d] = size(X);

% add intercept dimention
d = d+1;

% formulate linear prog problem
C = [zeros(d,1); ones(n,1)];
Z = [ones(n,1), X];
A = [Z, -eye(n); -Z, -eye(n)];
b = [y + epsilon; -y + epsilon];
lowerBound = [-inf(d,1); zeros(n,1)];

% linear program
[wr] = linprog(C, A, b,[],[],lowerBound);

% extract optimal weights
w = wr(1:d);
model.w = w;

% predict model
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
[t,~] = size(Xhat);

Zhat = [ones(t,1) Xhat];

yhat = Zhat*model.w;
end
