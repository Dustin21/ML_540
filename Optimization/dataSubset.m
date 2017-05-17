function [ X,y ] = dataSubset( X,y,sampleN )
%DATASUBSET Summary of this function goes here
%   Detailed explanation goes here

% dimensions
[n,d] = size(X);

% create subset of X 
X = X(randi(n,sampleN,1),:);
y = y(randi(n, sampleN, 1));

end

