function [ stochMat ] = marginalCK( p0, pT )

d = size(pT,3)+1;
T = length(p0);
stochMat = zeros(T,d);
stochMat(:,1) = p0;

for i = 2:d
    stochMat(:,i) = stochMat(:,i-1).' * pT(:,:,i-1);
end
    
end

