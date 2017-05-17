function [ maxSeq ] = marginalDecode( p0, pT )

d = size(pT,3)+1;
T = length(p0);
stochMat = zeros(T,d);
stochMat(:,1) = p0;
maxSeq = zeros(1,d);
maxSeq(1,1) = find(max(p0));


for i = 2:d
    stochMat(:,i) = stochMat(:,i-1).' * pT(:,:,i-1);
    
    if stochMat(1,i) > stochMat(2,i)
        maxSeq(1,i) = 1;
    else
        maxSeq(1,i) = 2;
    end    
end
   
end

