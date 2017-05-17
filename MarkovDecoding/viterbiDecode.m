function [ decoding ] = viterbiDecode( p0, pT )

[d,~,n] = size(pT);
maxprob = zeros(n+1,d);
argmax = zeros(n+1,d);
decoding = zeros(n+1,1);
maxprob(1,:) = p0;

% forward pass
for i = 1:n
    for j = 1:d
        if(maxprob(i,1) * pT(1,j,i) > maxprob(i,2) * pT(2,j,i))
            maxprob(i+1,j) = maxprob(i,1) * pT(1,j,i);
            argmax(i+1,j) = 1;
        else
            maxprob(i+1,j) = maxprob(i,2) * pT(2,j,i);
            argmax(i+1,j) = 2;
        end
    end
end

% decoding optimal path
if(maxprob(i+1,1) > maxprob(i+1,2))
    decoding(n+1) = 1;
else
    decoding(n+1) = 2;
end

for j = 1:n
    decoding(n+1-j) = argmax(n+1-j+1, decoding(n+1-j+1));
end

end

