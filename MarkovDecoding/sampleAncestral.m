function [ x_samples ] = sampleAncestral( p0, pT, n )

% initialize params
d = size(pT, 3)+1;
x_samples = zeros(n,d);

% run ancestral sampling
for i = 1:n
    if rand < p0(1,1)
        x_samples(i,1) = 0;
    else
        x_samples(i,1) = 1;
    end
    
    for j = 2:d
        if rand < pT(x_samples(i,j-1)+1, 1, j-1)
            x_samples(i,j) = 0;
        else
            x_samples(i,j) = 1;
        end
    end
end

end

