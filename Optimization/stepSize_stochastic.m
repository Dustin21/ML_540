load quantum.mat

% subset data
[Xsub,ysub] = dataSubset(X,y,1500);

% init output vectors
v = linspace(0,0.01,100);
out = zeros(length(v),1);

for m = 1:length(v)
    out(m) = runSGD(Xsub,ysub,v(m),false);
end

% extract optimal params
[val, idx] = min(out);
alphaOpt = v(idx);

% run SGD with optimal alpha and 
% dynamically adjusted step size
runSGD(X,y,alphaOpt, true)

