function [ model ] = leastSquaresEmpiricalBaysis( x, y )

% initialize grid of parameters
degree = 1:1:8;
sigma = 0.001:0.001:0.09;
lambda = 1:1:10;

% return the negative log-likelihood over parameter range
nll = logLikelihood(x, y, sigma, lambda, degree);

model.parOpt = nll.parOpt;
model.nllOpt = nll.nllOpt;
model.nll = nll.logMarginal;

end

function [opt] = logLikelihood(x, y, sigma, lambda, degree)
% initialize values
n = length(x);
nSig = length(sigma);
nLambda = length(lambda);
nDeg = length(degree);

identity = eye(n);
opt.logMarginal = zeros([nSig, nLambda, nDeg]);
opt.nllOpt = 9e10;

% iterate over grid of parameters and compute
% the marginal nll
for i = 1:nDeg
    for j = 1:nSig
        for k = 1:nLambda
            Xpoly = polyBasis(x, degree(i));
            C = identity * inv(power(sigma(j),2)) + inv(lambda(k)) * mtimes(Xpoly, Xpoly.');
            v = linsolve(C, y);
            opt.logMarginal(j,k,i) = logdet(C, Inf) + y.' * v;
                        
            % print params
            %[degree(i), sigma(j), lambda(k)]
            
            if opt.nllOpt > opt.logMarginal(j,k,i)
                opt.nllOpt = opt.logMarginal(j,k,i)
                opt.parOpt = [degree(i), sigma(j), lambda(k)]
            end 
        end
    end
end

end
        

function [Xpoly] = polyBasis(x,m)
n = length(x);
Xpoly = zeros(n,m+1);
for i = 0:m
    Xpoly(:,i+1) = x.^i;
end
end

