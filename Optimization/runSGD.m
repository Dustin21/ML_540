function [ff] = runSGD( X,y,alphaIn,stepSizeDyn )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% run SGD with optimal alpha

% dimensions
[n,d] = size(X);
lambdaFull = 1;

% Initialize
maxPasses = 10;
progTol = 1e-4;
w = zeros(d,1);
lambda = lambdaFull/n; % The regularization parameter on one example

w_old = w;
f_old = 0;

for t = 1:maxPasses*n

    % Choose variable to update
    i = randi(n);

    % Evaluate the gradient for example i
    [f,g] = logisticL2_loss(w,X(i,:),y(i),lambda);

    % Choose the step-size
    %alpha = 1/(lambda*t);
    alpha = alphaIn;
    

    if stepSizeDyn == true
        % Increase alpha by 10% if f improves,
        % otherwise reduce alpha by 50%
        if f - f_old < 0
            alpha = alpha + alpha*0.1;
        else
            alpha = alpha - alpha*0.5;
        end
    end

    % Take the stochastic gradient step
    w = w - alpha*g;

    if mod(t,n) == 0
        change = norm(w-w_old,inf);
        fprintf('Passes = %d, function = %.4e, change = %.4f\n',t/n,logisticL2_loss(w,X,y,lambdaFull),change);
        if change < progTol
            fprintf('Parameters changed by less than progTol on pass\n');
            break;
        end
        w_old = w;
        f_old = logisticL2_loss(w,X(i,:),y(i),lambda);
    end
end

ff = logisticL2_loss(w,X,y,lambda);
fprintf('\n');

end

