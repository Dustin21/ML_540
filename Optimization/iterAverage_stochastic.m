load quantum.mat
[n,d] = size(X);
lambdaFull = 1;

% Set averaging technique
% if freeRange = TRUE, start averaging after half of the iterations,
% otherwise cumulatively average all iterations.
freeRange = true;

% Initialize
maxPasses = 10;
progTol = 1e-4;
w = zeros(d,1);
lambda = lambdaFull/n; % The regularization parameter on one example

% Stochastic gradient
w_old = w;
w1 = w;
for t = 1:maxPasses*n
    
    % Choose variable to update
    i = randi(n);
    
    % Evaluate the gradient for example i
    [f,g] = logisticL2_loss(w,X(i,:),y(i),lambda);
    
    % Choose the step-size
    alpha = 1/(lambda*t);
           
    % Take the stochastic gradient step
    if freeRange == true
        if mod(t,n) < 25000
            w = w - alpha*g;
        else
            w = w1 - alpha*g;
            w1 = (w + w1)/2;
        end
    else
        w = w1 - alpha*g;
        w1 = (w + w1)/2;
    end
    
  
    if mod(t,n) == 0
        change = norm(w-w_old,inf);
        fprintf('Passes = %d, function = %.4e, change = %.4f\n',t/n,logisticL2_loss(w,X,y,lambdaFull),change);
        if change < progTol
            fprintf('Parameters changed by less than progTol on pass\n');
            break;
        end
        w_old = w;
    end
end