load quantum.mat
[n,d] = size(X);
lambdaFull = 1;

% Initialize
maxPasses = 10;
progTol = 1e-4;
w = zeros(d,1);
lambda = lambdaFull/n; % The regularization parameter on one example

% Adagrad param initialize
alpha = 1e-2;
delta = 1e-6;
g_old = zeros(d,1);

% Stochastic gradient
w_old = w;
for t = 1:maxPasses*n
    
    % Choose variable to update
    i = randi(n);
    
    % Evaluate the gradient for example i
    [f,g] = logisticL2_loss(w,X(i,:),y(i),lambda);
    
    % Adagrad
    g_old = g_old + g.*g;
    g_adj = g./(delta + sqrt(g_old));
    w = w - alpha*g_adj;
        
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