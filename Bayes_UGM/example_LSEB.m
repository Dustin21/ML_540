
% Clear variables and close figures
clear all
close all

% Load data
load basisData.mat % Loads X and y
[n,d] = size(X);


model = leastSquaresEmpiricalBaysis(X, y);