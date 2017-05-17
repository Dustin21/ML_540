clear all;
clc;

load viterbiData.mat

%decode_short1 = exactDecode(p0,pT_short1)

%decode_short2 = exactDecode(p0,pT_short2)

%decode_long = exactDecode(p0,pT_long);

%*******************************************
%************ Ancestral Sampling ***********
%*******************************************
% ancestral sampling
%x_marginal = sampleAncestral(p0, pT_long, 10000);
%sum(x_marginal == 0)/size(x_marginal,1)

%*******************************************
%*********** Chapman-Kolmogorov ************
%*******************************************

% Chapman-Kolmogorov marginal computation
%x_trueMarginal = marginalCK(p0, pT_long)
%sum(x_trueMarginal,1)

%*******************************************
%************ Marginal Decoding ************
%*******************************************

% decode marginal probability
marginalDecode(p0, pT_long)

%*******************************************
%************* Viterbi Decoding ************
%*******************************************

% run Viterbi decoding algorithm
%viterbiDecode(p0, pT_long)

%*******************************************
%*************** Conditioning **************
%*******************************************

% 1.2.1
% Sampling of conditional P(xj|x1=2)
p00 = [0,1]; % x1=2 with certainty 
x_marginal = sampleAncestral(p00, pT_long, 10000);
sum(x_marginal == 1)/size(x_marginal,1)

% 1.2.2
p00 = [0,1]; % x1=2 with certainty 
x_marginal = marginalCK(p00, pT_long);
x_marginal(2,:)

% 1.2.3
p00 = [0,1]; % x1=2 with certainty 
x_opt = viterbiDecode(p00, pT_long);
x_opt.'

% 1.2.4
p00 = [1,0];
x_opt = viterbiDecode(p00, pT_long);


