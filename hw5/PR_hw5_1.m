% PR_hw5_1 K_Means
clear all;
close all;
clc;

% mvnrnd - Multivariate normal random numbers
%  This MATLAB function returns an n-by-d matrix R of random vectors chosen from
%  the multivariate normal distribution with mean MU, and covariance SIGMA.
Sigma = [1, 0; 0, 1];
mu1 = [1, -1];
x1 = mvnrnd(mu1, Sigma, 200);
mu2 = [5.5, -4.5];
x2 = mvnrnd(mu2, Sigma, 200);
mu3 = [1, 4];
x3 = mvnrnd(mu3, Sigma, 200);
mu4 = [6, 4.5];
x4 = mvnrnd(mu4, Sigma, 200);
mu5 = [9, 0.0];
x5 = mvnrnd(mu5, Sigma, 200);
% Obtain the 1000 data points to be clustered
X = [x1; x2; x3; x4; x5];
real_mu = [mu1; mu2; mu3; mu4; mu5];
% Show the data point
plot(x1(:,1), x1(:,2), 'r.'); hold on;
plot(x2(:,1), x2(:,2), 'b.');
plot(x3(:,1), x3(:,2), 'k.');
plot(x4(:,1), x4(:,2), 'g.');
plot(x5(:,1), x5(:,2), 'm.');

target = [linspace(1,1,200),linspace(2,2,200),linspace(3,3,200),linspace(4,4,200),linspace(5,5,200)];
k = 5;
[ reorder_mu,label,iteration,acc,mse ]  = K_Means( X,target,k );
