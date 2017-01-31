% modeling distributions with the variational-autoencoder
% this simple example we learn a Gaussaian scale mixture distributed random
% varaible. In the simplex case where the Gaussian component is a
% univariate normal random variable and the multiplicative mixer is a
% Rayleigh distributed variable with size parameter 1.

clear all
close all
clc 

%% Create data 
N = 3000;
G = randn(1, N); % gaussian component
S = raylrnd(1, 1, N); % Raylegih mixer;

X = G.*S;

N_test = 100;
G = randn(1, N_test); % gaussian component
S = raylrnd(1, 1, N_test); % Raylegih mixer;

X_test = G.*S;

%% load basic stuff for neural nets
run nnet/basic.m;
addpath('./nnet');

%% create encoder and decoder networks
code_dim = 2;
in_dim = size(X, 1);
hid_dim = 100;
binary = false;
run VAE_standard_net.m
AE.encoder = encoder;
AE.decoder = decoder;

%% And Now the algorithm finally
trainparams.batchsize = 100;
trainparams.n_epochs = 100;
trainparams.stepsize = 0.001;
trainparams.n_monte = 10; % number of montecarlo samples 

AE = trainVariationalAutoencoder(AE, X, trainparams);

%% collect some samples

X_samples = sampleFromVAE(AE, N);

PX = naiveMarginalLikelihood(X_test, AE.decoder, 100);


%% Compare models
[X_count, bin] = hist(X, 10);
[X_samples_count, bin] = hist(X_samples, bin);
binsize = mean(bin(2:end) - bin(1:end-1));

semilogy(bin, X_count/(N*binsize))
hold on
semilogy(bin, X_samples_count/(N*binsize), 'r')
semilogy(X_test, PX, 'k.')
xlim([-10, 10])
ylim([1e-4, 1])
grid on 
legend('Estimated distribution from Training Data', 'Estimated distribution from VAE Samples', 'Estimated likelihood on Test Data using the VAE', 'Location', 'South')