%%% This is script is a sime illustration of the variational autencoder algorithm
% D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes." Second
% international conference on learning representaions ICLR, 2014.
clear all
close all
clc

vae_root = '/Data/Luis/Research/UM/Code/VAE_matlab/';
%% Generate data from two GSM distributions
N_train = 500;
N_dim = 2;
% Gaussian latent components
COV_x = sampleCOV(N_dim);
COV_o = sampleCOV(N_dim);
mu_x = randn(N_dim, 1);
mu_o = 2*randn(N_dim, 1);
% Simple case elliptic distribution (Gaussian are centered before scaling)
Gx_data = mvnrnd(zeros(1, N_dim), COV_x, N_train)';
Go_data = mvnrnd(zeros(1, N_dim), COV_o, N_train)';
% Scale components 
h_x = 1;
h_o = 2;
Sx_data = raylrnd(h_x, 1, N_train);
So_data = raylrnd(h_o, 1, N_train);
% multiplicative mixing 
Xx_data = bsxfun(@times, Gx_data, Sx_data);
Xo_data = bsxfun(@times, Go_data, So_data);
% add means
Xx_data = bsxfun(@plus, Xx_data, mu_x);
Xo_data = bsxfun(@plus, Xo_data, mu_o);
% show me the money
figure
plot(Xx_data(1,:), Xx_data(2,:), 'rx')
hold on 
plot(Xo_data(1,:), Xo_data(2,:), 'bo')



%% load basic stuff for neural nets
run init_nnet.m
%% create encoder and decoder networks
code_dim = 2;
in_dim = size(Xx_data, 1);
hid_dim = 200;
binary = false;
run VAE_mnist_net.m

Xx.encoder = encoder;
Xx.decoder = decoder;
Xo.encoder = encoder;
Xo.decoder = decoder;

%% And Now the algorithm finally
trainparams.batchsize = 100;
trainparams.n_epochs = 1;
trainparams.stepsize = 0.001;
trainparams.n_monte = 1; % number of montecarlo samples 
clc 
disp('Training Autoencdoer for images of 3')
Xx = trainVariationalAutoencoder(Xx, Xx_data, trainparams);
disp('Training Autoencoder for iamges of 8')
Xo = trainVariationalAutoencoder(Xo, Xo_data, trainparams);

PXx_x = naiveMarginalLikelihood(Xx_data(:,ceil(size(Xx_data, 2)*rand(1,1000))), Xx.decoder, 100);
PXx_o = naiveMarginalLikelihood(Xo_data(:,ceil(size(Xo_data, 2)*rand(1,1000))), Xx.decoder, 100);
PXo_x = naiveMarginalLikelihood(Xx_data(:,ceil(size(Xx_data, 2)*rand(1,1000))), Xo.decoder, 100);
PXo_o = naiveMarginalLikelihood(Xo_data(:,ceil(size(Xo_data, 2)*rand(1,1000))), Xo.decoder, 100);

save vae_oddball_weights Xx Xo 
save vae_oddball_samples Xx_data Xo_data