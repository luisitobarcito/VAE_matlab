%%% This is script is a sime illustration of the variational autencoder algorithm
% D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes." Second
% international conference on learning representaions ICLR, 2014.
clear all
close all
clc

vae_root = '/Data/Luis/Research/UM/Code/VAE_matlab/';
data_path = strcat(vae_root, 'data/');
%% Gather data from MNIST images
% This is a data set of handwritten digits 0 to 9
data_filename  = strcat(data_path, 'mnist_all.mat');
data = load(data_filename);

X3 = double(data.train3')/255;
X8 = double(data.train8')/255;



%% load basic stuff for neural nets
run init_nnet.m
%% create encoder and decoder networks
code_dim = 2;
in_dim = size(X3, 1);
hid_dim = 200;
binary = true;
run VAE_mnist_net.m

mnistAE3.encoder = encoder;
mnistAE3.decoder = decoder;
mnistAE8.encoder = encoder;
mnistAE8.decoder = decoder;

%% And Now the algorithm finally
trainparams.batchsize = 100;
trainparams.n_epochs = 3000;
trainparams.stepsize = 0.001;
trainparams.n_monte = 1; % number of montecarlo samples 
clc 
disp('Training Autoencdoer for images of 3')
mnistAE3 = trainVariationalAutoencoder(mnistAE3, X3, trainparams);
disp('Training Autoencoder for iamges of 8')
mnistAE8 = trainVariationalAutoencoder(mnistAE8, X8, trainparams);

PX3_3 = naiveMarginalLikelihood(X3(:,ceil(size(X3, 2)*rand(1,1000))), mnistAE3.decoder, 100);
PX3_8 = naiveMarginalLikelihood(X8(:,ceil(size(X8, 2)*rand(1,1000))), mnistAE3.decoder, 100);
PX8_3 = naiveMarginalLikelihood(X3(:,ceil(size(X3, 2)*rand(1,1000))), mnistAE8.decoder, 100);
PX8_8 = naiveMarginalLikelihood(X8(:,ceil(size(X8, 2)*rand(1,1000))), mnistAE8.decoder, 100);

save vae_mnist38_weights mnistAE3 mnistAE8 