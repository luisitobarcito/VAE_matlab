%%% This is script is a sime illustration of the variational autencoder algorithm
% D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes." Second
% international conference on learning representaions ICLR, 2014.
clear all
close all
clc

vae_root = '/home/lgsanchez/work/Code/research/VAE_matlab/';
data_path = strcat(vae_root, 'data/');
%% Gather data from MNIST images
% This is a data set of handwritten digits 0 to 9
data_filename  = strcat(data_path, 'mnist_all.mat');
data = load(data_filename);

digits = [data.train0;...
     data.train1;...
     data.train2;...
     data.train3;...
     data.train4;...
     data.train5;...
     data.train6;...
     data.train7;...
     data.train8;...
     data.train9]; 
 
X = double(digits')/255;



%% load basic stuff for neural nets
run init_nnet.m

%% create encoder and decoder networks
code_dim = 3;
in_dim = size(X, 1);
hid_dim = 100;
binary = true;
run VAE_mnist_net.m
mnistAE.encoder = encoder;
mnistAE.decoder = decoder;

%% And Now the algorithm finally
trainparams.batchsize = 100;
trainparams.n_epochs = 100;
trainparams.stepsize = 0.001;
trainparams.n_monte = 1; % number of montecarlo samples 

mnistAE = trainVariationalAutoencoder(mnistAE, X, trainparams);

PX = naiveMarginalLikelihood(X(:,ceil(size(X, 2)*rand(1,1000))), mnistAE.decoder, 10);
