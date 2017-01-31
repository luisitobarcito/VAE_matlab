%%% this script creates the network for the variational auto encoder with
%%% Gaussian encoder for mnist

%%%%%%%%%%% Encoder network %%%%%%%%%%%%%%%%
%%% nonlinear hidden layer %%%%%%%%%%%%%%%%%%%%%%%%
encoder.hidden.layers(1) = layer;
encoder.hidden.layers(1).W = randn(hid_dim, in_dim)/sqrt(in_dim);
encoder.hidden.layers(1).b = zeros(hid_dim, 1);
encoder.hidden.layers(1).f = relu;
encoder.hidden.layers(1).f_prime = relu_prime;
encoder.hidden.layers(1).delta_W = zeros(hid_dim, in_dim);
encoder.hidden.layers(1).delta_b = zeros(hid_dim, 1);
encoder.hidden.layers(1).grad_W = zeros(hid_dim, in_dim);
encoder.hidden.layers(1).grad_b = zeros(hid_dim, 1);

%%% encoder layer for reparametrization location-scale family %%%%%
encoder.mu.layers(1) = layer;
encoder.mu.layers(1).W = randn(code_dim, hid_dim)/sqrt(hid_dim);
encoder.mu.layers(1).b = zeros(code_dim, 1);
encoder.mu.layers(1).f = idty;
encoder.mu.layers(1).f_prime = idty_prime;
encoder.mu.layers(1).delta_W = zeros(code_dim, hid_dim);
encoder.mu.layers(1).delta_b = zeros(code_dim, 1);
encoder.mu.layers(1).grad_W = zeros(code_dim, hid_dim);
encoder.mu.layers(1).grad_b = zeros(code_dim, 1);


encoder.sigma.layer(1) = layer;
encoder.sigma.layers(1).W = randn(code_dim, hid_dim)/sqrt(hid_dim);
encoder.sigma.layers(1).b = zeros(code_dim, 1);
encoder.sigma.layers(1).f = idty;
encoder.sigma.layers(1).f_prime = idty_prime;
encoder.sigma.layers(1).delta_W = zeros(code_dim, hid_dim);
encoder.sigma.layers(1).delta_b = zeros(code_dim, 1);
encoder.sigma.layers(1).grad_W = zeros(code_dim, hid_dim);
encoder.sigma.layers(1).grad_b = zeros(code_dim, 1);


%%%%%%%%%%%% Decoder Network %%%%%%%%%%%%%%%%
%%% Decoding hidden %%%%%%%%%%%%%%%%%%%%%%%%%%
decoder.hidden.layers(1) = layer;
decoder.hidden.layers(1).W = randn(hid_dim, code_dim)/sqrt(code_dim);
decoder.hidden.layers(1).b = zeros(hid_dim, 1);
decoder.hidden.layers(1).f = relu;
decoder.hidden.layers(1).f_prime = relu_prime;
decoder.hidden.layers(1).delta_W = zeros(hid_dim, code_dim);
decoder.hidden.layers(1).delta_b = zeros(hid_dim, 1);
decoder.hidden.layers(1).grad_W = zeros(hid_dim, code_dim);
decoder.hidden.layers(1).grad_b = zeros(hid_dim, 1);


%%% decoder layer for reparametrization location-scale family %%%%%
if binary == true
    decoder.mu.layers(1) = layer;
    decoder.mu.layers(1).W = randn(in_dim, hid_dim)/sqrt(hid_dim);
    decoder.mu.layers(1).b = zeros(in_dim, 1);
    decoder.mu.layers(1).f = logisitc;
    decoder.mu.layers(1).f_prime = logistic_prime;
    decoder.mu.layers(1).delta_W = zeros(in_dim, hid_dim);
    decoder.mu.layers(1).delta_b = zeros(in_dim, 1);
    decoder.mu.layers(1).grad_W = zeros(in_dim, hid_dim);
    decoder.mu.layers(1).grad_b = zeros(in_dim, 1);
    
    decoder.type = 'bernouli'; 

else
    
    decoder.mu.layers(1) = layer;
    decoder.mu.layers(1).W = randn(in_dim, hid_dim)/sqrt(hid_dim);
    decoder.mu.layers(1).b = zeros(in_dim, 1);
    decoder.mu.layers(1).f = idty;
    decoder.mu.layers(1).f_prime = idty_prime;
    decoder.mu.layers(1).delta_W = zeros(in_dim, hid_dim);
    decoder.mu.layers(1).delta_b = zeros(in_dim, 1);
    decoder.mu.layers(1).grad_W = zeros(in_dim, hid_dim);
    decoder.mu.layers(1).grad_b = zeros(in_dim, 1);
    
    
    decoder.sigma.layer(1) = layer;
    decoder.sigma.layers(1).W = randn(in_dim, hid_dim)/sqrt(hid_dim);
    decoder.sigma.layers(1).b = zeros(in_dim, 1);
    decoder.sigma.layers(1).f = idty;
    decoder.sigma.layers(1).f_prime = idty_prime;
    decoder.sigma.layers(1).delta_W = zeros(in_dim, hid_dim);
    decoder.sigma.layers(1).delta_b = zeros(in_dim, 1);
    decoder.sigma.layers(1).grad_W = zeros(in_dim, hid_dim);
    decoder.sigma.layers(1).grad_b = zeros(in_dim, 1);
    
    decoder.type = 'gaussian';
    
end