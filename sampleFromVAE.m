function [X_samples] = sampleFromVAE(model, N)

code_dim = size(model.decoder.hidden.layers(1).W, 2);
%out_dim = size(model.decoder.mu.layers(end).W, 1);
    
Z = randn(code_dim, N);

model.decoder.hidden.layers = propagateForward(model.decoder.hidden.layers, Z);
model.decoder.mu.layers = propagateForward(model.decoder.mu.layers, model.decoder.hidden.layers(end).X_out);
mu = model.decoder.mu.layers(end).X_out;
switch model.decoder.type
    case 'bernouli'
       
%         X_samples = rand(size(mu)) <= mu;
        X_samples = mu;
    case 'gaussian'
        model.decoder.sigma.layers = propagateForward(model.decoder.sigma.layers, model.decoder.hidden.layers(end).X_out);
        sigma = sqrt(exp(model.decoder.sigma.layers(end).X_out));
        X_samples = bsxfun(@plus, bsxfun(@times, randn(size(mu)), sigma), mu);
    otherwise
        error('Not implemented, but you can contribute and add it');

end