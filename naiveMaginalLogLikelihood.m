function [PX] = naiveMarginalLikelihood(X, decoder, n_samples, p_z)

if ~exist('n_samples', 'var')
    n_samples = 100;
end


if ~exist('p_z', 'var')
    p_z = 'normal';
end

[code_dim, ~] = size(decoder.hidden.layers(1).X_in);
[x_dim, N] = size(X);

switch p_z
    case 'normal'
        Z = randn(code_dim, N, n_samples);
    otherwise
        error('Prior not implemented :(')
end

decoder.hidden.layers = propagateForward(decoder.hidden.layers, reshape(Z, [code_dim, N*n_samples]));
decoder.mu.layers = propagateForward(decoder.mu.layers, decoder.hidden.layers(end).X_out);
mu_dec = reshape(decoder.mu.layers(end).X_out, [x_dim, N, n_samples]);

switch decoder.type
    case 'bernouli'
        logPXZ = sum(bsxfun(@times, log(mu_dec), X) + bsxfun(@times, log(1-mu_dec), 1 - X), 1);
       
    
    case 'gaussian'
        decoder.sigma.layers = propagateForward(decoder.sigma.layers, decoder.hidden.layers(end).X_out);
        % the decoder function maps X to log(sigma^2) = encoder.sigma.layers(end).X_out
        sigma2_dec = exp(decoder.sigma.layers(end).X_out);
        sigma_dec = reshape(sqrt(sigma2_dec), [x_dim, N, n_samples]);
        E_dec = bsxfun(@minus, mu_dec, X);
        logPXZ = -(1/2)*log(2*pi)*x_dim -(1/2)*(sum(reshape(decoder.sigma.layers(end).X_out, [x_dim, N, n_samples]), 1)...
                                                   + sum((E_dec./sigma_dec).^2, 1));
    otherwise
        error('Unknown decoder type :(')

end
PXZ = exp(logPXZ);
PX = mean(PXZ, 3);
%logPX = mean(logPXZ, 3);
end