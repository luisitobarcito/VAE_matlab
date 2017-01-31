function [AE] = trainVariationalAutoencoder(AE, X, trainparams)

if ~exist('trainparams', 'var')
    trainparams = struct();
end

% defaults to load in case are not provided in trainparams
default.n_epochs = 100;
default.stepsize = 0.001;
default.n_monte = 1;
default.batchsize = 100;

trainparams = loadDefaults(trainparams, default);

%% And Now the algorithm finally
n_epochs = trainparams.n_epochs;
stepsize = trainparams.stepsize;
n_monte = trainparams.n_monte;
batchsize = trainparams.batchsize;

encoder = AE.encoder;
decoder = AE.decoder;
binary  = strcmp(decoder.type, 'bernouli');

in_dim = size(encoder.hidden.layers(1).W, 2);
code_dim = size(encoder.mu.layers(end).W, 1); 



[X_batches] = createMiniBatches(X, batchsize);
for iEpc = 1:n_epochs
    fprintf('Epoch  %d\n',iEpc);
    Lbound = zeros(size(X_batches, 3), 1);
    KL_term = zeros(size(X_batches, 3), 1);
    logPXZ_term = zeros(size(X_batches, 3), 1);
    for iBtch = 1 :size(X_batches, 3)
        %% Sample from the variational distribution Q(Z|X) (encoder)
        encoder.hidden.layers = propagateForward(encoder.hidden.layers, X_batches(:,:, iBtch));
        encoder.mu.layers = propagateForward(encoder.mu.layers, encoder.hidden.layers(end).X_out);
        encoder.sigma.layers = propagateForward(encoder.sigma.layers, encoder.hidden.layers(end).X_out);
        mu_enc = encoder.mu.layers(end).X_out;
        % the encoder function maps X to log(sigma^2) = encoder.sigma.layers(end).X_out
        sigma2_enc = exp(encoder.sigma.layers(end).X_out);
        sigma_enc = sqrt(sigma2_enc);
        % generate samples from auxiliary variable epsilon N(0,1)
        EPS_enc = randn(code_dim, batchsize, n_monte);
        Z = bsxfun(@plus, bsxfun(@times, EPS_enc, sigma_enc), mu_enc);
        
        %% Compute the variational bound for current batch
        KL_term(iBtch) = mean(-(1/2)*sum(1 + encoder.sigma.layers(end).X_out - mu_enc.^2 - sigma2_enc, 1), 2);
        decoder.hidden.layers = propagateForward(decoder.hidden.layers, reshape(Z, [code_dim, batchsize*n_monte]));
        if binary == true
            decoder.mu.layers = propagateForward(decoder.mu.layers, decoder.hidden.layers(end).X_out);
            mu_dec = reshape(decoder.mu.layers(end).X_out, [in_dim, batchsize, n_monte]);
            E_dec = bsxfun(@minus, mu_dec, X_batches(:,:, iBtch));
            
            logPXZ_term(iBtch) = mean(mean(sum(bsxfun(@times, log(mu_dec), X_batches(:,:, iBtch)) ...
                + bsxfun(@times, log(1-mu_dec), 1 - X_batches(:,:, iBtch)), 1), 3), 2);
            
            diff_logPXZ_mu_dec = (1/batchsize)*(1/n_monte)*(-E_dec./(mu_dec.*(1-mu_dec)));
            decoder.mu.layers = propagateBackward(decoder.mu.layers, reshape(diff_logPXZ_mu_dec, [in_dim, batchsize*n_monte]));
            diff_logPXZ_h_dec = decoder.mu.layers(1).Diff_in;
            
        else
            decoder.mu.layers = propagateForward(decoder.mu.layers, decoder.hidden.layers(end).X_out);
            decoder.sigma.layers = propagateForward(decoder.sigma.layers, decoder.hidden.layers(end).X_out);
            mu_dec = reshape(decoder.mu.layers(end).X_out, [in_dim, batchsize, n_monte]);
            % the decoder function maps X to log(sigma^2) = encoder.sigma.layers(end).X_out
            sigma2_dec = reshape(exp(decoder.sigma.layers(end).X_out), [in_dim, batchsize, n_monte]);
            sigma_dec = sqrt(sigma2_dec);
            
            E_dec = bsxfun(@minus, mu_dec, X_batches(:,:, iBtch));
            
            logPXZ_term(iBtch) = mean(-(1/2)*log(2*pi)*in_dim ...
                -(1/2)*mean(sum(reshape(decoder.sigma.layers(end).X_out, [in_dim, batchsize, n_monte]), 1)...
                + sum((E_dec./sigma_dec).^2, 1), 3), 2);
            
            diff_logPXZ_mu_dec = (1/batchsize)*(1/n_monte)*(-E_dec./sigma2_dec);
            diff_logPXZ_logsigma_dec = (1/batchsize)*(1/n_monte)*(1/2)*((E_dec./sigma_dec).^2 - 1);
            decoder.mu.layers = propagateBackward(decoder.mu.layers, reshape(diff_logPXZ_mu_dec, [in_dim, batchsize*n_monte]));
            decoder.sigma.layers = propagateBackward(decoder.sigma.layers, reshape(diff_logPXZ_logsigma_dec, [in_dim, batchsize*n_monte]));
            diff_logPXZ_h_dec = decoder.mu.layers(1).Diff_in + decoder.sigma.layers(1).Diff_in;
            
        end
        
        Lbound(iBtch) = -KL_term(iBtch) + logPXZ_term(iBtch);
        diff_KL_mu_enc = (1/batchsize)*mu_enc;
        diff_KL_logsigma_enc = (1/batchsize)*(1/2)*(sigma2_enc - 1);
        
        % backpropagate the gradients
        decoder.hidden.layers = propagateBackward(decoder.hidden.layers, diff_logPXZ_h_dec);
        diff_logPXZ_Z = reshape(decoder.hidden.layers(1).Diff_in, [code_dim, batchsize, n_monte]);
        diff_Lbound_mu_enc = sum(diff_logPXZ_Z, 3) - diff_KL_mu_enc;
        diff_Lbound_logsigma_enc = (1/2)*sum(bsxfun(@times, diff_logPXZ_Z.*EPS_enc, sigma_enc), 3) - diff_KL_logsigma_enc;
        encoder.mu.layers = propagateBackward(encoder.mu.layers, diff_Lbound_mu_enc);
        encoder.sigma.layers = propagateBackward(encoder.sigma.layers, diff_Lbound_logsigma_enc);
        diff_Lbound_h_end = encoder.mu.layers(1).Diff_in + encoder.sigma.layers(1).Diff_in;
        encoder.hidden.layers = propagateBackward(encoder.hidden.layers, diff_Lbound_h_end);
        
        % compute updates
        decoder.mu.layers = updateParamters(decoder.mu.layers, stepsize, 'sga');
        if binary == false
            decoder.sigma.layers = updateParamters(decoder.sigma.layers, stepsize, 'sga');
        end
        decoder.hidden.layers = updateParamters(decoder.hidden.layers, stepsize, 'sga');
        
        encoder.mu.layers = updateParamters(encoder.mu.layers, stepsize, 'sga');
        encoder.sigma.layers = updateParamters(encoder.sigma.layers, stepsize, 'sga');
        encoder.hidden.layers = updateParamters(encoder.hidden.layers, stepsize, 'sga');
        
    end
    fprintf('Variational bound is %f,\nKL %f, logPXZ %f\n', mean(Lbound), mean(KL_term), mean(logPXZ_term));
    
    
    
end
AE.encoder = encoder;
AE.decoder = decoder;

end

function [params] = loadDefaults(params, defaults)

for iFld = fields(defaults)
   if ~isfield(params, iFld)
       params.(iFld) = defaults.(iFld);
   end
end

end