function [X_batches] = createMiniBatches(X, batchsize, seed)

if ~exist('seed', 'var')
    rng('shuffle');
else
    rng(seed);
end

if isstruct(X);
    assert(isfield(X,'data') && isfield(X,'targets'), 'When data and targets are given as struct, it should be named struct_name.data and struct_name.targets.')
    has_targets = true;
    [X_dims, N] = size(X.data);
    [L_dims, ~] = size(X.targets);

else
    [X_dims, N] = size(X);
    has_targets = false;
end
    


n_batches = floor(N/batchsize);

rnd_idx = randperm(N);
if has_targets
    X_batches.data = zeros(X_dims, batchsize, n_batches);
    X_batches.targets = zeros(L_dims, batchsize, n_batches);
else
    X_batches = zeros(X_dims, batchsize, n_batches);
end
    
for iBtch = 1:n_batches
    if has_targets
        X_batches.data(:,:,iBtch) = X.data(:, rnd_idx(((iBtch-1)*batchsize + 1):iBtch*batchsize));
        X_batches.targets(:,:,iBtch) = X.targets(:, rnd_idx(((iBtch-1)*batchsize + 1):iBtch*batchsize));
    else
        X_batches(:,:,iBtch) = X(:, rnd_idx(((iBtch-1)*batchsize + 1):iBtch*batchsize));
    end
end

