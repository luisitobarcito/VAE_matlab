function [mu, Sigma] = vae_Mstep(w, Y, YY, YTY, varargin)
% MSTEP_VAE Compute MLEs for variational autoencdoer
% function [mu, Sigma] = Mstep_cond_gauss(w, Y, YY, YTY, varargin)
%
% We assume P(Y|Q=i) = N(Y; mu_i, Sigma_i)
% and w(i,t) = p(Q(t)=i|y(t)) = posterior responsibility
% See www.ai.mit.edu/~murphyk/Papers/learncg.pdf.
%
% INPUTS:
% w(i) = sum_t w(i,t) = responsibilities for each mixture component
%  If there is only one mixture component (i.e., Q does not exist),
%  then w(i) = N = nsamples,  and 
%  all references to i can be replaced by 1.
%
% If covariance is tied, Sigma has size d*d.
% But diagonal and spherical covariances are represented in full size.

[cov_type, tied_cov,  clamped_cov, clamped_mean, cov_prior, other] = ...
    process_options(varargin,...
		    'cov_type', 'full', 'tied_cov', 0,  'clamped_cov', [], 'clamped_mean', [], ...
		    'cov_prior', []);

[Ysz Q] = size(Y);
N = sum(w);
if isempty(cov_prior)
  %cov_prior = zeros(Ysz, Ysz, Q);
  %for q=1:Q
  %  cov_prior(:,:,q) = 0.01*cov(Y(:,q)');
  %end
  cov_prior = repmat(0.01*eye(Ysz,Ysz), [1 1 Q]);
end
%YY = reshape(YY, [Ysz Ysz Q]) + cov_prior; % regularize the scatter matrix
YY = reshape(YY, [Ysz Ysz Q]);

% Set any zero weights to one before dividing
% This is valid because w(i)=0 => Y(:,i)=0, etc
w = w + (w==0);
		    
if ~isempty(clamped_mean)
  mu = clamped_mean;
else
  % eqn 6
  %mu = Y ./ repmat(w(:)', [Ysz 1]);% Y may have a funny size
  mu = zeros(Ysz, Q);
  for i=1:Q
    mu(:,i) = Y(:,i) / w(i);
  end
end

if ~isempty(clamped_cov)
  Sigma = clamped_cov;
  return;
end

if ~tied_cov
  Sigma = zeros(Ysz,Ysz,Q);
  for i=1:Q
    if cov_type(1) == 's'
      % eqn 17
      s2 = (1/Ysz)*( (YTY(i)/w(i)) - mu(:,i)'*mu(:,i) );
      Sigma(:,:,i) = s2 * eye(Ysz);
    else
      % eqn 12
      SS = YY(:,:,i)/w(i)  - mu(:,i)*mu(:,i)';
      if cov_type(1)=='d'
	SS = diag(diag(SS));
      end
      Sigma(:,:,i) = SS;
    end
  end
else % tied cov
  if cov_type(1) == 's'
    % eqn 19
    s2 = (1/(N*Ysz))*(sum(YTY,2) + sum(diag(mu'*mu) .* w));
    Sigma = s2*eye(Ysz);
  else
    SS = zeros(Ysz, Ysz);
    % eqn 15
    for i=1:Q % probably could vectorize this...
      SS = SS + YY(:,:,i)/N - mu(:,i)*mu(:,i)';
    end
    if cov_type(1) == 'd'
      Sigma = diag(diag(SS));
    else
      Sigma = SS;
    end
  end
end

if tied_cov
  Sigma =  repmat(Sigma, [1 1 Q]);
end
Sigma = Sigma + cov_prior;
