function [NN] = updateParamters(NN, stepsize, method, varargin)
implemented_methods = {'sgd', 'sga', 'sgd_momentum', 'sga_momentum'};

if ~exist('method', 'var')
    method = 'sgd';
else
    assert(any(strcmp(method, implemented_methods)), 'Update method not implemented');
end

if regexp(method, 'momentum')
    assert(length(varargin) > 0, 'Momentum must be provided');
    momentum = varargin{1};
end

switch(method)
    case 'sgd'
        for iLyr = 1 : length(NN)
            NN(iLyr).delta_W = -stepsize*NN(iLyr).grad_W;
            NN(iLyr).delta_b = -stepsize*NN(iLyr).grad_b;
            NN(iLyr).W = NN(iLyr).W + NN(iLyr).delta_W;
            NN(iLyr).b = NN(iLyr).b + NN(iLyr).delta_b;
        end
        
    case 'sga'
        for iLyr = 1 : length(NN)
            NN(iLyr).delta_W = stepsize*NN(iLyr).grad_W;
            NN(iLyr).delta_b = stepsize*NN(iLyr).grad_b;
            NN(iLyr).W = NN(iLyr).W + NN(iLyr).delta_W;
            NN(iLyr).b = NN(iLyr).b + NN(iLyr).delta_b;
        end

    case 'sgd_momentum'
        for iLyr = 1 : length(NN)
            NN(iLyr).delta_W = momentum*NN(iLyr).delta_W - stepsize*NN(iLyr).grad_W;
            NN(iLyr).delta_b = momentum*NN(iLyr).delta_b - stepsize*NN(iLyr).grad_b;
            NN(iLyr).W = NN(iLyr).W + NN(iLyr).delta_W;
            NN(iLyr).b = NN(iLyr).b + NN(iLyr).delta_b;
        end
        
    case 'sga_momentum'
        for iLyr = 1 : length(NN)
            NN(iLyr).delta_W = momentum*NN(iLyr).delta_W + stepsize*NN(iLyr).grad_W;
            NN(iLyr).delta_b = NN(iLyr).delta_b + stepsize*NN(iLyr).grad_b;
            NN(iLyr).W = NN(iLyr).W + NN(iLyr).delta_W;
            NN(iLyr).b = NN(iLyr).b + NN(iLyr).delta_b;
        end

        
    otherwise
        error('How did we even get here?');
end

end