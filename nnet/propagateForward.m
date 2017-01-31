function [NN] = propagateForward(NN, X_in)

NN(1).X_in = X_in;
for iLyr = 1 : length(NN)
    if iLyr > 1
        NN(iLyr).X_in = NN(iLyr - 1).X_out;
    end
    NN(iLyr).Z = bsxfun(@plus, NN.W * NN(iLyr).X_in, NN(iLyr).b);
    NN(iLyr).X_out = NN(iLyr).f(NN(iLyr).Z);
end

end