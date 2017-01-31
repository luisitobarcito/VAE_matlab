function [NN] = propagateBackward(NN, Diff)
NN(end).Diff_out = Diff;
for iLyr = length(NN) : -1 : 1
    if iLyr < length(NN)
        NN(iLyr).Diff_out = NN(iLyr + 1).Diff_in;
    end
    NN(iLyr).Diff_in = (NN(iLyr).W'*(NN(iLyr).Diff_out.*NN(iLyr).f_prime(NN(iLyr).Z)));
    NN(iLyr).grad_W = (NN(iLyr).Diff_out.*NN(iLyr).f_prime(NN(iLyr).Z))*NN(iLyr).X_in';
    NN(iLyr).grad_b = sum(NN(iLyr).Diff_out.*NN(iLyr).f_prime(NN(iLyr).Z), 2);
end

end