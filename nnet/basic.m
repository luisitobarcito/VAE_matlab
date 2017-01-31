% Load basic neural net elements
% This script defines some basic elements that can be used across different
% applications of neural nets.
% This code is for research and didactic purposes and not been optimized
% for performance.


%%%%%% acivation functions %%%%%%%%%%%%

idty = inline('x');
idty_prime = inline('ones(size(x))');

logistic = inline('1./(1 + exp(-x))');
logistic_prime = inline('(1./(1 + exp(-x))).*(1./(1 + exp(x)))');

hyptan = inline('tanh(X)'); 
hyptan_prime = inline('1 - tanh(X).*tanh(X)');

relu = inline('max(0, x)');
relu_prime = inline('double(x > 0)');

softrelu = inline('log(1+exp(x))');
softrelu_prime = logistic;

satlin = inline('max(0, x)-max(0,x-1)','x');
satlin_prime = inline('max(0, x)>0 & max(0, x-1)==0','x');



%%%%% layer definition %%%%%%%%%%%%%%%%
% Layer is a struct composed of the following elements
% name : string with the name of the layer
% W : weight matrix
% b : bias vector
% f : activation function
% f_prime : derivative of the activation function
% delta_W : incrment for W updates
% delta_b : increment for b updates
% X_in : input data
% Z : intermediate data W*X_in + b 
% X_out : layer output f(Z)
% Diff_in : Backpropageted differenetial thorugh the layer with respect to
% the input
% Diff_out : differentials form higher level

layer = struct('name', [], 'W',[], 'b', [], 'f', [], 'f_prime', [], 'grad_W', [], 'grad_b', [], 'delta_W', [], 'delta_b', [], 'X_in', [], 'Z', [], 'X_out', [],'Diff_in', [], 'Diff_out', []);