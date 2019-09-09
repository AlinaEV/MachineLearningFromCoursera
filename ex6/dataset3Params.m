function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
mult_steps = [0.01 0.03 0.1 0.3 1 3 10 30]';
results = [];
for i = 1:length(mult_steps)
  for j = 1:length(mult_steps)
    model = svmTrain(X, y, mult_steps(i), @(x1, x2) gaussianKernel(x1, x2, mult_steps(j))); 
    predictions = svmPredict(model, Xval);
    err = mean(double(predictions ~= yval));
    results = [results; mult_steps(i) mult_steps(j) err];
  endfor
endfor

[err i] = min(results(:,3));
C = results(i,1);
sigma = results(i,2);
% =========================================================================

end
