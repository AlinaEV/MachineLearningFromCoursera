function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

y_matrix = eye(num_labels)(y,:) ; %[5000x10] Expand the 'y' output values into a matrix of single values
a1 = [ones(m,1) X]; %[5000x401]

%printf("y_matrixsize=%d %d\n",size(y_matrix));
%printf("ysize=%d %d\n",size(y));

%z2 = a1 * Theta1'; %[5000x25]
%a2 = sigmoid(z2);
%a2 = [ones(m,1) a2];  %[5000x26]
%z3 = a2 * Theta2';  % [5000x10]
%a3 = sigmoid(z3);
%h = a3;
%J = 1/m * sum((- y_matrix' * log(h) - (1 - y_matrix)' * log(1 - h)),m);

for i = 1:m
  z2 = Theta1 * a1(i,:)'; %[25x1]
  a2 = sigmoid(z2);
  a2 = [1 a2'];  %[1x26]
  z3 = Theta2 * a2';
  h = sigmoid(z3);  %[10x1]
  J = J + sum(- y_matrix(i,:) * log(h) - (1 - y_matrix(i,:)) * log(1 - h), num_labels);
endfor

reg = lambda / (2 * m) * (sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2)));
J = (1/m) * J + reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
Delta1 = zeros(hidden_layer_size, input_layer_size + 1);
Delta2 = zeros(num_labels, hidden_layer_size + 1);
%1.
z2 = Theta1 * a1'; %[h x m]
a2 = sigmoid(z2);
a2 = [ones(m,1) a2'];  %[m x(h+1)]
z3 = Theta2 * a2';
a3 = sigmoid(z3);  %[rxm]

%2. Layer3
delta3 = zeros(num_labels, m);  %[r x m]
delta3 = a3 - y_matrix';

%3. Layer2
delta2 = (Theta2(:,2:end)' * delta3) .* sigmoidGradient(z2); %[25x10]*[10x1].*[25x1]->[hxm]

%4. Accumulate the gradient
Delta1 = Delta1 + delta2 * a1; %(h x m)(m x n) --> (h x n)
Delta2 = Delta2 + delta3 * a2; %(r x m)(m x [h+1]) --> (r x [h+1])

%5. Gradients
Theta1_grad = 1/m * Delta1;
Theta2_grad = 1/m * Delta2;

##for i = 1:m
##  %1.
##  z2 = Theta1 * a1(i,:)'; %[25x1]->[h x m]
##  a2 = sigmoid(z2);
##  a2 = [1 a2'];  %[1x26]->[m x(h+1)]
##  z3 = Theta2 * a2';
##  a3 = sigmoid(z3);  %[10x1]
##  
##  %2. Layer3
##  delta3 = zeros(num_labels, 1);  %[r x m]
##  for k = 1:num_labels
##    delta3(k) = a3(k) - (y_matrix(i,k) == k);
##  endfor
##  
##  %3. Layer2
##  delta2 = (Theta2(:,2:end)' * delta3) .* sigmoidGradient(z2); %[25x10]*[10x1].*[25x1]->[hxm]
##  %delta2 = delta2(2:end);
##  
##  %4. Accumulate the gradient
##  Delta1 = Delta1 + delta2 * a1(i,:); %(h x m)(m x n) --> (h x n)
##  Delta2 = Delta2 + delta3 * a2; %(r x m)(m x [h+1]) --> (r x [h+1])
##  
##  %5. Gradients
##  Theta1_grad(i) = 1/m * Delta1;
##  Theta2_grad(i) = 1/m * Delta2;
##  
##endfor

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad = Theta1_grad + [zeros(hidden_layer_size,1) lambda/m * Theta1(:,2:end)];
Theta2_grad = Theta2_grad + [zeros(num_labels,1) lambda/m * Theta2(:,2:end)];

% -------------------------------------------------------------
%a1: 5000x401
%z2: 5000x25
%a2: 5000x26
%a3: 5000x10
%d3: 5000x10
%d2: 5000x25
%Theta1, Delta1 and Theta1_grad: 25x401
%Theta2, Delta2 and Theta2_grad: 10x26
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
