function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


    h = X * theta;    % The hypothesis: X has size (m x n), and theta is (n x 1), so hypotesis is (m x 1)
    err = h - y;    % errors vector (m x 1)
    delta = (1/m) * (X' * err);    % The change in theta (the "gradient") vector (n x 1)
                                    % Since X is (m x n), and the error vector is (m x 1), and the result 
                                    % is the same size as theta (which is (n x 1), we transpose X 
                                    % before we can multiply it by the error vector.
                                    % The vector multiplication automatically includes calculating 
                                    % the sum of the products.
    theta = theta - alpha * delta;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end
fprintf('features=%d\n',size(X, 2));
end
