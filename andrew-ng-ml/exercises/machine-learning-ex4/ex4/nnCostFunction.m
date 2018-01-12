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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%-------------------------------------------------------------------------------
% Adding the bias term to input layer
X_temp = [ones(m, 1) X];
A2 = sigmoid(X_temp*Theta1');
A2 = [ones(size(A2, 1),1) A2];  % Adding bias term in hidden layer
A3 = sigmoid(A2*Theta2');
% Converting given scalar output as vector multiclass output

my_y = zeros(size(y, 1), num_labels);
for i=1:size(y,1),
  my_y(i, y(i)) = 1;
end;

% The cost function
J = sum(sum(-my_y.*log(A3), 2) - sum((1-my_y).*log(1-A3), 2))/m;

% Calculating the reqularisation term
theta1 = Theta1(:, 2:end);
theta2 = Theta2(:, 2:end);
s1 = sum(sum(theta1.*theta1));
s2 = sum(sum(theta2.*theta2));

J = J + (s1 + s2)*lambda/(2*m);

% -------------------------------------------------------------

A1 = [ones(m, 1) X];
Z2 = A1 * Theta1';
A2 = [ones(size(Z2, 1),1) sigmoid(Z2)];
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);

delta3 = A3 - my_y;
delta2 = delta3 * Theta2 .* sigmoidGradient([ones(size(Z2, 1),1) Z2]);
delta_2 = delta2(:, 2:end);
  
Theta1_grad = Theta1_grad + delta_2' * A1;
Theta2_grad = Theta2_grad + delta3' * A2;

theta1 = [zeros(size(Theta1, 1), 1) theta1];
theta2 = [zeros(size(Theta2, 1), 1) theta2];

Theta1_grad = (Theta1_grad+lambda*theta1)/m;
Theta2_grad = (Theta2_grad+lambda*theta2)/m;

% =========================================================================



% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
