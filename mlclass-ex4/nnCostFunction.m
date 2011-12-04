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
%printf('%d %d',size(X));
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



X1=sigmoid(Theta1*[ones(size(X,1),1) X]');
X2=sigmoid(Theta2*[ones(1,size(X1,2));X1]);
%printf('%d %d\n',size(X2));
%printf('%d %d\n',size(y));
yy=zeros(size(X2));
for i= 1:m
	yy(y(i),i)=1;
end

J=sum(sum( -yy.* log(X2) - (1-yy).* log(1-X2) ))/m + lambda*(sum(sum(Theta2.*Theta2)) +sum(sum( Theta1.*Theta1)) - sum(Theta2(:,1).*Theta2(:,1))- sum (Theta1(:,1).*Theta1(:,1)) )/(2*m) ;



delta3=X2-yy; %size 10 x 5000
%printf('delta 3: %d %d\n',size(delta3));
%printf('Theta2: %d %d\n',size(Theta2));

%% input layer 400+1
%% hidden layer 25+1
%% output layer 10

delta2=(Theta2' * delta3)(2:end,:) .* (X1.*(1-X1)); %size 25 x 5000
%printf('%d %d\n',size(delta2));

Theta1_grad=((delta2*[ones(size(X,1),1),X]))/m; %size 401 x 25'
Theta2_grad=((delta3*[ones(size(X1,2),1),X1']))/m; %size 26 x 10'
[a,b]=size(Theta2_grad);

for i=1:a
	for j=2:b
		Theta2_grad(i,j)+=Theta2(i,j)*lambda/m;
	end
end

[a,b]=size(Theta1_grad);
for i=1:a
	for j=2:b
		Theta1_grad(i,j)+=Theta1(i,j)*lambda/m;
	end
end



%printf('Theta2_grad %d %d\n',size(Theta2_grad));
%printf('Theta1_grad %d %d\n',size(Theta1_grad));






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
