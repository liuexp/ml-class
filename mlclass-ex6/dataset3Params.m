function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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
C=0.64
sigma=0.08

%pred=svmPredict(svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)),Xval);
%err=mean(double(pred ~= yval));
%
%for p=1:15
%	curC=0.01*(2.^p)
%	for q=1:10
%		curS=0.01*(2.^q)
%		pred=svmPredict(svmTrain(X, y, curC, @(x1, x2) gaussianKernel(x1, x2, curS)),Xval);
%		curerr=mean(double(pred ~= yval));
%		if curerr <= err
%			C=curC;
%			sigma=curS;
%			err=curerr;
%		end;
%	end;
%end;
%printf('%.5f %.5f\n',C,sigma);








% =========================================================================

end
