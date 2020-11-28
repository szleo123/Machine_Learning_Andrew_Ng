function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C0 = 0;
sigma0 = 0;

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

model0= svmTrain(X, y, C0, @(x1, x2) gaussianKernel(x1, x2, sigma0));
predict0 = svmPredict(model0,Xval);
error0 = mean(double(predict0 ~= yval));

for C=[0.01,0.03,0.1,0.3,1,3,10,30]
    for sigma = [0.01,0.03,0.1,0.3,1,3,10,30]
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        prediction = svmPredict(model,Xval);
        error = mean(double(prediction ~= yval));
        if error < error0 
            C0 = C;
            sigma0 = sigma;
            error0 = error;
        end 
    end 
end 



C = C0; 
sigma = sigma0;
% =========================================================================

end
