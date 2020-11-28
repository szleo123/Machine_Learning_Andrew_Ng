function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %
    
    theta0Old = theta(1); theta1Old = theta(2);
    for k = 1:m 
        dJ0(k) = (1/m)*([theta0Old,theta1Old]*X(k,:)'-y(k))*X(k,1);
        dJ1(k) = (1/m)*([theta0Old,theta1Old]*X(k,:)'-y(k))*X(k,2);
        theta0New = theta0Old - alpha*sum(dJ0);
        theta1New = theta1Old - alpha*sum(dJ1);
    end 
    theta = [theta0New;theta1New];




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
