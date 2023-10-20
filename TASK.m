% Number of Monte Carlo samples
N = 1000000;

    
%tolerance = 0.03;  % Adjust as needed
% Initialize the counter for non-zero results
count_non_zero = 0;

% Specify the ranges for a, b, c, x, and y
 % i changed the notation for my ease :(
 % a,b,c are the phases and x,y are the angles 
a_min = 0;           
a_max = 2 * pi;
b_min = 0;
b_max = 2 * pi;
c_min = 0;
c_max = 2 * pi;
x_min = 0;
x_max = pi/4;
y_min = 0;
y_max = pi/4;

for i = 1:N
    % Generate random values for a, b, c, x, and y
    a = a_min + (a_max - a_min) * rand();   
    b = b_min + (b_max - b_min) * rand();
    c = c_min + (c_max - c_min) * rand();
    x = x_min + (x_max - x_min) * rand();
    y = y_min + (y_max - y_min) * rand();
    
    % Evaluate the expression
    result = a * (cos(x) - 1) + b * (1 - cos(y)) + c * (-cos(x) + cos(y));
    
    % Check if the expression is not equal to zero
    if abs(result) ~= 0
        count_non_zero = count_non_zero + 1;
    end
end

% Calculate the probability
probability_non_zero = count_non_zero / N;
disp(['Estimated probability that the expression is above than the threshold value: ', num2str(probability_non_zero)]);


% -----------------------------------------------

% the output of this code is 1. which is good.
% now if we set some threshold the probablity will decrease.
% i took a value of 0.01 as a threshold and the probablity(that the expression will not be equal to zero) decreased to 0.9637.