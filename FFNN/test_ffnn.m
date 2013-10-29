function [out] = test_ffnn
num = 20;
dim = 5;
r = rand(dim, num);
x = [r-1 r+1];
% x
x = whiten(x);
% x
% 1/40 * x*x'

y = [zeros(1,num) ones(1,num)];
ffnn(x,y,8)

end

function [newx] = whiten(x)

newx = zeros(size(x,1), size(x,2));
mu = mean(x, 2);
num_examples = size(x,2);
for i=1:num_examples
	newx(:,i) = x(:,i) - mu;	
end

cv = 1/num_examples * x * x';
L = chol(cv);
newx = inv(L') * x;

end
