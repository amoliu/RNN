function [out] = ffnn(x, y, hidden_size)
% X is the input data
% Y are the class labels

x = [ones(1, size(x,2)); x];
x_dim = size(x,1);  
y_dim = size(y,1);
l1_size = x_dim;
l2_size = hidden_size+1;
l3_size = y_dim;
W1 = rand(l2_size-1, l1_size); 
W2 = rand(l3_size, l2_size);

num_examples = size(x,2);

for j=1:200

Del1 = zeros(l2_size-1, l1_size);
Del2 = zeros(l3_size, l2_size);

num_correct = 0;

for i=1:num_examples

a1 = x(:,i);
z2 = W1 * a1;
a2 = [1; sigmoid(z2)];
z3 = W2 * a2;
a3 = sigmoid(z3);

a3
if (a3 > .5) == y(:,i)
num_correct = num_correct + 1;
end

del3 = (a3 - y(:,i));
del2 = W2(:,2:end)' * del3 .* (a3 .* (1-a3));

Del1 = Del1 +del2 * a1';
Del2 = Del2 + del3 * a2';

end

W1
W2
Del1
Del2

num_correct
num_examples

W1 = W1 - .1*(1/num_examples) * Del1;
W2 = W2 - .1*(1/num_examples) * Del2;

end

end


function [out] = feedforward(W1, W2, in)
z2 = W1 * in;
a2 = sigmoid(z2);
z3 = W2 * a2;
a3 = sigmoid(z3);
out = a3;
end

function [out] = sigmoid(d)
  out = 1.0 ./ (1 + exp(-d)); 
end
