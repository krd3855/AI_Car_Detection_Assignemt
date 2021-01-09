%% Author : Krishna Satish D S
function out = sigmoid(Weight)
    out = 1./(1+exp(-(Weight)));
end