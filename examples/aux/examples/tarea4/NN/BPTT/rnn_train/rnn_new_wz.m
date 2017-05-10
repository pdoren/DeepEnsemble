function net = rnn_new_wz(IUC, HUC, OUC)
% RNN_NEW_WZ - setup network as Williams&Zipser fully connected recurrent network
% net = rnn_new_wz(IUC, HUC, OUC)
% net - new network structure
% IUC - number of input units
% HUC - number of hidden units
% OUC - number of output units

% set number of all units
AUC = 1+IUC+HUC+OUC;

% set numbers of units
net.numInputUnits    = IUC;
net.numOutputUnits   = OUC;
net.numAllUnits      = AUC;

% set neuron masks
net.maskInputUnits   = [0; ones(IUC, 1); zeros(AUC-1-IUC, 1)];
net.maskOutputUnits  = [zeros(AUC-OUC, 1); ones(OUC, 1)];
net.indexOutputUnits = find(net.maskOutputUnits);
net.indexInputUnits  = find(net.maskInputUnits);

% set input weights
weight = struct('dest',0,'source',0,'delay',0,'value',0,'const',false,'act',1,'wtype',1);

n=1;
% all neurons weights
for i=(IUC+2:AUC),
    % threshold and input weights
    for j=(1:IUC+1),
        net.weights(n) = weight;
        net.weights(n).dest   = i;
        net.weights(n).source = j;
        n = n+1;
    end;
    % recurrent weights
    for j=(IUC+2:AUC),
        net.weights(n) = weight;
        net.weights(n).dest   = i;
        net.weights(n).source = j;
        net.weights(n).delay  = 1;
        n = n+1;
    end;
end;

% output weights
for i=(AUC-OUC+1:AUC),
    for j=[IUC+2:AUC-OUC],
        net.weights(n) = weight;
        net.weights(n).dest   = i;
        net.weights(n).source = j;
        n = n+1;
    end;
end;

% set number of weights
net.numWeights = n-1;

% initialize weight matrices form [-0.25, 0.25]
W_INIT_RNG = 0.5;
for i=(1:net.numWeights),
    net.weights(i).value = rand .* 2 .* W_INIT_RNG - W_INIT_RNG;
end;

% initialize starting activities from [0, 1]
net.maxDelay = max([net.weights.delay]);

net.actInit = rand(AUC-IUC-1, net.maxDelay);

