function [AO, ACT] = rnn_sim(net, IP)
% RNN_SIM - simulates rnn
% [AO, ACT] = rnn_sim(net, IP)
% AO  - activities of output units (delay acts. are removed)
% ACT - activities of all units (inp. hidn. outp.), incl. init. acts.
% net - RNN network
% IP  - input sequence

% get size of input sequence
[pattSize, pattNum] = size(IP);
if pattSize ~= net.numInputUnits; error ('Number of input units and input patterns do not match.'); end;

% calculate starting and stopping step
firstStep = net.maxDelay+1;
lastStep  = net.maxDelay+pattNum;

% prepare activities (threshod + all input, then initial hidden and output)
ACT = zeros(net.numAllUnits, lastStep);
ACT(1,:) = 1;
ACT(2:net.numInputUnits+1,firstStep:lastStep) = IP;
ACT(net.numInputUnits+2:net.numAllUnits, 1:net.maxDelay) = net.actInit;

% copy params (Matlab 13 Acceleration)
% add ending destination to unused value -1
numWeights = net.numWeights;
weightsDest   = [net.weights.dest]; weightsDest(end+1) = -1;
weightsSource = [net.weights.source];
weightsDelay  = [net.weights.delay];
weightsValue  = [net.weights.value];

% forward computation
for SI=(firstStep:lastStep),
    % initial settings
    nextdest = weightsDest(1);
    WI = 1;
    while WI<numWeights,
        % next activity and initial destinantion node  
        act = 0;
        dest=nextdest;
        while dest==nextdest,
            % calculation
            act = act + weightsValue(WI) * ACT(weightsSource(WI), SI-weightsDelay(WI)); 
            % get next destination node
            WI = WI+1;
            nextdest = weightsDest(WI);
        end;
        % calculate activity
        ACT(dest, SI) = 1 ./ (1+exp(-act));
    end;
end;    

% select output activities
AO = ACT(net.indexOutputUnits, firstStep:lastStep);
