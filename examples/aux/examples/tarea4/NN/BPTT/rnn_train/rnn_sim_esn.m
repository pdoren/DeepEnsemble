function [AO, ACT] = rnn_sim_esn(net, IP, TP)
% RNN_SIM_ESN - simulates ESN
% [AO, ACT] = rnn_esn_sim(net, IP)
% AO  - activities of output units (delay acts. are removed)
% ACT - activities of all units (inp. hidn. outp.), incl. init. acts.
% net - ESN network
% IP  - input sequence
% TP  - desired sequence


% get size of input sequence
[sizeInputPatt, numInputPatt] = size(IP);
if sizeInputPatt ~= net.numInputUnits; error ('Number of input units and input patterns do not match.'); end;

% get size of output sequence
[sizeOutputPatt, numOutputPatt] = size(TP);
if sizeOutputPatt ~= net.numOutputUnits; error ('Number of output units and output patterns do not match.'); end;

% calculate starting and stopping step
firstStep = net.maxDelay+1;
lastStep  = net.maxDelay+numInputPatt;

% get unit counts
AUC = net.numAllUnits;
IUC = net.numInputUnits;
OUC = net.numOutputUnits;

% copy ESN epoch persistent params
C = net.esn.C;
A = net.esn.A;
N = net.esn.N;

% prepare activities (threshod + all input, then initial hidden and output)
ACT = zeros(net.numAllUnits, lastStep);
ACT(1:AUC, 1:net.maxDelay) = net.actInit;
ACT(1,:) = 1;
ACT(2:net.numInputUnits+1,firstStep:lastStep) = IP;
TP = [net.actInit(net.numAllUnits-net.numOutputUnits+1:net.numAllUnits,:) ,TP];

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
            source = weightsSource(WI);
            if (source <= AUC-OUC) || (SI-firstStep+1 > numOutputPatt),
                act = act + weightsValue(WI) .* ACT(weightsSource(WI), SI-weightsDelay(WI));
            else
                act = act + weightsValue(WI) .*  TP(weightsSource(WI)-AUC+OUC, SI-weightsDelay(WI));
            end;
            
            % get next destination node
            WI = WI+1;
            nextdest = weightsDest(WI);
        end;

        % calculate activity
        if (C==1) || (A==1), ACT(dest, SI) = tanh(act);  
        else ACT(dest, SI) = (1-C*A)*ACT(dest, SI-1) + C*tanh(act); 
        end;
            
    end;
    
    ACT(IUC+1:AUC, SI) = ACT(IUC+1:AUC, SI) + (rand(AUC-IUC, 1)-0.5) .* N;
    
end;    

% select output activities
AO = ACT(net.indexOutputUnits, firstStep:lastStep);
