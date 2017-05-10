function [neto, AO, ACT] = rnn_train_rtrl(net, IP,  TP, isteps)
% RNN_TRAIN_RTRL - train rnn using RTRL algorithm
% [neto, AO, ACT] = rnn_train_rtrl(net, IP,  TP, isteps)
% neto   - trained network
% AO     - activities of output units (delay acts. are removed)
% ACT    - activities of all units (inp. hidn. outp.), incl. init. acts.
% net    - input RNN network
% IP     - input sequence
% TP     - target = desired output sequence
% isteps - initial = starting steps

% parameter checking
[inpSize, inpNum] = size(IP);
[tarSize, tarNum] = size(TP);
if inpSize ~= net.numInputUnits, error ('Number of input units and input pattern size do not match.'); end;
if tarSize ~= net.numOutputUnits, error ('Number of output units and target pattern size do not match.'); end;
if inpNum ~= tarNum, error ('Number of input and output patterns is different.'); end;
if isteps > inpNum,  error ('Number of initial steps is too large.'); end;

pattNum = inpNum;   

% calculate starting and stopping step
firstStep = net.maxDelay+1;
lastStep  = net.maxDelay+pattNum;
isteps = net.maxDelay+isteps;

% prepare activities (threshod + all input, then initial hidden and output)
ACT = zeros(net.numAllUnits, lastStep);
ACTD = zeros(net.numAllUnits, lastStep);
ACT(1,:) = 1;
ACT(2:net.numInputUnits+1,firstStep:lastStep) = IP;

% copy RTRL epoch persistent params
ACT(net.numInputUnits+2:net.numAllUnits, 1:net.maxDelay) = net.rtrl.actInit;
alpha = net.rtrl.alpha;
beta = net.rtrl.beta;
DLT_W  = net.rtrl.DLT_W;
DA_DW = net.rtrl.DA_DW;

% copy params (Matlab 13 Acceleration)
% add ending destination to unused value -1
numWeights = net.numWeights;
numOutputUnits = net.numOutputUnits;
numAllUnits = net.numAllUnits;
indexOutputUnits = net.indexOutputUnits;

weightsDest   = [net.weights.dest]; weightsDest(end+1) = -1;
weightsSource = [net.weights.source];
weightsDelay  = [net.weights.delay];
weightsValue  = [net.weights.value];

% Main loop 
for SI=(firstStep:lastStep),
    % Display progress
    if mod(SI,1000) == 0, disp(sprintf('Processing step %d', SI)); end;
    
    % *********************************************************************
    % forward flow
    % *********************************************************************
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
        act = 1 ./ (1+exp(-act));
        ACT(dest, SI) = act; 
        ACTD(dest, SI) = act * (1-act);
    end;

    % *********************************************************************
    % training after initial steps
    % *********************************************************************
    if SI > isteps,
        for J=(1:numWeights),
            % destination unit of weight J
            destj = weightsDest(J);
            sourcej = weightsSource(J);
            delayj = weightsDelay(J);
            
            % initial settings
            nextdest = weightsDest(1);
            WI = 1;
            while WI<numWeights,
                % next derivative and initial destinantion node
                der = 0;
                dest=nextdest;
                while dest==nextdest,
                    % calculation
                    der = der + weightsValue(WI) * DA_DW(weightsSource(WI), J, weightsDelay(WI)+1);
                    % get next destination node
                    WI = WI+1;
                    nextdest = weightsDest(WI);
                end;
                % calculate derivative
                if destj == dest, DA_DW(dest, J, 1) = ACTD(dest, SI) * (der + ACT(sourcej, SI-delayj));
                else DA_DW(dest, J, 1) = ACTD(dest, SI) * der;
                end;
            end;         
        end;
        
        % Ith step weight change computation step TAR_T - target pattern (skip NaN and Infs = no teacher sig.)
        res = TP(:,SI-firstStep+1) - ACT(indexOutputUnits,SI);
        res(~find(isfinite(res))) = 0;
        DLT_W = res' * DA_DW(indexOutputUnits,:,1) + beta * DLT_W;

        % Ith step weight changing
        DA_DW(:,:,2:end) = DA_DW(:,:,1:end-1); 
        weightsValue = weightsValue + alpha * DLT_W;

    end;
end;

% select output activities
AO = ACT(indexOutputUnits, firstStep:lastStep);
for J=(1:numWeights), net.weights(J).value = weightsValue(J); end;

% save RTRL epoch persistent params
net.rtrl.actInit = ACT(net.numInputUnits+2:net.numAllUnits, end-net.maxDelay+1:end);
net.rtrl.DLT_W = DLT_W;
net.rtrl.DA_DW = DA_DW;

neto = net;
