function [neto, AO, ACT] = rnn_train_bp(net, IP,  TP, isteps)
% RNN_TRAIN_BP - train rnn using standard BP algorithm
% [neto, AO, ACT] = rnn_train_bp(net, IP,  TP, isteps)
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
firstStep = net.bp.saveDelay+1;
lastStep  = net.bp.saveDelay+pattNum;
isteps = net.bp.saveDelay+isteps;

% prepare activities (threshod + all input, then initial hidden and output)
ACT = zeros(net.numAllUnits, lastStep);
ACTD = zeros(net.numAllUnits, lastStep);
ACT(1,:) = 1;
ACT(2:net.numInputUnits+1,firstStep:lastStep) = IP;
DE_DNA = zeros(net.numAllUnits);

% copy BP epoch persistent params (also recompute derivatives)
ACT(:, 1:net.bp.saveDelay) = net.bp.saveAct;
ACTD(:, 1:net.bp.saveDelay) = net.bp.saveAct .* (1-net.bp.saveAct);
alpha = net.bp.alpha;
beta = net.bp.beta;
DLT_W = net.bp.DLT_W;

% copy params (Matlab 13 Acceleration)
% add ending destination to unused value -1
numWeights = net.numWeights;
numOutputUnits = net.numOutputUnits;
numInputUnits = net.numInputUnits;
numAllUnits = net.numAllUnits;
indexOutputUnits = net.indexOutputUnits;

weightsDest   = [net.weights.dest]; weightsDest(end+1) = -1;
weightsSource = [net.weights.source];
weightsDelay  = [net.weights.delay];
weightsValue  = [net.weights.value];

% Main loop 
for SI=(firstStep:lastStep),
    % Display progress
    if mod(SI,10000) == 0, disp(sprintf('Processing step %d', SI)); end;
    
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
        
        DE_DNA(:) = 0;
        res = TP(:,SI-firstStep+1) - ACT(indexOutputUnits,SI);
        res(~find(isfinite(res))) = 0;
        DE_DNA(indexOutputUnits, 1) = res;
        DLT_W = beta * DLT_W;
        
        % initial settings
        nextdest = weightsDest(numWeights);
        WI = numWeights;
        while WI > 0,
            % next derivative and initial destinantion node
            dest=nextdest;
            
            DE_DNA(dest) = DE_DNA(dest) * ACTD(dest, SI);
            der = DE_DNA(dest);
            
            while dest==nextdest,
                % error derivative calculation
                delay = weightsDelay(WI)+1;
                source = weightsSource(WI);
                if (delay == 1) && (source > numInputUnits+1),
                    DE_DNA(source) = DE_DNA(source) + weightsValue(WI) * der;
                end;
                
                % weight change calculation
                DLT_W(WI) = DLT_W(WI) + der * ACT(source, SI-delay+1);
                
                % get next destination node
                WI = WI-1;
                if WI==0, break; end;
                nextdest = weightsDest(WI);
            end;
        end;         
        
        % Ith step weight changing
        weightsValue = weightsValue + alpha * DLT_W;
    end;
end;

% select output activities
AO = ACT(indexOutputUnits, firstStep:lastStep);
for J=(1:numWeights), net.weights(J).value = weightsValue(J); end;

% save BP epoch persistent params
net.bp.saveAct = ACT(:, end-net.bp.saveDelay+1:end);
net.bp.DLT_W = DLT_W;

neto = net;
