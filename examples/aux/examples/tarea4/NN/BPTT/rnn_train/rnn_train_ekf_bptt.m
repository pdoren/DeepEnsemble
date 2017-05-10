function [neto, AO, ACT] = rnn_train_ekf_bptt(net, IP,  TP, isteps)
% RNN_TRAIN_EKF_BPTT - train rnn using EKF-BPTT algorithm
% [neto, AO, ACT] = rnn_train_ekf_bptt(net, IP,  TP, isteps)
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
if isteps > inpNum, error ('Number of initial steps is too large.'); end;

pattNum = inpNum;   

% calculate starting and stopping step
firstStep = net.ekf_bptt.saveDelay+1;
lastStep  = net.ekf_bptt.saveDelay+pattNum;
isteps = net.ekf_bptt.saveDelay+isteps;

% prepare activities (threshod + all input, then initial hidden and output)
ACT = zeros(net.numAllUnits, lastStep);
ACTD = zeros(net.numAllUnits, lastStep);
ACT(1,:) = 1;
ACT(2:net.numInputUnits+1,firstStep:lastStep) = IP;

% derivatives of activities with respect of input, forward and recurrent weights in time T and T-1
DE_DNA = zeros(net.numOutputUnits, net.numAllUnits, net.ekf_bptt.wsize);
DAO_DW = zeros(net.numOutputUnits, net.numWeights);

% copy EKF-BPTT epoch persistent params (also recompute derivatives)
ACT(:, 1:net.ekf_bptt.saveDelay) = net.ekf_bptt.saveAct;
ACTD(:, 1:net.ekf_bptt.saveDelay) = net.ekf_bptt.saveAct .* (1-net.ekf_bptt.saveAct);
wsize = net.ekf_bptt.wsize;
DLT_W = net.ekf_bptt.DLT_W';

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

% EKF entities
K = zeros(numWeights, numOutputUnits);
H = zeros(numOutputUnits, numWeights);

lambda = net.ekf_bptt.lambda;
P = net.ekf_bptt.P;
R = net.ekf_bptt.R;
Q = net.ekf_bptt.Q;

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
        
        DE_DNA(:) = 0;
        DAO_DW(:) = 0;
        DE_DNA(:, indexOutputUnits, 1) = eye(numOutputUnits);
        
        for J=(1:wsize),

            % test of window index J
            if SI < J, break; end;
            
            % initial settings
            nextdest = weightsDest(numWeights);
            WI = numWeights;
            while WI > 0,
                % next derivative and initial destinantion node
                dest=nextdest;
                
                for K=(1:numOutputUnits),
                    DE_DNA(K, dest, J) = DE_DNA(K, dest, J) * ACTD(dest, SI-J+1);
                end;
                
                der = DE_DNA(:, dest, J);
                
                while dest==nextdest,
                    % error derivative calculation
                    delay = weightsDelay(WI)+J;
                    source = weightsSource(WI);
                    if (delay < wsize+1) && (source > numInputUnits+1),
                        for K=(1:numOutputUnits),
                            DE_DNA(K, source, delay) = DE_DNA(K, source, delay) + weightsValue(WI) * der(K);
                        end;
                    end;
                                      
                    % weight change calculation
                    for K=(1:numOutputUnits),
                        DAO_DW(K, WI) = DAO_DW(K, WI) + der(K) * ACT(source, SI-delay+1); 
                    end;
                    
                    % get next destination node
                    WI = WI-1;
                    if WI==0, break; end;
                    nextdest = weightsDest(WI);
                end;
            end;         
        end;
        
        % Ith step weight change computation step TAR_T - target pattern (skip NaN and Infs = no teacher sig.)
        res = TP(:,SI-firstStep+1) - ACT(indexOutputUnits,SI);
        res(~find(isfinite(res))) = 0;

        % Ith step Kalman update
        H = DAO_DW;
            
        K = P*H' * inv(H*P*H' + R);
        DLT_W = K * res;
        P = (P - K*(H*P) + Q) ./ lambda;
        
        % Ith step weight changing
        weightsValue = weightsValue + DLT_W';
    end;
end;

% select output activities
AO = ACT(indexOutputUnits, firstStep:lastStep);
for J=(1:numWeights), net.weights(J).value = weightsValue(J); end;

% save EKF-BPTT epoch persistent params
net.ekf_bptt.saveAct = ACT(:, end-net.ekf_bptt.saveDelay+1:end);
net.ekf_bptt.DLT_W = DLT_W';
net.ekf_bptt.P = P;

neto = net;
