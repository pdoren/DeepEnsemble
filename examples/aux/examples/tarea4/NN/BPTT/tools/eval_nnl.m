function [NNL, NNLSEQ, PSEQ] = eval_nnl(OS, TS, MINP)
%
%

if nargin < 2,
   error('Function requires 2 input arguments');
end;

if size(OS) ~= size(TS),
   error('Dimensions of outpur and target sequences do not match');
end;

if size(TS,1) == 1,
   error('NNL evaluation requires at least 2 dimensional patterns');
end;

if size(TS,1) == 1,
   error('NNL evaluation requires at least 2 dimensional patterns');
end;

if any(sum(TS)~=1),
   error('NNL evaluation requires one hot encoding of target sequence');
end;

if nargin < 3,
   MINP = 0.001;
end;

PSEQ = OS(find(TS))' ./ sum(OS);
PSEQ(find(~isfinite(PSEQ))) = 0;

NNLSEQ = -log(max([PSEQ; repmat(MINP, 1, size(PSEQ,2))])) ./ log(size(TS,1));
NNL = sum(NNLSEQ) ./ size(PSEQ,2);


