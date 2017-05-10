function [OSEQ] = loadseq(FILENAME, CDTYPE)
% LOADSEQ(FILENAME, CDTYPE) - loads character array from FILENAME 
% and transforms it into activation sequence according to the CDTYPE
% 
% FILENAME - file containing only characters
% CDTYPE   - 'OHE'  - one dimension for every symbol, values 1 or 0.
%          - 'REAL' - every symbol is assigned value from [0,1]

% try open file
[fid, msg] = fopen(FILENAME, 'r');
if fid == -1,
   error(sprintf('Error opening file %s (%s)',FILENAME, msg));
end;

% load it into character array
seq = fscanf(fid, '%s');
fclose(fid);

% check length
slen = length(seq);
if slen < 1,
   error(sprintf('No symbols in file %s', FILENAME));
end;

% get symbols and index sequence
[sym, dmy, indseq] = unique(seq);

% if no code encoding type, use one hot encoding
if nargin < 2, CDTYPE = 'OHE'; end;

% create activation codes
if strcmp(CDTYPE, 'REAL'), codes = linspace(0,1,length(sym)); end;
if strcmp(CDTYPE, 'OHE'), codes = eye(length(sym)); end;
if ~exist('codes'), error(sprintf('Unknown encoding type (%s)',CDTYPE)); end;
    
% transform input sequence into activation sequence
OSEQ = codes(:,indseq);

