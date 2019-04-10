%--------------------------------------------------------------------------
%
% GCC vector generator
%
% Author        : Junjie Wang
% Date          : Mar 15, 2019
%
% Related paper :
%
% Comment       :
% 
% Input:    micPos   - A list of micphone positions [(x1, y1, z1), (x2, y2, z2)...]
%           maxDis   - Maximal distance between all microphone pairs
%           room     - Room size (width, depth, height)
%           soundPos - Position of the sound source (x1, y1, z1)
% Output:   gcc      - GCC vector()
%
%--------------------------------------------------------------------------
function gcc = gccGenerator(micPos, maxDis, room, soundPos)

%% Initialize arguments
LL = room;                   % Room dimensions in meters (width, depth, hight)
%rp = [3-0.1 3 1.5; 3+0.1 3 1.5]; % Receiver positions in meter
rp = micPos;                      % Receiver positions in meter
sp = soundPos;                    % Speaker position in meter
M = size(rp,1);                   % Number of microphones
npairs = nchoosek(M, 2);          % Pairs of microphones

c = 340;                          % Sound velocity in meters / second
beta = 0.2;                       % Reverberation time in seconds
nsample = 1024;                   % RIR length in samples
order = 2;                        % Reflection order

%% Load anechoic sound source
[in, fs] = audioread('female_speech.wav');
if size(in,1) > size(in,2)
    in = in';
end
in = [in(fs+1:2*fs)];
len = length(in);

% Stationary source positions
sp_path = zeros(len, 3);
sp_path(1:end, 1) = sp(1);
sp_path(1:end, 2) = sp(2);
sp_path(1:end, 3) = sp(3);
% Stationary receiver positions
rp_path = zeros(len, 3, M);
for mm = 1:M
    rp_path(1:end, 1, mm) = rp(mm, 1);
    rp_path(1:end, 2, mm) = rp(mm, 2);
    rp_path(1:end, 3, mm) = rp(mm, 3);    
end

%% Generate micphone signals
%for i = 1:10
%tic
%for i = 1:10
    [out,beta_hat] = signal_generator(in,c,fs,rp_path,sp_path,LL,beta,nsample,'o',order);
%end
%toc
%end

%% Calculate GCC vectors
width = ceil(maxDis / c * fs);              % Half of gcc_width
gcc_width = width * 2 + 1;
frame_len_t = 0.1;                          % Frame length in time domain
frame_len = fs*frame_len_t;                 % Frame length in terms of samples
nframes = floor((len-1)/frame_len) + 1;

% Initialize gcc vector
gcc = zeros(npairs, gcc_width);

pp = 0;              % Pair index
for i = 1:M         
    for ii = (i+1):M % For each micphone pair, we calculate a GCC vector
        pp = pp + 1; % The pp th pair of microphones
        i_f = 0;
        % For each frame
        for ff = 1:(frame_len*fs):len
            i_f = i_f + 1;
            sig1 = reshape(out(i, ff:min(ff+frame_len-1,len)), [], 1);
            sig2 = reshape(out(ii, ff:min(ff+frame_len-1,len)), [], 1);

            [tau,R,lag] = gccphat(sig1, sig2, fs);

            % Find the center
            for loc = 1:length(R)
                if lag(loc) == 0
                    break;
                end
            end

            % Sum up gcc of each frame
            gcc(pp, :) = real(R(loc-width:loc+width));
        end
        % Get the average gcc value of all frames
        gcc(pp, :) = gcc(pp, :) / nframes;
    end
end 

gcc = num2cell(reshape(gcc, 1, []));
%size(gcc);
%res = {gcc_width, npairs};
%len = res{1} * npairs;
%res(end+1:end+len) = gcc;

end