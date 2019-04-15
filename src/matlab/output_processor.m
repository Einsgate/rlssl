LL = [10 10 3];                   % Room dimensions in meters (width, depth, hight)
rp = [3-0.1 3 1.5; 3+0.1 3 1.5]; % Receiver positions in meter
sp = [6 6 1.5];                    % Speaker position in meter
M = size(rp,1);                   % Number of microphones
npairs = nchoosek(M, 2);          % Pairs of microphones

c = 340;                          % Sound velocity in meters / second
beta = 0.2;                       % Reverberation time in seconds
nsample = 1024;                   % RIR length in samples
order = 2;                        % Reflection order

maxDis = 0.2;
fs = 16000;
len = 16000;

f = fopen("output.dat", "r");
out = fscanf(f, "%f %f ", [2 len]);
fclose(f);

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

gcc = reshape(gcc, 1, [])
gcc = num2cell(gcc);
