%% Settings
training_data = 0;      % 1 -> training data, 0 -> test data
frame_len_t = 0.1;      % frame length in time domain
gcc_width = 21;         % gcc vector length
width = 10;             % half of gcc_width

%% Load data
if training_data == 1
    load('training_data.mat');
else
    load('test_data.mat');
end
M = size(data, 3);  % Number of microphones
npairs = nchoosek(M, 2); % Pairs of microphones
nangles = length(angles);
ndis = length(dis);
len = size(data, 4);     % Length for each signal
frame_len = fs*frame_len_t;
nframes = floor((len-1)/frame_len) + 1;

%% Preprocess
% Open the label file
if training_data == 1
    flabel = fopen("training_labels.txt", 'w');
else
    flabel = fopen("test_labels.txt", 'w');
end

ia = 0;
for aa = angles
    ia = ia + 1;
    id = 0;
    for dd = dis
        id = id + 1;
        pp = 0;
        
        % Initialize gcc vector
        gcc = zeros(npairs, gcc_width);
        for i = 1:M
            for ii = (i+1):M
                pp = pp + 1; % The pp th pair of microphones
                i_f = 0;
                % For each frame
                for ff = 1:(frame_len*fs):len
                    i_f = i_f + 1;
                    sig1 = reshape(data(ia, id, i, ff:min(ff+frame_len-1,len)), [], 1);
                    sig2 = reshape(data(ia, id, ii, ff:min(ff+frame_len-1,len)), [], 1);
                   
                    [tau,R,lag] = gccphat(sig1, sig2, fs);
                    
                    % Find center
                    for loc = 1:length(R)
                        if lag(loc) == 0
                            break;
                        end
                    end
                    
                    % Sum up gcc of each frame
                    gcc(pp, :) = R(loc-width:loc+width);
                end
                % Get the average gcc value of all frames
                gcc(pp, :) = gcc(pp, :) / nframes;
            end
        end 
        
        
        % Save gcc vcectors
        count = (ia-1)*length(dis) + id;
        if training_data == 1
            filename = "training_" + count + ".txt";
        else
            filename = "test_" + count + ".txt";
        end
        fid = fopen(filename, 'w');
        fprintf(fid, '%f\n', reshape(gcc, [], 1));
        fclose(fid);
        
        % Save labels
        fprintf(flabel, '%d %f\n', aa, dd);
    end
end

% Close the label file
fclose(flabel);