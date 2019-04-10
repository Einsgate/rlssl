%--------------------------------------------------------------------------
%
% Data Generator
%
% Author        : Junjie Wang
% Date          : Feb 25, 2019
%
% Related paper :
%
% Comment       :
%
%--------------------------------------------------------------------------

close all;
%clear;
set(0,'DefaultAxesFontSize',12)

%% Settings
training_data = 0;               % 1 -> Traning data, 0 -> Test data
if training_data == 1
    angles = 0:10:180;           % Source azimuth angles 
    dis = [2.5, 2, 1.5];         % Distance between source and the mic center
else
    angles = 0:30:180;           % Source azimuth angles 
    dis = [2.3, 1.8];            % Distance between source and the mic center
end
c = 340;                         % Sound velocity in meters / second
LL = [10 10 3];                    % Room dimensions in meters (width, depth, hight)
beta = 0.2;                      % Reverberation time in seconds
nsample = 1024;                  % RIR length in samples
order = 2;                       % Reflection order
rp = [3-0.1 3 1.5; 3+0.1 3 1.5]; % Receiver positions in meter
cp = (rp(1,:) + rp(end,:))/2;
M = size(rp,1);                  % Number of microphones
% hop = 32;                      % Refresh rate of the AIR
% sp = [0.5 3 1.5];              % Initial source position
% type_of_movement = 'line';     % Source movement 'arc' or 'line'

%% Load anechoic sound source
[in, fs] = audioread('female_speech.wav');
if size(in,1) > size(in,2)
    in = in';
end
in = [in(fs+1:2*fs)];
len = length(in);

data = zeros(length(angles), length(dis), M, len);

id = 0;
for dd = dis
    id = id+1;
    ia = 0;
    for aa = angles
        ia = ia+1;
        %% Generate source path
        sp_path = zeros(len, 3);
        rp_path = zeros(len, 3, M);
        
        % Convert (theta, phi, rd) to (x, y, z)
        [xx yy zz] = sph2cart(aa*pi/180, 0, dd); 
        sp = cp + [xx yy zz];
        
        % Stationary source positions
        sp_path(1:end, 1) = sp(1);
        sp_path(1:end, 2) = sp(2);
        sp_path(1:end, 3) = sp(3);
        
        % Stationary receiver positions
        for mm = 1:M
            rp_path(1:end, 1, mm) = rp(mm, 1);
            rp_path(1:end, 2, mm) = rp(mm, 2);
            rp_path(1:end, 3, mm) = rp(mm, 3);    
        end
        
        %% Generate microphone signal
        [out,beta_hat] = signal_generator(in,c,fs,rp_path,sp_path,LL,beta,nsample,'o',order);
    
        %% Store the signal in data
        data(ia, id, :, :) = out;
        
%         for mm = 1:M
%             filename = "data_" + aa + "_" + dd + "_" + mm + ".txt"
%             fid = fopen(filename, 'w');
%             fprintf(fid, '%f\n', data(i, j, mm, :));
%             fclose(fid);
%         end
       
        
%         %% Plot source path and signals
%         figure(1);
%         plot3(rp(1,1),rp(1,2),rp(1,3),'x');
%         hold on;
%         for mm = 2:M
%             plot3(rp(mm,1),rp(mm,2),rp(mm,3),'x');
%         end
%         plot3(sp_path(:,1),sp_path(:,2),sp_path(:,3),'r.');
%         axis([0 LL(1) 0 LL(2) 0 LL(3)]);
%         grid on;
%         box on;
%         axis square;
%         hold off;
% 
%         figure(2)
%         t = 0:1/fs:(length(in)-1)/fs;
%         subplot(211); plot(t,in); title('in(n)');xlabel('Time [Seconds]');ylabel('Amplitude');
%         subplot(212); plot(t,out'); title('out(n)');xlabel('Time [Seconds]');ylabel('Amplitude');
%
%         %% Play generated microphone signal
%          soundsc(out'./max(max(abs(out))),fs);
    end
end

if training_data == 1
    save('training_data', 'data', 'dis', 'angles', 'fs');
else
    save('test_data', 'data', 'dis', 'angles', 'fs');
end