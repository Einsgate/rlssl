f = fopen("input.dat", "r");

fs = sscanf(fgets(f), "%d")
len = sscanf(fgets(f), "%d")
[in n] = fscanf(f, "%f", [1 len]);

fclose(f);

LL = [10 10 3];                   % Room dimensions in meters (width, depth, hight)
rp = [3-0.1 3 1.5; 3+0.1 3 1.5]; % Receiver positions in meter
sp = [6 6 1.5];                    % Speaker position in meter
M = size(rp,1);                   % Number of microphones
npairs = nchoosek(M, 2);          % Pairs of microphones

c = 340;                          % Sound velocity in meters / second
beta = 0.2;                       % Reverberation time in seconds
nsample = 1024;                   % RIR length in samples
order = 2;                        % Reflection order

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


[out,beta_hat] = signal_generator(in,c,fs,rp_path,sp_path,LL,beta,nsample,'o',order);

f = fopen("output.dat", "w");
fprintf(f, "%f %f\n", out);
fclose(f);

