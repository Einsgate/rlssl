[in, fs] = audioread("../../female_speech.wav");
if size(in,1) > size(in,2)
    in = in';
end
in = [in(fs+1:2*fs)];
len = length(in);

f = fopen("input.dat", "w");
fprintf(f, "%d\n", fs);
fprintf(f, "%d\n", len);
fprintf(f, "%f\n", in);
fclose(f);