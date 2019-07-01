import numpy as np
from scipy.io import wavfile


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=1):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    return tau, cc

#splitChannel4('./output.wav')
sig1 = np.array([0, 0, 0, 1, 2, 3, 0, 0])
sig2 = np.array([0, 1, 2, 3, 0, 0, 0, 0])


sampleRate, data1 = wavfile.read("mic1.wav")
sampleRate, data2 = wavfile.read("mic2.wav")
sampleRate, data3 = wavfile.read("mic3.wav")
sampleRate, data4 = wavfile.read("mic4.wav")


#tau, cc = gcc_phat(sig1, sig2)
tau, cc = gcc_phat(data3, data4)
print(tau)
print(cc)