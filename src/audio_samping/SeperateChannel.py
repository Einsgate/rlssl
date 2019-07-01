# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:16:02 2019

@author: neo
"""

import numpy as np
from scipy.io import wavfile


def splitChannel(srcFile):
    
    sampleRate,musicData = wavfile.read(srcFile)
    
    left = []
    right  =[]
    for item in musicData:
        left.append(item[0])
        right.append(item[1])
    print(np.array(left).dtype)
    wavfile.write('./left.wav',sampleRate, np.array(left))
    wavfile.write('./right.wav',sampleRate,np.array(right))
    
    
def splitChannel4(srcFile):
    srcName = srcFile.split('.')[0];
    
    sampleRate,musicData = wavfile.read(srcFile)
    
    mic1 = []
    mic2  =[]
    mic3 = []
    mic4 = []
    for item in musicData:
        mic1.append(item[0])
        mic2.append(item[1])
        mic3.append(item[2])
        mic4.append(item[3])

    wavfile.write(srcName + "_1.wav", sampleRate, np.array(mic1))
    wavfile.write(srcName + "_2.wav", sampleRate, np.array(mic2))
    wavfile.write(srcName + "_3.wav", sampleRate, np.array(mic3))
    wavfile.write(srcName + "_4.wav", sampleRate, np.array(mic4))
    
    
def mixtwoMusic(File1,File2):
    
    sampleRate1, music1 = wavfile.read(File1)
    sampleRate2, music2 = wavfile.read(File2)
    
    print("1: %d"%(sampleRate1))
    print("2: %d"%(sampleRate2))
    
    left = []
    right = []
    
    for item in music1:
        left.append(item[0])
    for item in music2:
        right.append(item[1])
    print(len(left),len(right))
    if len(left)>len(right):
        left=left[0:len(right)]
    else:
        right=right[0:len(left)]
    l1 = np.array(left)
    r1 = np.array(right)
    print(len(r1),len(l1),l1.dtype,r1.dtype)
    output = np.array([l1,r1],dtype=np.int16)
    wavfile.write('./mixed.wav',sampleRate1,output.T)


splitChannel4("output.wav")