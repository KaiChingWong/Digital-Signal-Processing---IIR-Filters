"""
Digital Signal Processing 4

Assignment 3: IIR Filters

By Kai Ching Wong (GUID:2143747W)
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from IIR_Filter_Class import IIR2filter as IIR2filter

#Read WAV file from computer
audio_fs,audio_data = wavfile.read('Samuel_Recording.wav')
noise_fs,noise_data = wavfile.read('Construction_Noise.wav')
signal_fs,signal_data = wavfile.read('Samuel_Clean.wav')

#Normalising
def normalize(data):
    data = data/1000
    dataout = 2*(data-min(data))/(max(data)-min(data))
    dataout = (dataout-1)
    dataout = dataout- np.mean(dataout)
    return dataout

#Finding the length of time for the audio sample
time = len(audio_data)/audio_fs 
t = np.linspace(0,time,len(audio_data))

time_n = len(noise_data)/noise_fs
noise_t = np.linspace(0,time_n,len(noise_data))

time_signal = len(signal_data)/signal_fs
signal_t = np.linspace(0,time_signal,len(signal_data))

#Butterworth Filters
def butter_lowpass_filter (data,fc,fs):
    
    #order of filter
    n = 2 
    
    #cutoff frequency
    fnyq = 0.5*fs
    fn = fc/fnyq

    #Coefficients 
    b,a = signal.butter(n,fn,btype='lowpass', analog=False)
    
    #Frequency response 
    w,h = signal.freqz(b,a)
    h1 = 20*np.log10(np.abs(h))
    wc = w/np.pi/2
    
    # Plotting Frequency 
    plt.figure(1)
    plt.plot(wc,h1,'g')
    plt.xscale('log')
    plt.title('Butterworth Lowpass Filter')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Amplitude (dB)')
    plt.margins(0,0.1)
    plt.grid(which='both', axis='both')
    plt.show()
    
    # Calling the filter class
    f1 = IIR2filter(a,b)
    
    # Array of Zeros
    lpout = np.zeros(len(data),dtype = 'int16')
    
    for i in range (len(data)):
       
        lpout[i] = f1.filter(data[i])
    
    lpout = np.real(lpout)

    return lpout

def butter_highpass_filter (data,fc,fs):
    n=2
    
    fnyq=0.5*fs
    fn= fc/fnyq

    b,a=signal.butter(n,fn,btype='highpass', analog=False)
    
    w,h=signal.freqz(b,a)
    h1=20*np.log10(np.abs(h))
    wc=w/np.pi/2
    
    plt.figure(1)
    plt.plot(wc,h1,'g')
    plt.xscale('log')
    plt.title('Butterworth Highpass Filter')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Amplitude (dB)')
    plt.margins(0,0.1)
    plt.grid(which='both', axis='both')
    plt.show()
    
    f2=IIR2filter(a,b)
     
    hpout=np.zeros(len(data),dtype = 'int16')
    
    for i in range (len(data)):
       
        hpout[i]=f2.filter(data[i])
    
    hpout=np.real(hpout)
    
    return hpout

def butter_bandpass_filter (data,flc,fhc, fs):
    n=1
    
    fnyq=0.5*fs
    fn1= flc/fnyq
    fn2= fhc/fnyq
 
    b,a=signal.butter(n,[fn1,fn2],btype='bandpass', analog=False)
    
    w,h=signal.freqz(b,a)
    h1=20*np.log10(np.abs(h))
    wc=w/np.pi/2
    
    plt.figure(1)
    plt.plot(wc,h1,'g')
    plt.xscale('log')
    plt.title('Butterworth Bandpass Filter')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Amplitude (dB)')
    plt.margins(0,0.1)
    plt.grid(which='both', axis='both')
    plt.show()
    
    f3=IIR2filter(a,b)
    
    bpout=np.zeros(len(data),dtype = 'int16')
    
    for i in range (len(data)):
       
        bpout[i]=f3.filter(data[i])
    
    bpout=np.real(bpout)
    
    return bpout

def butter_bandstop_filter (data,flc,fhc, fs):
    n=1

    fnyq=0.5*fs
    fn1= flc/fnyq
    fn2= fhc/fnyq

    b,a=signal.butter(n,[fn1,fn2],btype='bandstop', analog=False)
    
    w,h=signal.freqz(b,a)
    h1=20*np.log10(np.abs(h))
    wc=w/np.pi/2
    
    plt.figure(1)
    plt.plot(wc,h1,'g')
    plt.xscale('log')
    plt.title('Butterworth Bandstop Filter')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Amplitude (dB)')
    plt.margins(0,0.1)
    plt.grid(which='both', axis='both')
    plt.show()
    
    f4=IIR2filter(a,b)
    
    bsout=np.zeros(len(data),dtype = 'int16')
    
    for i in range (len(data)):
       
        bsout[i]=f4.filter(data[i])

    bsout=np.real(bsout)
        
    return bsout

output = butter_lowpass_filter(audio_data,1800,audio_fs)
output = butter_highpass_filter(output,120,audio_fs)
output = butter_bandstop_filter(output,50,120,audio_fs)
output = butter_bandstop_filter(output,180,250,audio_fs)
output = butter_bandstop_filter(output,310,400,audio_fs)
output = butter_bandstop_filter(output,4000,5000,audio_fs)
step = 3
while step > 0:
    output = butter_highpass_filter(output,120,audio_fs)
    output = butter_bandstop_filter(output,50,120,audio_fs)
    output = butter_bandstop_filter(output,180,250,audio_fs)
    output = butter_bandstop_filter(output,310,400,audio_fs)
    output = butter_bandstop_filter(output,4000,5000,audio_fs)
    step = step -1

output=normalize(output)
xf_audio_after = np.fft.fft(output)
wavfile.write('Filtered_Samuel_Recording.wav',audio_fs,output)

audio_data=normalize(audio_data)
xf = np.fft.fft(audio_data) #Fourier Transform the audio sample to frequency in Hz
f = np.linspace(0,audio_fs,len(audio_data))

noise_data=normalize(noise_data)
xf_n = np.fft.fft(noise_data)
noise_f = np.linspace(0,noise_fs,len(noise_data))

signal_data = normalize(signal_data)
xf_signal = np.fft.fft(signal_data)
signal_f = np.linspace(0,signal_fs,len(signal_data))

#Ideal Filter Response
IFR =  np.ones(44100)
IFR[50:120+1] = 0.0000000001
IFR[44100-120:44100-50+1] = 0.0000000001
IFR[180:250+1] = 0.0000000001
IFR[44100-250:44100-180+1] = 0.0000000001
IFR[310:400+1] = 0.0000000001
IFR[44100-400:44100-310+1] = 0.0000000001
IFR[4000:5000+1] = 0.0000000001
IFR[44100-5000:44100-4000+1] = 0.0000000001
IFR = IFR*2000

plt.figure(2)
plt.plot(signal_f,abs(xf_signal),linewidth=0.5)
plt.title('Signal Frequency Spectrum')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')
plt.xscale('log')
plt.savefig('Signal Frequency Spectrum.svg')

plt.figure(3)
plt.plot(f,abs(xf_audio_after),linewidth=0.5)
plt.title('Audio Frequency Spectrum')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')
plt.xscale('log')
plt.savefig('Audio Frequency Spectrum.svg')

plt.figure(4)
plt.plot(noise_f,abs(xf_n),linewidth=0.5)
plt.title('Noise Frequency Spectrum')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')
plt.xscale('log')
plt.savefig('Noise Frequency Spectrum.svg')

plt.figure(5)
plt.plot(20*np.log10(abs(IFR/2000)))
plt.title('Ideal Filter Response')
plt.ylabel('Amplitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.xscale('log')
plt.savefig('Ideal Filter Response.svg')

plt.figure(6,figsize=(10,10))
plt.subplot(211)
plt.plot(t,audio_data,linewidth=0.5)
plt.ylabel('Amplitude')
plt.xlabel('time (s)')
plt.title('Audio Before Filtering')
plt.subplot(212)
plt.plot(t,output,linewidth=0.5)
plt.ylabel('Amplitude')
plt.xlabel('time (s)')
plt.title('Audio After Filtering')
plt.savefig('Audio Before and After Filtering.svg')
