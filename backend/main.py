#playing around with librosa

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.signal
import sklearn as sk #later for kmeans clustering
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw() # hide the root window

def extract_features(path=None):
    path = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=[("Audio files", "*.mp3 *.wav *.flac *.aiff *.ogg")],
    )

    if path is None:
        print("No file selected.")
        return
    print(f"Loading file: {path}")
    #load audio file    
    y, sr = librosa.load(path, sr=None)


    #deltas help find rapid changes
    cent_diff = np.diff(librosa.feature.spectral_centroid(y=y, sr=sr)[0])     # zero flattens array, which makes it easy to plot + analyze
    rms_diff = np.diff(librosa.feature.rms(y=y)[0]) 
    zcr_diff = np.diff(librosa.feature.zero_crossing_rate(y)[0])
    onset_diff = np.diff(librosa.onset.onset_strength(y=y, sr=sr))
    chroma_diff = np.diff(librosa.feature.chroma_stft(y=y, sr=sr)[0]) #lol c.diff
    #times for plotting
    times = librosa.frames_to_time(np.arange(len(rms_diff)), sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)

    return times, tempo, beat_times, rms_diff, cent_diff, zcr_diff, onset_diff, chroma_diff



def detect_cues (rms_diff, cent_diff, zcr_diff, onset_diff, chroma_diff, times, beat_times, tempo):
    #TODO: implement feature matrix

    scaler = sk.preprocessing.MinMaxScaler()
    rms_diff_z = scaler.fit_transform(rms_diff.reshape(-1, 1)).flatten()
    cent_diff_z = scaler.fit_transform(cent_diff.reshape(-1, 1)).flatten() 
    zcr_diff_z = scaler.fit_transform(zcr_diff.reshape(-1, 1)).flatten()
    onset_diff_z = scaler.fit_transform(onset_diff.reshape(-1, 1)).flatten()
    chroma_diff_z = scaler.fit_transform(chroma_diff.reshape(-1, 1)).flatten()

    #smooth all features with a gaussian filter
    rms_diff_z = scipy.ndimage.gaussian_filter1d(rms_diff_z, sigma=2)
    cent_diff_z = scipy.ndimage.gaussian_filter1d(cent_diff_z, sigma=2)
    zcr_diff_z = scipy.ndimage.gaussian_filter1d(zcr_diff_z, sigma=2)
    onset_diff_z = scipy.ndimage.gaussian_filter1d(onset_diff_z, sigma=2)
    chroma_diff_z = scipy.ndimage.gaussian_filter1d(chroma_diff_z, sigma=2)
    
    #TODO: try using kmeans clustering to identify sections of high activity instead of peak finding, or in conjunction with peak finding
    #weighted fusion of features
    #weights determined by trial and error
    score = (0.4 * rms_diff_z + 
                0.13 * cent_diff_z + 
                0.1 * zcr_diff_z + 
                0.25 * onset_diff_z +
                0.15 * chroma_diff_z)
    
    #boost onset
    score = score * (1 + onset_diff_z)

    #peak finding

    #only consider peaks above 90th percentile of score within a local window of 20 seconds, 
    threshold = np.percentile(score, 90)
    #window of 20 seconds
    distance = int(20 * (len(rms_diff) / times[-1])) #number of frames in 20 seconds
    #promience of at least 0.1
    peaks, _ = scipy.signal.find_peaks(score, height=threshold, distance=distance, prominence=0.1)
    cue_times = times[peaks]

    #quanzalize to nearest beat
    beats_to_sec = 60/tempo

    #min spacing of 16 beats
    min_beats = 16
    min_interval = min_beats * beats_to_sec


    #quantize cue times to nearest beat, ensuring minimum spacing
    cue_times_quantized = []
    last_cue_time = -np.inf
    for t in cue_times:
        if t - last_cue_time < min_interval:
            continue
        #find nearest beat
        nearest_beat = beat_times[np.argmin(np.abs(beat_times - t))]
        if nearest_beat - last_cue_time >= min_interval:
            cue_times_quantized.append(nearest_beat)
            last_cue_time = nearest_beat
    
    #convert to m:ss format
    cue_times_quantized = [f"{int(t // 60)}:{int(t % 60):02}" for t in cue_times_quantized]
    return cue_times_quantized
        





print("Welcome to the audio feature extractor!")
times, tempo, beat_times, rms_diff, cent_diff, zcr_diff, onset_diff, chroma_diff = extract_features()
cue_times = detect_cues(rms_diff, cent_diff, zcr_diff, onset_diff, chroma_diff, times, beat_times, tempo)
print("Expected cue points", cue_times) 