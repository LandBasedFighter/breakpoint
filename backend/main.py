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
    chroma_diff = np.diff(librosa.feature.chroma_cqt(y=y, sr=sr)[0]) #lol c.diff
    #times for plotting
    times = librosa.frames_to_time(np.arange(len(rms_diff)), sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)

    return times, tempo, beat_times, rms_diff, cent_diff, zcr_diff, onset_diff, chroma_diff, y, sr



def detect_cues (rms_diff, cent_diff, zcr_diff, onset_diff, chroma_diff, times, beat_times, tempo,y, sr):

    scaler = sk.preprocessing.MinMaxScaler()
    rms_diff_z = scaler.fit_transform(rms_diff.reshape(-1, 1)).flatten()
    cent_diff_z = scaler.fit_transform(cent_diff.reshape(-1, 1)).flatten() 
    zcr_diff_z = scaler.fit_transform(zcr_diff.reshape(-1, 1)).flatten()
    onset_diff_z = scaler.fit_transform(onset_diff.reshape(-1, 1)).flatten()
    chroma_diff_z = scaler.fit_transform(chroma_diff.reshape(-1, 1)).flatten() # chroma will also be used to calculate novelty 
    #smooth all features with a gaussian filter
    rms_diff_z = scipy.ndimage.gaussian_filter1d(rms_diff_z, sigma=2)
    cent_diff_z = scipy.ndimage.gaussian_filter1d(cent_diff_z, sigma=2)
    zcr_diff_z = scipy.ndimage.gaussian_filter1d(zcr_diff_z, sigma=2)
    onset_diff_z = scipy.ndimage.gaussian_filter1d(onset_diff_z, sigma=2)
    chroma_diff_z = scipy.ndimage.gaussian_filter1d(chroma_diff_z, sigma=2)

    #calculate novelty from chroma using ssm

    ssm = librosa.segment.recurrence_matrix(chroma_diff_z, mode='affinity')
    novelty = np.sqrt(np.sum(np.diff(ssm, axis=1)**2, axis=0))



    # make all features the same length, replace with extracting everything with the same hop length later
    min_len = min(len(rms_diff_z), len(cent_diff_z), len(zcr_diff_z), 
              len(onset_diff_z), len(chroma_diff_z), len(novelty))

    rms_diff_z = rms_diff_z[:min_len]
    cent_diff_z = cent_diff_z[:min_len]
    zcr_diff_z = zcr_diff_z[:min_len]
    onset_diff_z = onset_diff_z[:min_len]
    chroma_diff_z = chroma_diff_z[:min_len]
    novelty = novelty[:min_len]
    times = times[:min_len]

    #combine features into a single score
    #weights determined by trial and error
    score = (0.3 * rms_diff_z + 
             0.1 * cent_diff_z + 
             0.05 * zcr_diff_z + 
             0.2 * onset_diff_z +
            0.05 * chroma_diff_z + 
             0.3 * novelty)
    
    
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
times, tempo, beat_times, rms_diff, cent_diff, zcr_diff, onset_diff, chroma_diff,y,sr = extract_features()
cue_times = detect_cues(rms_diff, cent_diff, zcr_diff, onset_diff, chroma_diff, times, beat_times, tempo,y,sr)
print("Expected cue points", cue_times) 