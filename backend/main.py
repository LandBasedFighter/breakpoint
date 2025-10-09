#playing around with librosa

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
def extract_features(path):
    path = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=[("Audio files", "*.mp3 *.wav *.flac *.aiff *.ogg")],
    )

    y, sr = librosa.load(path, sr=None)
    # zero flattens array, which makes it easy to plot + analyze





    #deltas help find rapid changes
    cent_diff = np.diff(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    rms_diff = np.diff(librosa.feature.rms(y=y)[0]) 
    zcr_diff = np.diff(librosa.feature.zero_crossing_rate(y)[0] )

    #times for plotting
    times = librosa.frames_to_time(np.arange(len(rms_diff)), sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)

    return times, tempo, beat_times, rms_diff, cent_diff, zcr_diff

def detect_cues(rms_diff, times, beat_times):
    # Step 1: find peaks in a weighted combination of features
    wght = ((rms_diff/np.max(rms_diff)) * 0.6) + ((cent_diff/np.max(cent_diff)) * 0.2) + ((zcr_diff/np.max(zcr_diff))) * 0.2 # all normalized to prevent clustering in begeinning

    peaks, props = sig.find_peaks(wght, height=0.02, prominence=0.05, distance=100)

    # Step 2: filter by strong prominence
    strong = props["prominences"] > np.percentile(props["prominences"], 85)
    filtered_peaks = peaks[strong]

    # Step 3: convert frames to time
    candidate_times = times[filtered_peaks]

    # Step 4: snap to nearest beats
    cue_points = [beat_times[np.argmin(np.abs(beat_times - t))] for t in candidate_times]

    # Step 5: remove duplicates & sort
    cue_points = sorted(list(set(cue_points)))
    #Step 6: filter out cue points that are too close together and quantize to nearest beat
    # (at least 16 beats apart)
    # this is to avoid overwhelming the user with too many cues
    min_beats = 16
    filtered_cue_points = []
    for t in cue_points:
        if not filtered_cue_points or (t - filtered_cue_points[-1]) >= (min_beats * 60 / tempo):
            filtered_cue_points.append(t)


    #step 7: make readable
    return filtered_cue_points


print("Welcome to the audio feature extractor!")
times,tempo, beat_times, rms_diff, cent_diff, zcr_diff = extract_features("path/to/your/audio/file.wav")
cue_times = [f"{int(t // 60)}:{int(t % 60):02}" for t in detect_cues(rms_diff, times,beat_times)]
print("Expected cue points", cue_times) #these are very bad on some songs, but good enough for a first pass