#playing around with librosa

import librosa
import matplotlib.pyplot as plt
import numpy as np

# load audio
y, sr = librosa.load("Rush - Troye Sivan.mp3", sr=None)

rms = librosa.feature.rms(y=y)[0]

zcr = librosa.feature.zero_crossing_rate(y)[0]

cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

frames = range(len(rms))
times = librosa.frames_to_time(frames, sr=sr)
plt.figure(figsize=(14, 6))
plt.plot(times, rms, label='RMS Energy', color='orange')
plt.plot(times, zcr, label='Zero Crossing Rate', color='blue')
plt.plot(times, cent / cent.max(), label='Spectral Centroid (normalized)', color='green')
# ^ centroid normalized so it fits on same scale

plt.xlabel('Time (s)')
plt.ylabel('Feature Value')
plt.title('Basic Audio Features Over Time')
plt.legend()
plt.show()