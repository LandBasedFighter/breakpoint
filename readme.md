# breakpoint

experimenting with automatic [cue point detection]([url](https://arxiv.org/html/2407.06823v1#S3)) in songs. 


# how it works

breakpoint uses librosa to extract zcr, spectral centroid, rms, chroma, and onset envelope and performs feature ranking to beats in a song.

# usage

run in cli using:

```python breakpoint.py```
