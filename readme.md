# breakpoint

finds transition points in songs by fusing spectral, rhythmic, and harmonic
features, then snapping the results to the beat grid.

built for djs, editors, and anyone who needs to know where a track changes
without scrubbing through it manually.

## example

$ breakpoint "track.mp3"

loading track.mp3 (3:47, 174 bpm)
extracting features...
found 6 cue points:

  0:00   intro
  0:47
  1:34   [drop]
  2:21
  3:08
  3:41   [outro]

[screenshot or plot here]

## how it works

five features are extracted per frame, then differenced to find rapid change:

| feature | what it catches |
|---|---|
| rms energy | volume shifts, drops, breakdowns |
| spectral centroid | brightness changes, filter sweeps |
| zero-crossing rate | percussive vs. tonal content |
| onset strength | rhythmic density changes |
| chroma (cqt) | harmonic movement, key changes |

chroma additionally feeds a self-similarity matrix, and the column-wise
difference of that matrix gives a novelty curve that catches structural
boundaries the raw features miss.

all six signals are normalized, gaussian-smoothed, and combined into a
weighted score. peaks are picked above the 90th percentile with a minimum
spacing, then quantized to the nearest detected beat and filtered to enforce
a 16-beat minimum gap so cues land musically rather than mid-phrase.

## install

git clone https://github.com/landbasedfighter/breakpoint
cd breakpoint
pip install -r requirements.txt

## usage

breakpoint song.mp3

supports mp3, wav, flac, aiff, ogg.

## known limitations

- weights in the score function are hand-tuned, not learned
- assumes roughly constant tempo; tracks with tempo changes drift
- tuned on [genre you tested]; performance on [other genre] is untested
- no confidence scores, every returned cue is treated as equally likely

## roadmap

- k-means clustering over feature vectors to classify cue types
- learned weights from a labeled dataset