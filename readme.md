# breakpoint

`breakpoint` is a small Python CLI that finds likely transition points in a
song and snaps them to the beat grid.

It is aimed at DJs, editors, and producers who want a quick first pass at
where a track changes without scrubbing through the entire waveform.

## Example

```powershell
python backend\main.py "C:\Music\track.mp3"
```

```text
Analyzing track.mp3...
Tempo: 174.2 BPM
Cue points (4):
  0:47
  1:34
  2:21
  3:08
```

The timestamps are structural candidates, not labeled sections. The tool does
not currently decide whether a point is an intro, drop, breakdown, or outro.

## How it works

The pipeline loads audio at 22,050 Hz and measures five frame-level features:

| Feature | Signal it captures |
| --- | --- |
| RMS energy | Volume shifts, drops, and breakdowns |
| Spectral centroid | Brightness changes and filter sweeps |
| Zero-crossing rate | Changes between percussive and tonal content |
| Onset strength | Rhythmic-density changes |
| CQT chroma | Harmonic movement and key changes |

Chroma also feeds a self-similarity matrix. Changes between adjacent matrix
columns produce a harmonic novelty curve that helps identify larger structural
boundaries.

The six signals use fixed bounded transforms, Gaussian smoothing, and a
weighted fusion score. Peaks above the 90th percentile are spaced by at least
20 seconds, quantized to the nearest detected beat, and filtered to keep at
least 16 beats between cue points.

## Stack

- Python
- librosa
- NumPy
- SciPy

## Install

```powershell
git clone https://github.com/LandBasedFighter/breakpoint.git
cd breakpoint
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Usage

```powershell
python backend\main.py path\to\song.mp3
```

Use `--help` to view the command description:

```powershell
python backend\main.py --help
```

## Known limitations

- Feature weights and peak thresholds are hand-tuned rather than learned.
- Beat quantization assumes a mostly steady tempo.
- The chroma self-similarity matrix has quadratic memory usage, so very long
  files are expensive to analyze.
- Results are candidate timestamps without confidence scores or section labels.

## Possible next steps

- Export cue points as JSON or CSV.
- Evaluate detection quality against hand-labeled tracks.
- Learn feature weights from a labeled dataset.
