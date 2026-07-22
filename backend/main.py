import argparse
from pathlib import Path

import librosa
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.spatial.distance


SAMPLE_RATE = 22050
HOP_LENGTH = 512
CUE_WINDOW_SECONDS = 20
MIN_CUE_BEATS = 16


def audio_file_path(value):
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Audio file does not exist: {path}")
    return path


def align_frames(feature, frame_count):
    return librosa.util.fix_length(
        feature,
        size=frame_count,
        axis=-1,
        mode="edge",
    )


def bounded_difference(feature, scale=1.0):
    difference = np.abs(np.diff(feature, axis=-1)) / scale
    return np.clip(difference, 0.0, 1.0)


def extract_features(path):
    y, sr = librosa.load(str(path), sr=SAMPLE_RATE)

    frame_count = 1 + len(y) // HOP_LENGTH
    if frame_count < 3:
        raise ValueError("Audio file is too short to detect cue points.")

    rms = align_frames(
        librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0],
        frame_count,
    )
    spectral_centroid = align_frames(
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            hop_length=HOP_LENGTH,
        )[0],
        frame_count,
    )
    spectral_centroid = np.clip(spectral_centroid / (sr / 2.0), 0.0, 1.0)
    zero_crossing_rate = align_frames(
        librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0],
        frame_count,
    )
    onset_envelope = align_frames(
        librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=HOP_LENGTH,
        ),
        frame_count,
    )
    chroma = align_frames(
        librosa.feature.chroma_cqt(
            y=y,
            sr=sr,
            hop_length=HOP_LENGTH,
        ),
        frame_count,
    )

    rms_change = bounded_difference(np.clip(rms, 0.0, 1.0))
    centroid_change = bounded_difference(spectral_centroid)
    zcr_change = bounded_difference(np.clip(zero_crossing_rate, 0.0, 1.0))
    onset_change = -np.expm1(-np.abs(np.diff(onset_envelope)))

    chroma_delta = np.diff(np.clip(chroma, 0.0, 1.0), axis=1)
    chroma_change = np.clip(
        np.linalg.norm(chroma_delta, axis=0) / np.sqrt(chroma.shape[0]),
        0.0,
        1.0,
    )

    chroma_distances = scipy.spatial.distance.cdist(
        chroma.T,
        chroma.T,
        metric="euclidean",
    )
    self_similarity = 1.0 - np.clip(
        chroma_distances / np.sqrt(chroma.shape[0]),
        0.0,
        1.0,
    )
    novelty = np.clip(
        np.linalg.norm(np.diff(self_similarity, axis=1), axis=0)
        / np.sqrt(self_similarity.shape[0]),
        0.0,
        1.0,
    )

    times = librosa.frames_to_time(
        np.arange(frame_count - 1),
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    tempo, beats = librosa.beat.beat_track(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    tempo = float(np.asarray(tempo).reshape(-1)[0])
    beat_times = librosa.frames_to_time(
        beats,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    return (
        times,
        tempo,
        beat_times,
        rms_change,
        centroid_change,
        zcr_change,
        onset_change,
        chroma_change,
        novelty,
    )


def detect_cues(
    rms_change,
    centroid_change,
    zcr_change,
    onset_change,
    chroma_change,
    novelty,
    times,
    beat_times,
    tempo,
):
    features = np.vstack(
        [
            rms_change,
            centroid_change,
            zcr_change,
            onset_change,
            chroma_change,
            novelty,
        ]
    )
    if features.shape[1] != len(times):
        raise ValueError("Feature frames are not aligned to the shared time axis.")

    smoothed_features = scipy.ndimage.gaussian_filter1d(
        features,
        sigma=2,
        axis=1,
    )
    weights = np.array([0.3, 0.1, 0.05, 0.2, 0.05, 0.3])
    score = weights @ smoothed_features

    if score.size < 3 or beat_times.size == 0:
        return []

    threshold = np.percentile(score, 90)
    distance = max(
        1,
        round(CUE_WINDOW_SECONDS * SAMPLE_RATE / HOP_LENGTH),
    )
    peaks, _ = scipy.signal.find_peaks(
        score,
        height=threshold,
        distance=distance,
        prominence=0.1,
    )
    cue_times = times[peaks]

    if not np.isfinite(tempo) or tempo <= 0:
        raise ValueError(f"Invalid tempo returned by beat tracking: {tempo}")

    min_interval = MIN_CUE_BEATS * (60.0 / tempo)
    cue_times_quantized = []
    last_cue_time = -np.inf

    for cue_time in cue_times:
        nearest_beat = beat_times[np.argmin(np.abs(beat_times - cue_time))]
        if nearest_beat - last_cue_time >= min_interval:
            cue_times_quantized.append(nearest_beat)
            last_cue_time = nearest_beat

    return [
        f"{int(cue_time // 60)}:{int(cue_time % 60):02}"
        for cue_time in cue_times_quantized
    ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect beat-aligned cue points in an audio file."
    )
    parser.add_argument("audio_file", type=audio_file_path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"Analyzing {args.audio_file.name}...")
    (
        times,
        tempo,
        beat_times,
        rms_change,
        centroid_change,
        zcr_change,
        onset_change,
        chroma_change,
        novelty,
    ) = extract_features(args.audio_file)
    cue_times = detect_cues(
        rms_change,
        centroid_change,
        zcr_change,
        onset_change,
        chroma_change,
        novelty,
        times,
        beat_times,
        tempo,
    )
    print(f"Tempo: {tempo:.1f} BPM")

    if not cue_times:
        print("Cue points: none found")
        return

    print(f"Cue points ({len(cue_times)}):")
    for cue_time in cue_times:
        print(f"  {cue_time}")


if __name__ == "__main__":
    main()
