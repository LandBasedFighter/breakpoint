# cuely

experimenting with automatic cue point detection in audio files. 

## usage


this will load the audio, extract RMS / ZCR / spectral centroid, and plot them. The output is currently just a graph of these features, but over time it will be a mp3 file with id3v2 tagging that splits the audio into the desired cue points. later, i'll write something that will support importing into serato, virtualdj, rekordbox,, and traktor.