import numpy as np
import subprocess as sp
import os
import string
from glob import glob
import re
import shutil
import librosa
import portion as P
from scipy.signal import find_peaks, peak_widths
import ffmpeg

DEVNULL = open(os.devnull, "w")

def ffmpeg_load_audio(
    filename,
    sr=44100,
    mono=False,
    normalize=True,
    in_type=np.int16,
    out_type=np.float32,
):
    channels = 1 if mono else 2
    format_strings = {
        np.float64: "f64le",
        np.float32: "f32le",
        np.int16: "s16le",
        np.int32: "s32le",
        np.uint32: "u32le",
    }
    format_string = format_strings[in_type]
    command = [
        "ffmpeg",
        "-i",
        filename,
        "-f",
        format_string,
        "-acodec",
        "pcm_" + format_string,
        "-ar",
        str(sr),
        "-ac",
        str(channels),
        "-",
    ]

    p = sp.Popen(command, stdout=sp.PIPE, stderr=DEVNULL, bufsize=4096)

    bytes_per_sample = np.dtype(in_type).itemsize
    frame_size = bytes_per_sample * channels
    chunk_size = frame_size * sr  # read in 1-second chunks
    raw = b""
    with p.stdout as stdout:
        while True:
            data = stdout.read(chunk_size)
            if data:
                raw += data
            else:
                break
    audio = np.frombuffer(raw, dtype=in_type).astype(out_type)
    if channels > 1:
        audio = audio.reshape((-1, channels)).transpose()
    if audio.size == 0:
        return audio, sr
    if issubclass(out_type, np.floating):
        if normalize:
            peak = np.abs(audio).max()
            if peak > 0:
                audio /= peak
        elif issubclass(in_type, np.integer):
            audio /= np.iinfo(in_type).max
    return audio, sr

def find_active(f):
    frame_length=512
    hop_length=256
    audio, sr = ffmpeg_load_audio(
        f, sr=8000, in_type=np.float32, out_type=np.float32
    )

    S = librosa.stft(audio[0],hop_length=hop_length,win_length=frame_length)
    rms= librosa.feature.rms(audio[0],frame_length=frame_length,hop_length=hop_length, center=True)[0]
    logS = librosa.amplitude_to_db(np.abs(S))
    logSmean=logS.mean(axis=0)
    dy = logSmean.max()-logSmean.min()
    peaks, peak_prop = find_peaks(logSmean, height=logSmean.min()+dy/5, prominence=dy/5)
    fp = peak_widths(logSmean,peaks,rel_height=0.9)

    intervals=P.closed(0,0)
    for ll,rr in zip(fp[2],fp[3]):
        intervals = intervals | P.closed(ll,rr)
    bounds=np.array(P.to_data(intervals)[1:])[:,1:3]
    left_frames=[bounds[0,0]]
    right_frames=[bounds[0,1]]
    tol = 0.5*sr/hop_length
    for l,r in bounds[1:]:
        if l-right_frames[-1]>tol:
            left_frames.append(int(l))
            right_frames.append(int(r))
        else:
            right_frames[-1]=r
    
    audiolen=logS.shape[0]
    left_frames=np.array(left_frames)/audiolen        
    right_frames=np.array(right_frames)/audiolen 
    return left_frames,right_frames


path="E:\\VideoSplit\\"
fname = "footage.mp4"
root,ext = fname.split('.')
n_frames = int(sp.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", 
                     "stream=nb_read_frames", "-of", "csv=p=0", path+fname] ))

lfs, rfs = find_active(fname)
print(lfs,rfs)
in_file = ffmpeg.input(path+fname)


for i,(lf,rf) in enumerate(zip(lfs,rfs)):
    if rf>1:
        rf=1.
    outname=f'{root}{i:04d}.{ext}'
    print(i, n_frames/60*lf, n_frames/60*rf)
    sp.call(f'ffmpeg -y -ss {n_frames/60*lf} -i sample_video_cut.mp4 -to {n_frames/60*rf} -c:v copy -c:a copy {outname}')

    