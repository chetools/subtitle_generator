import numpy as np
import subprocess as sp
import os
import torch
import torchaudio
import soundfile as sf
import pandas as pd
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from bidict import bidict
import string
from glob import glob

# text = """Dante has officially infiltrated Hypixel. You probably want to stick around to the end because it gets crazy.
# Yesterday at around 9:20 pm eastern time, I was in a call with my coop member Vudge when he stumbled upon a Dante goon in the main Hypixel hub. I instantly went and followed him but the Dante goon didn’t appear for me, so he recorded the following footage for me:
# He says he isn’t jebaiting me, but I have a feeling that he is, leave a comment letting me know if you see the goon. Subscribe and leave a like because I will be giving away another 100 million coins if we hit 10,000 subscribers but by the way"""

# text = text.translate(text.maketrans(string.punctuation, " " * len(string.punctuation)))
# text = text.lower()
# text = text.split()

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
    audio = np.fromstring(raw, dtype=in_type).astype(out_type)
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


audio, sr = ffmpeg_load_audio(
    "pig.mp4", sr=16000, in_type=np.float32, out_type=np.float32
)


device = torch.device("cpu")
model, decoder, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_stt",
    language="en",  # also available 'de', 'es'
    device=device,
)
# audio, sr = sf.read("pig.wav")
audio = np.expand_dims(audio.astype(np.float32)[0], 0)
res = []
batch_size = 5 * sr
N = audio.size // batch_size

sections = np.array_split(audio, N, 1)
for i, section in enumerate(sections):
    output = model(torch.from_numpy(section))[0]
    print(output.shape)
    s, dlist = decoder(output.cpu(), section.size, word_align=True)
    print(s)
    for d in dlist:
        d["start_ts"] = (d["start_ts"] + i * batch_size) / sr
        d["end_ts"] = (d["end_ts"] + i * batch_size) / sr
        print(d["word"], d["start_ts"], d["end_ts"])
        res.append(d)


df = pd.DataFrame(res)
df.to_csv("result.csv")

voice_words = df["word"].values
print(" ".join(voice_words))

# df = pd.read_csv("result.csv")
# voice_words = df["word"].values

# print(" ".join(voice_words))

# all_words = set(text).union(set(voice_words))

# word_dict = bidict(zip(all_words, range(len(all_words))))
# word_dict["-"] = "-"
# voice = [word_dict[word] for word in voice_words]
# script = [word_dict[word] for word in text]
# alignments = pairwise2.align.globalxx(voice, script, gap_char=["-"])
# for a in alignments:
#     for v, s in zip(a.seqA, a.seqB):
#         print(word_dict.inverse[v], word_dict.inverse[s])
