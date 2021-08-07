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
import re
import shutil



text = """after the questionable
events of the last part
where Spadeless lost
his pets...
and then his house
he wanted my
teammate Vudge dead
which happened
i then found out that he
murdered Vudge for
totally no reason
with his friends AgentMac
and Hilton
i didn't even have time
for a proper funeral
so i then just murdered them
IN MINECRAFT
and i also mean tried since
these little pesky
combat loggers
just didn't give me a break
i then gave up and was
told by Wolfurian to go
to the nether to a secret
base"""
text = re.sub(r"[,.!?-] ", " ", text, 0, re.MULTILINE)
text = text.lower()
text = text.split()
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

def to_text(f):
    print(f)
    audio, sr = ffmpeg_load_audio(
        f, sr=16000, in_type=np.float32, out_type=np.float32
    )
    audio = np.expand_dims(audio.astype(np.float32)[0], 0)


    device = torch.device("cpu")
    model, decoder, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_stt",
        language="en",  # also available 'de', 'es'
        device=device,
    )
    # audio, sr = sf.read("pig.wav")
    # audio = np.expand_dims(audio.astype(np.float32)[:,0], 0)
    print(f'{audio.shape=}')

    res = []
    batch_size = 3 * sr
    N = audio.size // batch_size

    sections = np.array_split(audio, N, 1)
    section_start=0
    for i, section in enumerate(sections):
        input=torch.from_numpy(section)
        print(f'{input.shape=} {input.dtype=}')
        output = model(input)[0]
        print(f'{output.shape=} {output.dtype=}')
        decoded=decoder(output.cpu(), section.size, word_align=True)
        if len(decoded)==2:
            s,dlist = decoded
            for d in dlist:
                d["start_ts"] = (d["start_ts"] + section_start) / sr
                d["end_ts"] = (d["end_ts"] +  section_start) / sr
                print(d["word"], d["start_ts"], d["end_ts"])
                res.append(d)
            section_start+=section.size


    df = pd.DataFrame(res)
    df.to_csv("voice_times.csv")

    voice_words = df["word"].values
    print(" ".join(voice_words))

def align():
    df = pd.read_csv("voice_times.csv")
    voice_words = df["word"].values
    start_times = df["start_ts"].values
    end_times = df["end_ts"].values

    all_words = set(text).union(set(voice_words))

    word_dict = bidict(zip(all_words, range(len(all_words))))
    word_dict["-"] = "-"
    voice = [word_dict[word] for word in voice_words]
    script = [word_dict[word] for word in text]
    align = pairwise2.align.globalxx(voice, script, gap_char =['-'], one_alignment_only=True)[0]
    matches=[]
    mismatches=[]
    mismatch_voice=word_dict.inverse[align.seqA[0]]   
    mismatch_text=word_dict.inverse[align.seqB[0]]
    v_end_pos=0
    v_start_pos=0
    t_end_pos=0
    t_start_pos=0
    for i,(v, t) in enumerate(zip(align.seqA[1:], align.seqB[1:])):
        print(word_dict.inverse[v], word_dict.inverse[t])
        if v==t:
            mismatches.append([t_start_pos, t_end_pos, start_times[v_start_pos], end_times[v_end_pos], mismatch_voice, mismatch_text])
            v_end_pos+=1
            v_start_pos=v_end_pos
            t_end_pos+=1
            t_start_pos=t_end_pos
            mismatch_voice=word_dict.inverse[v]
            mismatch_text=word_dict.inverse[t]
        else:
            if word_dict.inverse[v] != '-' :
                v_end_pos+=1
            if word_dict.inverse[t] != '-' :
                t_end_pos+=1
            mismatch_voice+=' ' + word_dict.inverse[v]
            mismatch_text+=' ' + word_dict.inverse[t]


    mismatches.append([t_start_pos, t_end_pos, start_times[v_start_pos], end_times[v_end_pos], mismatch_voice, mismatch_text])

    idx=[]
    times=[]
    for t_start_pos, t_end_pos, start_time, end_time, v, t in mismatches:
        print('-'*30)
        print(v)
        print(t)
        print(t_start_pos,t_end_pos, start_time, end_time)
        d={}

        t2 = t.translate(t.maketrans(string.punctuation, " " * len(string.punctuation)))
        t2 = " ".join(t2.split())

        idx.append(t_start_pos)
        times.append(start_time)

    idx.append(t_end_pos+1)
    times.append(end_time)

    np.savez('it',idx=np.array(idx),time=np.array(times))
    it = np.load('it.npz')
    print(it['idx'], len(it['idx']))
    print(it['time'], len(it['time']))

def make_copies():
    it = np.load('d:\Github\subtitle_generator\it.npz')
    idx=it['idx']
    times=it['time']
    print(times, len(times))
    path_out='E:\\Blender\\BlenderVideo\\'
    path_in='E:\\Blender\\BlenderOut\\'

    i=0
    subtitle_images = glob(path_in+'*.png')
    print(subtitle_images)
    for subtitle_image in subtitle_images:
        name, subtitle_n, w_start, w_end = subtitle_image.split('_')
        *_, name = name.split('\\')
        w_end,_ = w_end.split('.')
        w_start, w_end, subtitle_n =int(w_start), int(w_end), int(subtitle_n)
        w_start+=1
        w_end+=1
        t_start, t_end = np.interp([w_start, w_end], idx, times)
        f_start, f_end = int(t_start*10), int(t_end*10)
        print(name, w_start, w_end, f_start, f_end)
        for copy_n in range(f_end-f_start):
            # file_out = f'{name}_{subtitle_n:04d}_{copy_n:05d}.png'
            # file_out = f'{name}{i:04d}.png'
            # print(file_out)
            
            name=f'{i:04d}'.translate(str.maketrans('0123456789','abcdefghij'))
            shutil.copyfile(subtitle_image, path_out+name+'.png')
            i+=1



# to_text("E:\\Blender\\audio.mp4")
# align()
make_copies()