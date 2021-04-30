import ffmpeg
import sys
import torch
import torchaudio

device = torch.device("cpu")
model, decoder, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_stt",
    language="en",  # also available 'de', 'es'
    device=device,
)

text = """Dante has officially infiltrated Hypixel. You probably want to stick around to the end because it gets crazy.
Yesterday at around 9:20 pm eastern time, I was in a call with my coop member Vudge when he stumbled upon a Dante goon in the main Hypixel hub. I instantly went and followed him but the Dante goon didn’t appear for me, so he recorded the following footage for me:
He says he isn’t jebaiting me, but I have a feeling that he is, leave a comment letting me know if you see the goon. Subscribe and leave a like because I will be giving away another 100 million coins if we hit 10,000 subscribers but by the way"""

data = torchaudio.load("DanteTest.webm")
print(data)

# in_file = ffmpeg.input("DanteTest.webm")


# v1 = in_file.video.filter("trim", start=0, end=2).setpts("PTS-STARTPTS")
# a1 = in_file.audio.filter("atrim", start=0, end=2).filter("asetpts", "PTS-STARTPTS")


# v4 = v1.overlay(v2)


# v6 = ffmpeg.concat(v4, a1, v=1, a=1)
# out = ffmpeg.output(v6, "out.mp4")
# out.run()
