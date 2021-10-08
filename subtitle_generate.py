import os
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

colors={0:(255, 255, 255, 255),
    1:(0, 0, 0, 255),
    2:(255, 0, 0, 255),
    3:(255, 0, 221, 255),  #pink
    4:(0,0,255,255), #blue
    5:(0,255,0.69,255),
    6:(128,0,128,255)} #purple

text="""after the events of
the last part where
i found a tunnel
leading me to my
enemy's base and
leaked my coordinates
on the public smp since
they are always shown
on stream...
my enemies came over
prepared for an
all out war...
being a freshman
in high school i did
what any 9th grader
would do
work extremely hard
and try to succeed
yeah about that...
full story coming soon
server ip in Discord""" 

root='./subtitles/'
if not os.path.exists(root):
    os.makedirs(root)
texts = text.split('\n')

imw,imh=1920,250
fnt=ImageFont.truetype("Dosis-Bold.ttf",115)

i=0
w_start=0
for text in texts:
    z = re.match(r"[0-9]{3}", text)
    if z:
        font_c = int(text[0])
        outline_c = int(text[1])
        shadow_c = int(text[2])
        joined = ' '.join(text[3:].split())
    else:
        font_c=0
        outline_c=1
        shadow_c=1
        joined = ' '.join(text.split())

    w_end = w_start + len(text.split())
    name = root+f'a_{i:04d}_{w_start:04d}_{w_end:04d}.png'
    print(text)
    i+=1
    w_start = w_end


    im = Image.new('RGBA', (imw,imh), (0,0,0,0))
    d = ImageDraw.Draw(im)
    w,h=d.textsize(text,fnt)
    d.text(((imw-w)//2,(imh-h)//2), text, font=fnt, fill=colors[shadow_c])
    im = im.filter(ImageFilter.GaussianBlur(radius=30))
    imnp_raw=np.asarray(im)
    imnp=imnp_raw.copy()
    imnp[:,:,3]=(imnp_raw[:,:,3]*(255.0/imnp_raw[:,:,3].max())).astype(np.uint8)


    imface = Image.new('RGBA', (imw,imh), (0,0,0,0))
    d2 = ImageDraw.Draw(imface)
    d2.text(((imw-w)//2,(imh-h)//2), text, font=fnt, fill=colors[font_c], stroke_width=7, stroke_fill=colors[outline_c])
    imfacenp = np.asarray(imface)

    blank = np.zeros((imh,imw,4),dtype=np.uint8)
    compnp=np.where((imnp[:,:,3]>0)[:,:,None], imnp, blank)
    compnp=np.where((imfacenp[:,:,3]>200)[:,:,None], imfacenp, compnp)

    Image.fromarray(compnp,'RGBA').save(name)