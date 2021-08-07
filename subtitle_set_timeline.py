import numpy as np
from lxml import etree

def set_timeline():
    it = np.load('d:\Github\subtitle_generator\it.npz')
    idx=it['idx']
    times=it['time']
    print(times, len(times))


    tree = etree.parse('Timeline 1.xml')
    clips=tree.xpath('//video//clipitem')
    for clip in clips:
        name=clip.xpath('name/text()')[0]
        start=clip.xpath('start')[0]
        end=clip.xpath('end')[0]
        start.text=str(20)
        end.text=str(30)
        name, subtitle_n, w_start, w_end = name.split('_')
        *_, name = name.split('\\')
        w_end,_ = w_end.split('.')
        w_start, w_end, subtitle_n =int(w_start), int(w_end), int(subtitle_n)
        t_start, t_end = np.interp([w_start, w_end], idx, times)
        start.text=str(int(t_start*60))
        end.text=str(int(t_end*60))

    f = open('subtitle_timeline.xml', 'wb')
    f.write(etree.tostring(tree, pretty_print=True))
    f.close()
    
set_timeline()