import cv2
import numpy as np
class Writer:
 def __init__(self,width,height,bitdepth=16,greyscale=True,**kw):
  assert bitdepth==16 and greyscale
  self.width,self.height=width,height
 def write(self,stream,rows):
  a=np.asarray(list(rows),dtype=np.uint16).reshape(self.height,self.width)
  ok,b=cv2.imencode(".png",a);assert ok
  stream.write(b.tobytes())
