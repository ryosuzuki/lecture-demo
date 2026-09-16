#!/usr/bin/env python3
"""Local classroom server. Static labs + optional real Coconut inference.
Bind localhost, no API credentials and no public upload required.
"""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import json, sys, threading, argparse, re
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'latent'))
lock=threading.Lock()
reason_lock=threading.Lock()
sys.path.insert(0,str(ROOT/'reason'))
try:
 import backend as reason_backend
except ImportError:
 reason_backend=None
try:
 import infer
 latent_available=infer.CHECKPOINT.exists()
except ImportError:
 infer=None;latent_available=False
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT),**kwargs)
 def send_head(self):
  self._remaining=None
  path=Path(self.translate_path(self.path))
  range_header=self.headers.get('Range')
  if range_header and path.is_file() and path.suffix.lower() in ('.mp4','.webm'):
   size=path.stat().st_size
   match=re.fullmatch(r'bytes=(\d*)-(\d*)',range_header.strip())
   if not match:return super().send_head()
   a,b=match.groups()
   if not a and not b:return super().send_head()
   start=int(a) if a else max(0,size-int(b))
   end=min(size-1,int(b)) if a and b else size-1
   if start>=size or end<start:
    self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return None
   f=path.open('rb');f.seek(start);self._remaining=end-start+1
   self.send_response(206);self.send_header('Content-Type',self.guess_type(str(path)));self.send_header('Accept-Ranges','bytes');self.send_header('Content-Range',f'bytes {start}-{end}/{size}');self.send_header('Content-Length',str(self._remaining));self.end_headers();return f
  return super().send_head()
 def copyfile(self,source,outputfile):
  if getattr(self,'_remaining',None) is None:return super().copyfile(source,outputfile)
  remaining=self._remaining
  while remaining>0:
   block=source.read(min(65536,remaining))
   if not block:break
   outputfile.write(block);remaining-=len(block)
 def json(self,obj,status=200):
  content=json.dumps(obj).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
 def do_GET(self):
  if self.path=='/api/health':
   state={'latent':latent_available,'local':True,'reason':False}
   if reason_backend:
    try: state.update(reason_backend.health())
    except Exception: pass
   return self.json(state)
  return super().do_GET()
 def do_POST(self):
  if self.path=='/api/reason':
   if not reason_backend:return self.json({'error':'Local reasoning model is not installed'},503)
   if not reason_lock.acquire(blocking=False):return self.json({'error':'Reasoning inference busy; retry shortly'},429)
   try:
    size=int(self.headers.get('Content-Length','0'))
    if not 0<size<24000:raise ValueError('Invalid request size')
    data=json.loads(self.rfile.read(size))
    self.json(reason_backend.run(data.get('question',''),data.get('thinking',True),data.get('max_tokens',256),data.get('n',1)))
   except Exception as e:self.json({'error':str(e)},400)
   finally:reason_lock.release()
   return
  if self.path!='/api/latent':return self.json({'error':'Not found'},404)
  if not latent_available:return self.json({'error':'Model environment/checkpoint not installed. Recorded inference is available.'},503)
  if not lock.acquire(blocking=False):return self.json({'error':'Another inference is running; retry shortly.'},429)
  try:
   size=int(self.headers.get('Content-Length','0'))
   if not 0<size<20000:raise ValueError('Invalid request size')
   data=json.loads(self.rfile.read(size));result=infer.run(data.get('question'),data.get('steps',6));self.json(result)
  except Exception as e:self.json({'error':str(e)},400)
  finally:lock.release()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--port',default=8108,type=int);a=p.parse_args()
 print(f'Reasoning Lab: http://127.0.0.1:{a.port}/; latent inference={latent_available}',flush=True)
 ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
