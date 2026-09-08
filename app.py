import uuid, shutil, subprocess, threading, json, time
from pathlib import Path
import cv2, numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
BASE=Path(__file__).resolve().parent; DATA=BASE/'data'
UPLOADS,OUTPUTS,FRAMES=DATA/'uploads',DATA/'outputs',DATA/'frames'
for p in (UPLOADS,OUTPUTS,FRAMES): p.mkdir(parents=True,exist_ok=True)
app=FastAPI(title='4K Video Chatbot'); app.mount('/static',StaticFiles(directory=BASE/'static'),name='static')
JOBS={}; CANCEL_EVENTS={}; LOCK=threading.Lock()
def run(cmd):
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 if p.returncode: raise RuntimeError(p.stdout[-4000:])
 return p.stdout
def run_cancellable(cmd,event):
 p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 while True:
  if event.is_set():
   try: p.terminate(); p.wait(timeout=3)
   except Exception:
    try:p.kill()
    except Exception:pass
   raise InterruptedError('Đã dừng xử lý.')
  rc=p.poll()
  if rc is not None:
   out=p.stdout.read() if p.stdout else ''
   if rc: raise RuntimeError(out[-4000:])
   return out
  time.sleep(.2)
def ffprobe(path):
 out=run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,duration','-of','json',str(path)])
 return json.loads(out)['streams'][0]
def manual_mask(w,h,x,y,ww,hh):
 mask=np.zeros((h,w),np.uint8); x=max(0,min(w-1,int(x/100*w))); y=max(0,min(h-1,int(y/100*h))); ww=max(1,int(ww/100*w)); hh=max(1,int(hh/100*h)); cv2.rectangle(mask,(x,y),(min(w-1,x+ww),min(h-1,y+hh)),255,-1); return mask
def auto_mask(frame,wm_path):
 sample=cv2.imread(str(wm_path),cv2.IMREAD_COLOR)
 if sample is None: raise RuntimeError('Không đọc được ảnh mẫu watermark.')
 fh,fw=frame.shape[:2]; sh,sw=sample.shape[:2]
 if sh>fh or sw>fw: raise RuntimeError('Ảnh mẫu watermark lớn hơn video.')
 gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); sgray=cv2.cvtColor(sample,cv2.COLOR_BGR2GRAY)
 result=cv2.matchTemplate(gray,sgray,cv2.TM_CCOEFF_NORMED); _,_,_,loc=cv2.minMaxLoc(result)
 mask=np.zeros((fh,fw),np.uint8); x,y=loc; pad=max(3,int(min(sw,sh)*.12)); cv2.rectangle(mask,(max(0,x-pad),max(0,y-pad)),(min(fw,x+sw+pad),min(fh,y+sh+pad)),255,-1); return mask
def cleanup(jid):
 shutil.rmtree(FRAMES/jid,ignore_errors=True)
 for p in list(UPLOADS.glob(f'{jid}_*'))+list(OUTPUTS.glob(f'{jid}_*')): p.unlink(missing_ok=True)
def process(jid,vp,wp,mode,coords):
 event=CANCEL_EVENTS[jid]; outdir=FRAMES/jid
 try:
  with LOCK:JOBS[jid].update(status='processing',message='Đang đọc video...')
  meta=ffprobe(vp); w,h=int(meta['width']),int(meta['height']); fps=eval(meta.get('r_frame_rate','30/1'))
  outdir.mkdir(parents=True,exist_ok=True); cap=cv2.VideoCapture(str(vp)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1; mask=None; i=0
  while True:
   if event.is_set(): raise InterruptedError('Đã dừng xử lý.')
   ok,frame=cap.read()
   if not ok: break
   if mask is None: mask=manual_mask(w,h,*coords) if mode=='manual' else (auto_mask(frame,wp) if wp else manual_mask(w,h,82,82,16,16))
   cv2.imwrite(str(outdir/f'{i:08d}.png'),cv2.inpaint(frame,mask,5,cv2.INPAINT_TELEA)); i+=1
   if i%5==0:
    with LOCK:JOBS[jid].update(progress=int(i/total*70),message=f'Xóa watermark: {i}/{total} frame')
  cap.release()
  if event.is_set(): raise InterruptedError('Đã dừng xử lý.')
  with LOCK:JOBS[jid].update(progress=75,message='Đang upscale lên 4K...')
  tw,th=(2160,3840) if h>=w else (3840,2160); silent=OUTPUTS/f'{jid}_silent.mp4'
  run_cancellable(['ffmpeg','-y','-framerate',str(fps),'-i',str(outdir/'%08d.png'),'-vf',f'scale={tw}:{th}:flags=lanczos','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p',str(silent)],event)
  final=OUTPUTS/f'{jid}_4k.mp4'
  try: run_cancellable(['ffmpeg','-y','-i',str(silent),'-i',str(vp),'-map','0:v:0','-map','1:a:0?','-c:v','copy','-c:a','aac','-b:a','192k','-shortest',str(final)],event)
  except InterruptedError: raise
  except Exception:
   if event.is_set(): raise InterruptedError('Đã dừng xử lý.')
   shutil.copy2(silent,final)
  shutil.rmtree(outdir,ignore_errors=True); silent.unlink(missing_ok=True)
  with LOCK:JOBS[jid].update(status='done',progress=100,message='Hoàn tất',download=f'/download/{jid}')
 except InterruptedError:
  cleanup(jid)
  with LOCK:JOBS[jid].update(status='cancelled',progress=0,message='Đã dừng xử lý.',download=None)
 except Exception as e:
  cleanup(jid)
  with LOCK:JOBS[jid].update(status='error',message=str(e),download=None)
@app.get('/',response_class=HTMLResponse)
def home(): return (BASE/'static'/'index.html').read_text(encoding='utf-8')
@app.post('/api/jobs')
async def create_job(video:UploadFile=File(...),watermark:UploadFile|None=File(None),mode:str=Form('auto'),x:float=Form(82),y:float=Form(82),ww:float=Form(16),hh:float=Form(16)):
 if not video.filename.lower().endswith(('.mp4','.mov','.mkv','.webm','.avi')): raise HTTPException(400,'Định dạng video không được hỗ trợ.')
 jid=uuid.uuid4().hex; vp=UPLOADS/f'{jid}_video{Path(video.filename).suffix.lower()}'
 with vp.open('wb') as f: shutil.copyfileobj(video.file,f)
 wp=None
 if watermark and watermark.filename:
  wp=UPLOADS/f'{jid}_wm{Path(watermark.filename).suffix.lower()}'
  with wp.open('wb') as f: shutil.copyfileobj(watermark.file,f)
 with LOCK:JOBS[jid]={'status':'queued','progress':0,'message':'Đang xếp hàng','download':None}; CANCEL_EVENTS[jid]=threading.Event()
 threading.Thread(target=process,args=(jid,vp,wp,mode,(x,y,ww,hh)),daemon=True).start(); return {'job_id':jid}
@app.get('/api/jobs/{job_id}')
def get_job(job_id):
 with LOCK:
  if job_id not in JOBS: raise HTTPException(404,'Không tìm thấy tác vụ.')
  return dict(JOBS[job_id])
@app.post('/api/jobs/{job_id}/cancel')
def cancel_job(job_id):
 with LOCK:
  job=JOBS.get(job_id); event=CANCEL_EVENTS.get(job_id)
  if not job or not event: raise HTTPException(404,'Không tìm thấy tác vụ.')
  if job['status'] in ('done','error','cancelled'): return {'ok':False,'status':job['status'],'message':'Tác vụ đã kết thúc.'}
  event.set(); job['message']='Đang dừng...'; return {'ok':True,'status':'cancelling','message':'Đang dừng xử lý...'}
@app.get('/download/{job_id}')
def download(job_id):
 with LOCK:
  job=JOBS.get(job_id); path=OUTPUTS/f'{job_id}_4k.mp4'
  if not job or job.get('status')!='done': raise HTTPException(404,'Video chưa sẵn sàng.')
 if not path.exists(): raise HTTPException(404,'File đã được dọn khỏi máy chủ.')
 return FileResponse(path,media_type='video/mp4',filename='video_4k.mp4')
