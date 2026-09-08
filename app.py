import uuid
import shutil
import subprocess
import threading
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
UPLOADS, OUTPUTS, FRAMES = DATA/"uploads", DATA/"outputs", DATA/"frames"
for p in (UPLOADS, OUTPUTS, FRAMES):
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="4K Video Chatbot")
app.mount("/static", StaticFiles(directory=BASE/"static"), name="static")
JOBS = {}
LOCK = threading.Lock()

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode:
        raise RuntimeError(p.stdout[-4000:])
    return p.stdout

def ffprobe(path):
    out = run(["ffprobe","-v","error","-select_streams","v:0",
               "-show_entries","stream=width,height,r_frame_rate,duration",
               "-of","json",str(path)])
    return json.loads(out)["streams"][0]

def manual_mask(w, h, x, y, ww, hh):
    m = np.zeros((h,w), np.uint8)
    x1 = max(0,min(w-1,int(w*x/100)))
    y1 = max(0,min(h-1,int(h*y/100)))
    x2 = max(x1+1,min(w,int(w*(x+ww)/100)))
    y2 = max(y1+1,min(h,int(h*(y+hh)/100)))
    m[y1:y2,x1:x2] = 255
    return m

def auto_mask(frame, sample_path):
    h,w = frame.shape[:2]
    # Watermark sample is searched in the right half of the frame.
    roi_x = int(w*0.55)
    crop = frame[:, roi_x:]
    sample = cv2.imread(str(sample_path))
    if sample is None:
        return manual_mask(w,h,82,82,16,16)

    sh,sw = sample.shape[:2]
    if sh > crop.shape[0] or sw > crop.shape[1]:
        scale = min(crop.shape[0]/max(sh,1), crop.shape[1]/max(sw,1))
        sample = cv2.resize(sample,(max(8,int(sw*scale)),max(8,int(sh*scale))))

    g1 = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(sample,cv2.COLOR_BGR2GRAY)
    if g2.shape[0] > g1.shape[0] or g2.shape[1] > g1.shape[1]:
        return manual_mask(w,h,82,82,16,16)

    res = cv2.matchTemplate(g1,g2,cv2.TM_CCOEFF_NORMED)
    _,_,_,loc = cv2.minMaxLoc(res)
    sx,sy = loc
    sw,sh = g2.shape[1],g2.shape[0]
    pad = max(4,int(min(sw,sh)*0.15))
    x1=max(0,roi_x+sx-pad); y1=max(0,sy-pad)
    x2=min(w,roi_x+sx+sw+pad); y2=min(h,sy+sh+pad)

    patch = frame[y1:y2,x1:x2]
    hsv = cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(hsv,np.array([0,0,150]),np.array([180,120,255]))
    bright = cv2.morphologyEx(bright,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    bright = cv2.dilate(bright,np.ones((5,5),np.uint8),iterations=1)
    if cv2.countNonZero(bright) < max(10,bright.size*0.005):
        bright[:] = 255

    m=np.zeros((h,w),np.uint8)
    m[y1:y2,x1:x2]=bright
    return cv2.dilate(m,np.ones((3,3),np.uint8),iterations=1)

def process(job_id, video_path, wm_path, mode, coords):
    try:
        with LOCK:
            JOBS[job_id].update(status="processing",message="Đang đọc video...")
        meta=ffprobe(video_path)
        w,h=int(meta["width"]),int(meta["height"])
        fps=eval(meta.get("r_frame_rate","30/1"))
        outdir=FRAMES/job_id
        outdir.mkdir(parents=True,exist_ok=True)

        cap=cv2.VideoCapture(str(video_path))
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        mask=None; i=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            if mask is None:
                if mode=="manual":
                    mask=manual_mask(w,h,*coords)
                elif wm_path:
                    mask=auto_mask(frame,wm_path)
                else:
                    mask=manual_mask(w,h,82,82,16,16)
            clean=cv2.inpaint(frame,mask,5,cv2.INPAINT_TELEA)
            cv2.imwrite(str(outdir/f"{i:08d}.png"),clean)
            i+=1
            if i%5==0:
                with LOCK:
                    JOBS[job_id]["progress"]=int(i/total*70)
                    JOBS[job_id]["message"]=f"Xóa watermark: {i}/{total} frame"
        cap.release()

        with LOCK:
            JOBS[job_id].update(progress=75,message="Đang upscale lên 4K...")

        if h>=w: tw,th=2160,3840
        else: tw,th=3840,2160
        silent=OUTPUTS/f"{job_id}_silent.mp4"
        run(["ffmpeg","-y","-framerate",str(fps),"-i",str(outdir/"%08d.png"),
             "-vf",f"scale={tw}:{th}:flags=lanczos",
             "-c:v","libx264","-preset","medium","-crf","18","-pix_fmt","yuv420p",str(silent)])
        final=OUTPUTS/f"{job_id}_4k.mp4"
        try:
            run(["ffmpeg","-y","-i",str(silent),"-i",str(video_path),
                 "-map","0:v:0","-map","1:a:0?","-c:v","copy","-c:a","aac",
                 "-b:a","192k","-shortest",str(final)])
        except Exception:
            shutil.copy2(silent,final)

        shutil.rmtree(outdir,ignore_errors=True)
        silent.unlink(missing_ok=True)
        with LOCK:
            JOBS[job_id].update(status="done",progress=100,message="Hoàn tất",
                                download=f"/download/{job_id}")
    except Exception as e:
        with LOCK:
            JOBS[job_id].update(status="error",message=str(e))

@app.get("/",response_class=HTMLResponse)
def home():
    return (BASE/"static/index.html").read_text(encoding="utf-8")

@app.post("/api/jobs")
async def create_job(video:UploadFile=File(...), watermark:UploadFile|None=File(None),
                     mode:str=Form("auto"), x:float=Form(82), y:float=Form(82),
                     ww:float=Form(16), hh:float=Form(16)):
    if not video.filename.lower().endswith((".mp4",".mov",".mkv",".webm",".avi")):
        raise HTTPException(400,"Định dạng video không được hỗ trợ.")
    jid=uuid.uuid4().hex
    vp=UPLOADS/f"{jid}_video{Path(video.filename).suffix.lower()}"
    with vp.open("wb") as f: shutil.copyfileobj(video.file,f)
    wp=None
    if watermark and watermark.filename:
        wp=UPLOADS/f"{jid}_wm{Path(watermark.filename).suffix.lower()}"
        with wp.open("wb") as f: shutil.copyfileobj(watermark.file,f)
    with LOCK:
        JOBS[jid]={"status":"queued","progress":0,"message":"Đang xếp hàng","download":None}
    threading.Thread(target=process,args=(jid,vp,wp,mode,(x,y,ww,hh)),daemon=True).start()
    return {"job_id":jid}

@app.get("/api/jobs/{jid}")
def status(jid):
    with LOCK:
        if jid not in JOBS: raise HTTPException(404,"Không tìm thấy job.")
        return JOBS[jid]

@app.get("/download/{jid}")
def download(jid):
    p=OUTPUTS/f"{jid}_4k.mp4"
    if not p.exists(): raise HTTPException(404,"Video chưa sẵn sàng.")
    return FileResponse(p,media_type="video/mp4",filename="video_4k_no_watermark.mp4")
