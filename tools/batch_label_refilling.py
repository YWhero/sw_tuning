"""Episode-specific SAM drafts for Refilling; keeps existing labels and records review work."""
import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageDraw
from labeling_storage import Dataset, atomic_json, read_json, digest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/Refilling_Dispenser_lerobot_v21'
OUT = ROOT / 'results/refilling_all_episode_labeling'
CAMERAS = {'head':'cam_left_head', 'left':'cam_left_wrist', 'right':'cam_right_wrist'}


def caps(frame, roi=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (120,70,25), (175,255,255))
    if roi:
        x0,y0,x1,y1 = roi
        keep = np.zeros_like(mask); keep[y0:y1,x0:x1] = mask[y0:y1,x0:x1]; mask = keep
    mask = cv2.erode(mask, np.ones((5,5),np.uint8))
    n, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    return [{'xy':centers[i].tolist(), 'area':int(stats[i,cv2.CC_STAT_AREA])}
            for i in range(1,n) if 50 < stats[i,cv2.CC_STAT_AREA] < 9000]


def prompt(frame, positive, negative=()):
    return {'frame':int(frame), 'points':list(positive)+list(negative),
            'labels':[1]*len(positive)+[0]*len(negative)}


def dark_point(frame, target, offset):
    x,y = np.array(target)+offset
    x0,x1=max(0,int(x)-20),min(frame.shape[1],int(x)+21)
    y0,y1=max(0,int(y)-15),min(frame.shape[0],int(y)+16)
    if x1<=x0 or y1<=y0:return None
    yy,xx=np.mgrid[y0:y1,x0:x1]
    dark=frame[y0:y1,x0:x1].max(2)<65
    dark &= ((xx-target[0])**2+(yy-target[1])**2)>32**2
    if dark.sum()<20:return None
    distance=(xx-x)**2+(yy-y)**2;distance=np.where(dark,distance,np.inf)
    iy,ix=np.unravel_index(np.argmin(distance),distance.shape)
    return [int(xx[iy,ix]),int(yy[iy,ix])]



def hand_templates():
    if hasattr(hand_templates,'cache'):return hand_templates.cache
    ds=Dataset(DATA,'observation.images.rgb.cam_left_wrist');j=ds.latest(0)
    masks=np.load(Path(j['directory'])/'object_masks.npz')['2'];templates=[]
    points={0:[[430,145],[450,320]],130:[[435,110]],240:[[290,120],[440,60]],
            320:[[360,25]],520:[[380,190],[400,295]],720:[[340,210],[320,310],[455,255]],
            800:[[360,270],[260,320]],880:[[440,145],[450,325]]}
    for index,xy_list in points.items():
        gray=cv2.cvtColor(ds.frame(0,index),cv2.COLOR_BGR2GRAY)
        for x,y in xy_list:
            if not masks[index,y,x]:continue
            patch=gray[y-15:y+16,x-15:x+16]
            if patch.shape==(31,31) and patch.std()>15:templates.append(patch)
    if len(templates)<4:raise ValueError('Insufficient reviewed hand templates')
    hand_templates.cache=templates
    return templates


def match_hand(frame):
    gray=cv2.cvtColor(frame[:400],cv2.COLOR_BGR2GRAY);matches=[]
    for patch in hand_templates():
        for scale in (.85,1.,1.15):
            p=cv2.resize(patch,None,fx=scale,fy=scale)
            result=cv2.matchTemplate(gray[:,150:],p,cv2.TM_CCOEFF_NORMED)
            _,score,_,(x,y)=cv2.minMaxLoc(result)
            if score>=.90:matches.append((score,[x+150+p.shape[1]//2,y+p.shape[0]//2]))
    points=[]
    for score,xy in sorted(matches,reverse=True):
        if all(np.linalg.norm(np.array(xy)-p)>35 for p in points):points.append(xy)
        if len(points)==3:break
    return points


def prepare(ep):
    left=Dataset(DATA,'observation.images.rgb.cam_left_wrist')
    initial_left=left.frame(ep,0);tray=caps(initial_left,(50,0,330,100))
    if len(tray) not in (1,2,3):raise ValueError(f'Uncertain tray bottle count: {len(tray)}')
    category=3-len(tray)
    plan={'episode':ep,'reference_task':category,'tray_bottles':len(tray),
          'selection':'rightmost remaining bottle on tray, checked against initial left wrist cap layout', 'cameras':{}}
    for route,cam in CAMERAS.items():
        ds=Dataset(DATA,'observation.images.rgb.'+cam);meta=ds.episode(ep);first=ds.frame(ep,0)
        height,width=first.shape[:2];grasp_end=meta['sub_tasks'][0]['end']
        hand=[];bottle=[];evidence=[]
        if route in ('head','left'):
            roi=(880,180,1100,450) if route=='head' else (50,0,330,100)
            candidates=caps(first,roi)
            override=read_json(OUT/'head_cap_overrides.json',{}).get('episodes',{}).get(str(ep)) if route=='head' else None
            if override:candidates=[{'xy':xy} for xy in override]
            if len(candidates)!=len(tray):raise ValueError(f'{cam}: cap count disagreement {len(candidates)} != {len(tray)}')
            target=max(candidates,key=lambda c:c['xy'][0])['xy']
            all_caps=caps(first)
            if override:all_caps=[c for c in all_caps if not(880<c['xy'][0]<1100 and 180<c['xy'][1]<450)]+candidates
            distractors=[c['xy'] for c in all_caps if np.linalg.norm(np.array(c['xy'])-target)>30]
            x,y=target
            pos=[target,[min(width-1,x+8),min(height-1,y+65)]]
            neg=distractors+([[1070,480]] if route=='head' else [[x,min(height-1,y+160)]])
            bottle.append(prompt(0,pos,neg));evidence.append({'frame':0,'bottle_center':target})
            if route=='head':
                hand=[prompt(0,[[843,384],[871,543],[990,653]],[[750,300],[1000,480]])]
            else:
                # Same initial open gripper pose as reviewed episode 0.
                hand=[prompt(0,[[415,150],[448,142],[430,320],[460,323]],[[450,180],[430,560]])]
                last=np.array(target); last_seen=0
                for index in range(60,min(meta['length'],int(meta['length']*.85)),60):
                    f=ds.frame(ep,index)
                    options=[c for c in caps(f) if all(np.linalg.norm(np.array(c['xy'])-d)>48 for d in distractors)]
                    if not options:continue
                    c=min(options,key=lambda c:np.linalg.norm(np.array(c['xy'])-last));xy=c['xy']
                    distance=np.linalg.norm(np.array(xy)-last)
                    if distance>180 and index-last_seen<180:continue
                    last=np.array(xy);last_seen=index
                    # Cap-only prompts collapse the label to the cap. Include visible bottle body.
                    if np.linalg.norm(last-target)>40:
                        bx,by=int(xy[0]+12),int(xy[1]+45)
                        if 0<=bx<width and 0<=by<height and f[by,bx].max()>100:
                            bottle.append(prompt(index,[xy,[bx,by]],distractors))
                    if np.linalg.norm(last-target)>80 and xy[1]>70 and xy[0]>150:
                        points=[dark_point(f,xy,offset) for offset in [(-40,-12),(-25,78)]]
                        points=[p for p in points if p is not None]
                        if len(points)==2:
                            hand.append(prompt(index,points,[xy,[430,560]]))
                    evidence.append({'frame':index,'bottle_center':xy})
        else:
            hand=[prompt(0,[[60,550],[40,600],[410,535],[445,585]],[[240,400],[240,600]])]
            for fraction in (.3,.6,.9):
                index=int(grasp_end*fraction);f=ds.frame(ep,index)
                candidates=caps(f,(140,280,350,500))
                if not candidates:continue
                target=(max(candidates,key=lambda c:c['xy'][1]) if fraction<.8 else min(candidates,key=lambda c:abs(c['xy'][0]-240)))['xy'];x,y=target
                others=[c['xy'] for c in caps(f) if np.linalg.norm(np.array(c['xy'])-target)>45]
                negative=others
                if fraction==.3:
                    negative=negative+[[max(0,x-80),min(639,y+210)],[min(479,x+80),min(639,y+210)]]
                bottle.append(prompt(index,[target,[x,min(639,y+70)]],negative));evidence.append({'frame':index,'bottle_center':target})
            if len(bottle)<2:raise ValueError('Right wrist target ambiguous in grasp phase')
        if route=='left':
            by_frame={p['frame']:p for p in hand}
            for index in range(60,meta['length'],60):
                f=ds.frame(ep,index);points=[xy for xy in match_hand(f) if first[xy[1],xy[0]].max()>110 and f[xy[1],xy[0]].max()<100]
                if points:
                    old=by_frame.get(index)
                    if old:old['points']=points+old['points'];old['labels']=[1]*len(points)+old['labels']
                    else:by_frame[index]=prompt(index,points,[c['xy'] for c in caps(f)]+[[430,560]])
                    if not any(e['frame']==index for e in evidence):evidence.append({'frame':index,'hand_template_anchor':True})
            hand=list(by_frame.values());evidence.sort(key=lambda e:e['frame'])
        seed={'schema_version':2,'episode':ep,'camera':ds.camera,'frame':0,'width':width,'height':height,
              'instruction':ds.instruction(ep,0),'objects':[{'id':1,'name':'water bottle','prompts':bottle},{'id':2,'name':'right hand','prompts':hand}]}
        ds.validate_seed(seed)
        directory=OUT/f'episode_{ep:06d}';directory.mkdir(parents=True,exist_ok=True)
        path=directory/f'{cam}_seed.json';atomic_json(path,seed)
        plan['cameras'][route]={'seed':str(path),'anchors':evidence}
        # Exact seed images for visual checking; no review MP4.
        sheet=Image.new('RGB',(320*4,260*((len(evidence)+3)//4)))
        for k,item in enumerate(evidence):
            f=ds.frame(ep,item['frame'])
            for obj in seed['objects']:
                for p in obj['prompts']:
                    if p['frame']!=item['frame']:continue
                    for xy,label in zip(p['points'],p['labels']):
                        cv2.circle(f,tuple(map(int,xy)),5,(0,255,0) if label else (0,0,255),2)
            im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));im.thumbnail((320,235));x=k%4*320;y=k//4*260;sheet.paste(im,(x,y+20));ImageDraw.Draw(sheet).text((x+3,y+2),f'{cam} frame {item["frame"]}',fill='white')
        sheet.save(directory/f'{cam}_seeds.jpg',quality=90)
    atomic_json(OUT/f'episode_{ep:06d}'/'plan.json',plan)
    return plan


def api(route,path,value=None):
    base='http://127.0.0.1:8769/'+route
    if value is None:
        with urlopen(base+path,timeout=60) as r:return json.load(r)
    token=api(route,'/api/server')['token']
    req=Request(base+path,data=json.dumps(value).encode(),headers={'Content-Type':'application/json','X-Label-Token':token})
    with urlopen(req,timeout=120) as r:return json.load(r)


def audit(ep,route,job):
    p=Path(job['directory']);seed=read_json(p/'seed.json');shape=(job['frames'],seed['height'],seed['width'])
    with np.load(p/'object_masks.npz') as z:arrays={o['name']:z[str(o['id'])] for o in seed['objects']}
    summary={'episode':ep,'camera':CAMERAS[route],'job_id':job['job_id'],'frames':shape[0],
             'visual_review':'pending','auto_approved':False,'objects':{}}
    union=np.zeros(shape,dtype=np.uint8)
    for name,a in arrays.items():
        assert a.shape==shape and a.dtype==np.uint8 and np.isin(a,[0,1]).all()
        union|=a;areas=a.sum((1,2));summary['objects'][name]={'empty_frames':np.where(areas==0)[0].tolist(),'area_min':int(areas.min()),'area_max':int(areas.max())}
    assert np.array_equal(union,np.load(p/'masks.npz')['arr_0']);del union
    summary['seed_validation']=read_json(p/'seed_validation.json',{})
    directory=OUT/f'episode_{ep:06d}'/CAMERAS[route];directory.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(job['video'])
    for i in range(shape[0]):
        ok,f=cap.read();assert ok
        if i%128==0:sheet=Image.new('RGB',(160*16,140*8))
        for name,a in arrays.items():
            m=a[i]>0;color=(50,220,50) if name=='water bottle' else (220,80,220)
            f[m]=(f[m]*.55+np.array(color)*.45).astype('uint8')
        im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));im.thumbnail((160,120));x=i%16*160;y=i%128//16*140;sheet.paste(im,(x,y+18));ImageDraw.Draw(sheet).text((x+2,y+2),str(i),fill='white')
        if i%128==127 or i==shape[0]-1:sheet.save(directory/f'all_{i//128:02d}.jpg',quality=85)
    cap.release();atomic_json(directory/'audit.json',summary)
    return summary


def run(episodes):
    status=read_json(OUT/'batch_status.json',{'completed':{},'failed':{},'skipped_existing':{},'visual_review':'pending'})
    status['stage']='generation_running'
    for ep in episodes:
        if ep==0:continue
        for route in CAMERAS:
            key=f'{ep}:{route}'
            try:
                if __import__('shutil').disk_usage(ROOT).free<30*1024**3:raise RuntimeError('Less than 30 GiB disk free')
                current=api(route,f'/api/job?episode={ep}')
                record=read_json(OUT/f'episode_{ep:06d}'/f'{route}_submitted.json')
                if current.get('status')!='unlabelled' and not(record and current.get('job_id')==record.get('job_id')):
                    status['skipped_existing'][key]=current.get('status');continue
                plan_path=OUT/f'episode_{ep:06d}'/'plan.json'
                plan=read_json(plan_path) or prepare(ep)
                seed=read_json(plan['cameras'][route]['seed'])
                if key in status['completed'] and current.get('seed_hash')==digest(seed) and current.get('status') in ('awaiting_review','approved'):continue
                if not record or current.get('status')=='failed' or (current.get('status')!='running' and current.get('seed_hash')!=digest(seed)):
                    api(route,'/api/seed',seed)
                    current=api(route,'/api/complete',{'episode':ep})
                    atomic_json(OUT/f'episode_{ep:06d}'/f'{route}_submitted.json',current)
                status['current']={'episode':ep,'camera':route,'job_id':current['job_id']};atomic_json(OUT/'batch_status.json',status)
                print('GENERATING',ep,route,current['job_id'],flush=True)
                while current['status']=='running':
                    time.sleep(5);current=api(route,f'/api/job?episode={ep}')
                if current['status'] not in ('awaiting_review','approved'):raise RuntimeError(str(current))
                result=audit(ep,route,current);status['completed'][key]={'job_id':current['job_id'],'frames':result['frames'],'visual_review':'pending'}
                status['failed'].pop(key,None)
                print('GENERATED',ep,route,'visual review pending',flush=True)
            except Exception as exc:
                status['failed'][key]=str(exc);print('FAILED',key,str(exc),flush=True)
            finally:
                status['updated_at']=time.time();atomic_json(OUT/'batch_status.json',status)
    status['stage']='generation_finished_review_pending';status['current']=None;atomic_json(OUT/'batch_status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--episodes',type=int,nargs='+',default=list(range(1,86)));args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if args.prepare_only:
        for ep in args.episodes:print(json.dumps(prepare(ep)),flush=True)
    else:run(args.episodes)
