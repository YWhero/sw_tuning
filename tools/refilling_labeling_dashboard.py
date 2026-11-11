"""Single-port dashboard for the three running Refilling labeling servers."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPConnection
import re
from pathlib import Path

CAMERAS = {'head': (8766, '왼쪽 헤드캠'), 'left': (8767, '왼쪽 손목캠'), 'right': (8768, '오른쪽 손목캠')}
PAGE_PATH = Path(__file__).with_name('refilling_labeling_page.html')
# Compatibility for direct single-camera URLs; the default page uses shared controls.
URL = re.compile(r'''(["'`])/(?!/)''')
COMPACT = '<style>body{padding:8px}main{grid-template-columns:110px minmax(0,1fr);gap:6px}.card{padding:8px}aside:last-child{grid-column:1/-1}#episodes{max-height:45vh}header h1{font-size:15px}.stage{max-height:40vh}</style>'

class Handler(BaseHTTPRequestHandler):
    def do_GET(self): self.handle_request()
    def do_POST(self): self.handle_request()
    def handle_request(self):
        if self.headers.get('Host','').split(':')[0] not in ('localhost','127.0.0.1'):
            self.send_error(403); return
        if self.path == '/' and self.command == 'GET':
            b=PAGE_PATH.read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(b);return
        parts=self.path.split('/',2)
        if len(parts)!=3 or parts[1] not in CAMERAS:
            self.send_error(404);return
        key=parts[1];port=CAMERAS[key][0];path='/'+parts[2]
        origin=self.headers.get('Origin')
        if self.command=='POST' and origin and origin not in ('http://'+self.headers['Host'],):
            self.send_error(403);return
        length=int(self.headers.get('Content-Length','0'))
        if length<0 or length>1024*1024:self.send_error(413);return
        body=self.rfile.read(length) if length else None
        headers={k:self.headers[k] for k in ('Content-Type','X-Label-Token','Range') if k in self.headers}
        headers['Host']=f'127.0.0.1:{port}'
        if origin: headers['Origin']=f'http://127.0.0.1:{port}'
        conn=HTTPConnection('127.0.0.1',port,timeout=120)
        try:
            conn.request(self.command,path,body=body,headers=headers);res=conn.getresponse()
            is_html=res.getheader('Content-Type','').startswith('text/html')
            b=None
            if is_html:
                page=URL.sub(lambda m:m[1]+'/'+key+'/',res.read().decode())
                b=page.replace('</style>','</style>'+COMPACT,1).encode()
            self.send_response(res.status)
            for k,v in res.getheaders():
                if k.lower() not in ('connection','transfer-encoding','content-length','server','date'): self.send_header(k,v)
            size=str(len(b)) if b is not None else res.getheader('Content-Length')
            if size is not None:self.send_header('Content-Length',size)
            self.end_headers()
            if b is not None:self.wfile.write(b)
            else:
                while chunk:=res.read(65536):self.wfile.write(chunk)
        except (BrokenPipeError,ConnectionResetError):pass
        finally:conn.close()

if __name__=='__main__':
    import os
    import subprocess
    import sys
    server = ThreadingHTTPServer(('127.0.0.1',8769),Handler)
    root = Path(__file__).resolve().parents[1]
    log = (root/'logs/labeling_sam.log').open('a')
    worker = subprocess.Popen([sys.executable, '-u', str(root/'tools/labeling_sam_worker.py'),
                               '--socket',str(root/'logs/labeling_sam.sock'),
                               '--runtime',str(root/'configs/labeling_runtime.json'),
                               '--parent',str(os.getpid())], stdout=log, stderr=subprocess.STDOUT)
    print('Three-camera labeling: http://127.0.0.1:8769',flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
        (root/'logs/labeling_sam.sock').unlink(missing_ok=True)
        log.close()
