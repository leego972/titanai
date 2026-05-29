import json, random, sys
from string import Template

random.seed(44)
out = []
binaries = ["./malware","./suspicious","./unknown","./target","./crackme","./sample","./dropper"]

# 1. Static analysis
STATIC = Template('''import subprocess, hashlib, re, sys
from pathlib import Path

BINARY = "$BINARY"

def hashes(path):
    data=open(path,"rb").read()
    return {
        "md5":    hashlib.md5(data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size":   len(data)
    }

def file_type(path):
    return subprocess.run(["file",path],capture_output=True,text=True).stdout.strip()

def interesting_strings(path):
    r=subprocess.run(["strings","-n","6",path],capture_output=True,text=True)
    found=[]
    pats=[
        (r"https?://\\S+","URL"),
        (r"\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}","IP"),
        (r"(?i)(password|passwd|key|token|secret|api)","Credential"),
        (r"(?i)(cmd\\.exe|powershell|/bin/sh|/bin/bash)","Shell"),
        (r"(?i)(createfile|virtualalloc|loadlibrary|getprocaddress)","WinAPI"),
        (r"[A-Za-z0-9+/]{20,}={0,2}","Base64-like"),
    ]
    for line in r.stdout.splitlines():
        for pat,label in pats:
            if re.search(pat,line,re.I):
                found.append((label,line.strip())); break
    return found

def pe_imports(path):
    try:
        import pefile
        pe=pefile.PE(path)
        return {e.dll.decode(errors="ignore"):
                [i.name.decode(errors="ignore") for i in e.imports if i.name]
                for e in pe.DIRECTORY_ENTRY_IMPORT}
    except Exception as e:
        return {"error":str(e)}

def detect_packers(path):
    sigs=["UPX0","UPX1","PECompact","ASPack","Themida","VMProtect"]
    data=open(path,"rb").read(0x400)
    found=[s for s in sigs if s.encode() in data]
    return found

if __name__=="__main__":
    print(f"=== Static Analysis: {BINARY} ===")
    print(f"[*] {file_type(BINARY)}")
    h=hashes(BINARY); print(f"[*] MD5={h[\"md5\"]} SHA256={h[\"sha256\"]} Size={h[\"size\"]}")
    packers=detect_packers(BINARY)
    if packers: print(f"[!] Packer: {packers}")
    print("[*] Interesting strings:")
    for label,s in interesting_strings(BINARY)[:25]: print(f"  [{label}] {s}")
    print("[*] PE imports:")
    for dll,fns in pe_imports(BINARY).items(): print(f"  {dll}: {fns[:5]}")
''')

for i in range(2000):
    code = STATIC.safe_substitute(BINARY=binaries[i%len(binaries)])
    qs = [f"Write a Python static analysis script for malware {binaries[i%len(binaries)]}: hashes, strings, PE imports, packer detection.",
          f"Build a malware triage tool: file type, MD5/SHA256, interesting strings, PE imports for {binaries[i%len(binaries)]}.",
          f"How do I statically analyze {binaries[i%len(binaries)]}? Python: pefile, strings, hashes, packer signatures.",
          f"Create Python malware static triage: IOC extraction, PE imports, packer detection for {binaries[i%len(binaries)]}."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"Static: {len(out)}", file=sys.stderr)

# 2. Frida dynamic analysis
FRIDA = Template('''import frida, sys, json, time

TARGET = "$TARGET"

SCRIPT = """
"use strict";
var hooks = {crypto:[], network:[], files:[], registry:[]};

["CryptEncrypt","CryptDecrypt","BCryptEncrypt","BCryptDecrypt"].forEach(fn => {
    var a=Module.findExportByName(null,fn);
    if(a) Interceptor.attach(a,{onEnter:function(args){send({type:"crypto",fn:fn});}});
});

["connect","WSAConnect"].forEach(fn => {
    var a=Module.findExportByName("ws2_32.dll",fn)||Module.findExportByName(null,fn);
    if(a) Interceptor.attach(a,{onEnter:function(args){
        try {
            var s=args[1];
            var port=(s.add(2).readU8()<<8)|s.add(3).readU8();
            var ip=s.add(4).readU8()+"."+s.add(5).readU8()+"."+s.add(6).readU8()+"."+s.add(7).readU8();
            send({type:"network",fn:fn,ip:ip,port:port});
        } catch(e){}
    }});
});

var cf=Module.findExportByName("kernel32.dll","CreateFileW");
if(cf) Interceptor.attach(cf,{onEnter:function(args){
    send({type:"file",path:args[0].readUtf16String()});
}});

console.log("[Frida] Hooks active");
"""

iocs = {"network":[],"file":[],"crypto":[],"registry":[]}

def on_message(msg, data):
    if msg["type"]=="send":
        p=msg["payload"]; t=p.get("type","?")
        iocs[t]=iocs.get(t,[])+[p]
        print(f"[{t.upper()}] {p}")

def run(target, secs=60):
    try: s=frida.attach(target)
    except frida.ProcessNotFoundError:
        pid=frida.spawn([target]); frida.resume(pid); s=frida.attach(pid)
    sc=s.create_script(SCRIPT); sc.on("message",on_message); sc.load()
    print(f"[*] Monitoring {target} for {secs}s...")
    time.sleep(secs); s.detach()
    return iocs

if __name__=="__main__":
    r=run(TARGET, 60)
    print("\\n=== IOC Summary ===")
    for t,items in r.items(): print(f"  {t}: {len(items)}")
''')

targets = ["malware.exe","suspicious.exe","dropper.exe","sample.exe","unknown.exe"]
for i in range(2000):
    code = FRIDA.safe_substitute(TARGET=targets[i%len(targets)])
    qs = [f"Write Python Frida hooks for dynamic malware analysis of {targets[i%len(targets)]}: crypto, network, file, registry.",
          f"Build Frida-based malware monitor in Python: hook WSAConnect, CreateFileW, CryptEncrypt for {targets[i%len(targets)]}.",
          f"How do I use Frida to analyze malware {targets[i%len(targets)]} dynamically? Python hooks for Windows API calls.",
          f"Create Python Frida IOC collector for {targets[i%len(targets)]}: C2 IPs, dropped files, crypto operations."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"Frida: {len(out)}", file=sys.stderr)

# 3. Deobfuscation
DEOBFUS = Template('''import re, base64, sys, string
from itertools import cycle

BINARY = "$BINARY"

def xor_decrypt(data, key):
    return bytes(b^k for b,k in zip(data, cycle([key] if isinstance(key,int) else key)))

def rc4(key, data):
    S=list(range(256)); j=0
    for i in range(256):
        j=(j+S[i]+key[i%len(key)])%256; S[i],S[j]=S[j],S[i]
    i=j=0; out=[]
    for b in data:
        i=(i+1)%256; j=(j+S[i])%256; S[i],S[j]=S[j],S[i]
        out.append(b^S[(S[i]+S[j])%256])
    return bytes(out)

def b64_decode(s):
    s=s+"=="
    try: return base64.b64decode(s)
    except: return None

def brute_xor(data):
    results=[]
    for key in range(256):
        dec=xor_decrypt(data,key)
        pc=sum(chr(b) in string.printable for b in dec)/max(len(dec),1)
        if pc>0.8: results.append((key,dec.decode(errors="ignore")))
    return results

if __name__=="__main__":
    with open(BINARY,"rb") as f: data=f.read()
    print(f"[*] Analyzing {BINARY} ({len(data)} bytes)")

    b64_pat=re.compile(rb"[A-Za-z0-9+/]{20,}={0,2}")
    print("[*] Base64 candidates:")
    for m in b64_pat.finditer(data):
        dec=b64_decode(m.group().decode(errors="ignore"))
        if dec and all(chr(b) in string.printable for b in dec[:10]):
            print(f"  {m.group()[:40]} -> {dec[:80]}")

    print("[*] XOR brute-force (first 512 bytes):")
    for key,dec in brute_xor(data[:512]):
        if any(x in dec for x in ["http","cmd","exec",".exe",".dll"]):
            print(f"  key=0x{key:02x}: {dec[:100]}")

    print("[*] RC4 common keys:")
    for kw in [b"infected",b"malware",b"backdoor",b"update",b"config"]:
        dec=rc4(kw,data[:0x200])
        if any(b in dec for b in [b"http",b"MZ",b"cmd"]):
            print(f"  key={kw}: {dec[:80]}")
''')

for i in range(2000):
    code = DEOBFUS.safe_substitute(BINARY=binaries[i%len(binaries)])
    qs = [f"Write a Python deobfuscation tool for {binaries[i%len(binaries)]}: XOR brute-force, RC4 common keys, base64 decode.",
          f"Build a malware config extractor: scan {binaries[i%len(binaries)]} for XOR/RC4/base64 encoded C2 strings.",
          f"How do I deobfuscate malware {binaries[i%len(binaries)]}? Python: XOR single-byte brute force, RC4, base64 variants.",
          f"Create Python malware deobfuscator for {binaries[i%len(binaries)]}: extract C2 URLs, keys, encoded payloads."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"Deobfus: {len(out)}", file=sys.stderr)

# 4. r2/Ghidra analysis
R2 = Template('''import subprocess, json, sys

BINARY = "$BINARY"

def r2(binary, cmds):
    combined=" ; ".join(cmds)
    r=subprocess.run(["r2","-q","-c",combined,"-c","q",binary],
                     capture_output=True,text=True,timeout=60)
    return r.stdout

def analyze(binary):
    print(f"[*] radare2 analysis: {binary}")
    out=r2(binary,["aaa","afl","ii","izz","axt sym.imp.system","axt sym.imp.exec"])
    print(out[:3000])
    return out

def find_dangerous(binary):
    dangerous=["system","execve","popen","ShellExecute","WinExec","strcpy","gets","sprintf"]
    for fn in dangerous:
        out=r2(binary,[f"axt sym.imp.{fn}"])
        if out.strip():
            print(f"[!] Cross-refs to {fn}:")
            print(out[:500])

if __name__=="__main__":
    analyze(BINARY)
    print("\\n[*] Dangerous function cross-refs:")
    find_dangerous(BINARY)
''')

for i in range(2000):
    code = R2.safe_substitute(BINARY=binaries[i%len(binaries)])
    qs = [f"Write Python radare2 analysis script for {binaries[i%len(binaries)]}: functions, imports, dangerous xrefs.",
          f"Build Python RE automation using r2pipe/radare2 for {binaries[i%len(binaries)]}: aaa, afl, xrefs to system/exec.",
          f"How do I automate binary analysis of {binaries[i%len(binaries)]} with radare2 in Python?",
          f"Create Python radare2 script: analyze {binaries[i%len(binaries)]}, find cross-refs to dangerous functions."]
    out.append({"messages":[{"role":"user","content":qs[i%4]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"r2: {len(out)}", file=sys.stderr)

random.shuffle(out)
while len(out)<8000:
    out.extend(out[:8000-len(out)])
out=out[:8000]
with open("/workspace/data_upgrades/upgrade_bw.jsonl","w") as f:
    for item in out: f.write(json.dumps(item)+"\n")
print(f"Written {len(out)} to upgrade_bw.jsonl", file=sys.stderr)
