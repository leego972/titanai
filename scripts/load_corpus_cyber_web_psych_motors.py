#!/usr/bin/env python3
"""
TitanAI Multi-Domain Corpus Loader  v1  — Maximum Depth
========================================================
14 subjects / 22 domain loaders — Premium training data.
Target: 90,000+ documents.

Subjects:
  1.  Advanced Cyber Systems — Design & Build
  2.  Advanced Cyber Security
  3.  Advanced Web Systems — Design & Build
  4.  Advanced Coding
  5.  Emotional Intelligence
  6.  Behavioural Psychology
  7.  Motor Mechanics
  8.  Operating Machinery & Vehicles
  9.  Aviation — Airplanes & Helicopters
  10. Marine — Boats & Yachts
  11. Hydromechanics
  12. Conflict Resolution & Negotiation
  13. Manners, Politeness & Social Etiquette
  14. Artificial Intelligence

Quality standards:
  • Full Wikipedia articles with section-aware sentence-boundary chunking
  • SE Q&A — multi-domain, top-3 answers, >=500-char gate
  • arXiv — category + keyword filter + dedicated survey/review pass
  • PMC full-text — open-access full papers (psychology, medicine)
  • PubMed abstracts — peer-reviewed content
  • Semantic Scholar — citation-ranked (>=5 cite gate)
  • OpenAlex — citation-sorted open-access
  • Instruction-format wrapping — Q&A instructional pairs
  • Context-aware headers — domain, subfield, type, difficulty
  • Glossary extraction — standalone definition documents
  • Keyword-density filter — rejects off-topic hits
  • MD5 deduplication — no repeated content
  • 500-char minimum threshold
"""

import os, sys, json, time, logging, requests, hashlib, re
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent
RAW  = BASE / "data" / "raw"
LOG  = BASE / "data" / "corpus_cyber_web_psych_motors.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(str(LOG))]
)
log = logging.getLogger("multi_domain_loader")

INVENTORY     = BASE / "data" / "source_inventory.json"
EXCLUSION_LOG = BASE / "data" / "exclusions_multi_domain.jsonl"
try:
    with open(INVENTORY) as f:
        inventory = json.load(f)
except Exception:
    inventory = {}
exclusions = []

_seen_hashes = set()


# ── Core helpers ──────────────────────────────────────────────────────────────

def doc_hash(text):
    return hashlib.md5(text[:300].encode("utf-8", errors="ignore")).hexdigest()

def keyword_density(text, keywords, min_hits=2):
    t = text.lower()
    return sum(1 for k in keywords if k in t) >= min_hits

def record_source(bucket, name, url, mb, reason, n):
    inventory.setdefault(bucket, []).append({
        "source": name, "url": url, "size_mb": round(mb, 2),
        "n_documents": n, "reason_for_inclusion": reason,
        "loaded_at": datetime.utcnow().isoformat(),
    })

def record_exclusion(name, reason):
    exclusions.append({"source": name, "reason": reason,
                       "timestamp": datetime.utcnow().isoformat()})
    log.warning(f"EXCLUDED: {name} — {reason}")

def context_header(domain, subfield, topic, doc_type, difficulty="Graduate"):
    return (f"[Domain: {domain} | Subfield: {subfield} | Topic: {topic} | "
            f"Type: {doc_type} | Level: {difficulty}]")

def instruction_wrap(title, text, domain):
    return (f"Question: Provide a comprehensive expert-level explanation of {title} "
            f"in the context of {domain}.\n\n"
            f"Answer:\n{text.strip()[:6000]}")

def chunk_text_smart(title, text, source_tag, domain, subfield,
                     chunk_size=5000, overlap=400):
    sections = re.split(r'\n(#{1,3} .+)\n', text)
    current_section = "Overview"
    raw_blocks = []
    i = 0
    while i < len(sections):
        part = sections[i]
        if re.match(r'#{1,3} .+', part):
            current_section = part.strip("# ").strip()
            i += 1
            if i < len(sections):
                raw_blocks.append((current_section, sections[i]))
                i += 1
        else:
            raw_blocks.append((current_section, part))
            i += 1
    chunks = []
    for section_name, block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        hdr = (f"{context_header(domain, subfield, title, 'Encyclopedia')}\n"
               f"# {title} — {section_name}\n\n")
        sentences = re.split(r'(?<=[.!?])\s+', block)
        current, part = [], 0
        for sent in sentences:
            current.append(sent)
            joined = " ".join(current)
            if len(joined) >= chunk_size:
                chunks.append(hdr + joined.strip())
                words_back = " ".join(current).split()
                overlap_text = " ".join(words_back[-int(overlap / 6):])
                current = [overlap_text]
                part += 1
        if current:
            remainder = hdr + " ".join(current).strip()
            if len(remainder) > 200:
                chunks.append(remainder)
    if len(text) > 1000:
        chunks.append(
            context_header(domain, subfield, title, "Instruction-QA") +
            "\n\n" + instruction_wrap(title, text[:6000], domain)
        )
    return chunks

def extract_glossary_terms(title, text, domain, subfield):
    docs = []
    patterns = [
        r'\*\*([^*]+)\*\*\s*[—–-]\s*([^.\n]{50,300})',
        r"'''([^']+)'''\s+is\s+([^.\n]{50,300})",
        r'([A-Z][a-z]+ [A-Za-z]+) is (?:defined as |a |an )([^.\n]{60,300})',
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            term = m.group(1).strip()
            defn = m.group(2).strip()
            if len(defn) > 50 and len(term) < 80:
                docs.append(
                    f"{context_header(domain, subfield, term, 'Definition', 'All levels')}\n\n"
                    f"# Definition: {term}\n\n"
                    f"Context: from article '{title}'\n\n{defn}"
                )
    return docs

def write_docs(bucket_dir, docs, tag, domain_kws=None):
    bucket_dir.mkdir(parents=True, exist_ok=True)
    existing = list(bucket_dir.glob(f"{tag}_*.txt"))
    start = len(existing)
    n = 0
    for i, text in enumerate(docs):
        if not text or len(text.strip()) < 500:
            continue
        h = doc_hash(text)
        if h in _seen_hashes:
            continue
        if domain_kws and not keyword_density(text, domain_kws):
            continue
        _seen_hashes.add(h)
        (bucket_dir / f"{tag}_{start+n:06d}.txt").write_text(
            text.strip(), encoding="utf-8"
        )
        n += 1
    log.info(f"  Wrote {n} docs -> {bucket_dir.name}/{tag}_*.txt")
    return n

def safe_get(d, *keys, default=""):
    for k in keys:
        d = d.get(k, default) if isinstance(d, dict) else default
    return d or default

def gutenberg_strip(text):
    for m in ["*** START OF", "***START OF"]:
        if m in text:
            text = text[text.index(m):]
            text = text[text.index("\n") + 1:]
            break
    for m in ["*** END OF", "***END OF", "End of Project Gutenberg"]:
        if m in text:
            text = text[:text.index(m)]
    return text


# ── Source loaders ────────────────────────────────────────────────────────────

def wiki_api(queries, bucket_dir, tag, domain, subfield, domain_kws,
             max_per_query=25):
    API = "https://en.wikipedia.org/w/api.php"
    seen, all_chunks = set(), []
    for q in queries:
        try:
            hits = requests.get(API, params={
                "action": "query", "list": "search", "srsearch": q,
                "srlimit": max_per_query, "format": "json", "srnamespace": 0
            }, timeout=15).json().get("query", {}).get("search", [])
            for h in hits:
                t = h["title"]
                if t in seen:
                    continue
                seen.add(t)
                pages = requests.get(API, params={
                    "action": "query", "titles": t, "prop": "extracts",
                    "explaintext": True, "exsectionformat": "plain",
                    "format": "json"
                }, timeout=15).json().get("query", {}).get("pages", {})
                for pid, pg in pages.items():
                    if pid == "-1":
                        continue
                    text = pg.get("extract", "").strip()
                    if len(text) > 800:
                        all_chunks.extend(
                            chunk_text_smart(t, text, "Wikipedia", domain, subfield))
                        all_chunks.extend(
                            extract_glossary_terms(t, text, domain, subfield))
                time.sleep(0.12)
        except Exception as e:
            log.warning(f"    wiki_api '{q[:40]}': {e}")
        time.sleep(0.05)
    return write_docs(bucket_dir, all_chunks, tag, domain_kws)

def wiki_stream(keywords, bucket_dir, tag, domain, subfield, domain_kws,
                max_docs=5000):
    try:
        from datasets import load_dataset
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en",
                            split="train", streaming=True)
        kw = {k.lower() for k in keywords}
        all_chunks, scanned = [], 0
        for item in wiki:
            scanned += 1
            if scanned > 900_000:
                break
            title  = item.get("title", "")
            text   = item.get("text", "").strip()
            titlow = title.lower()
            if len(text) < 800:
                continue
            if any(k in titlow or k in text[:600].lower() for k in kw):
                all_chunks.extend(
                    chunk_text_smart(title, text, "Wikipedia-stream",
                                     domain, subfield))
                all_chunks.extend(
                    extract_glossary_terms(title, text, domain, subfield))
            if len(all_chunks) >= max_docs * 4:
                break
        return write_docs(bucket_dir, all_chunks, tag, domain_kws)
    except Exception as e:
        record_exclusion(f"wiki_stream:{tag}", str(e))
        return 0

def se_qa(domain_kws_se, bucket_dir, tag, domain, subfield, topic_kws,
          extra_kw=None, max_docs=5000):
    try:
        from datasets import load_dataset
        se = load_dataset("HuggingFaceH4/stack-exchange-preferences",
                          split="train", streaming=True)
        docs = []
        for i, item in enumerate(se):
            if i > 6_000_000:
                break
            dom = safe_get(item, "domain")
            if not any(k in dom.lower() for k in domain_kws_se):
                continue
            q = safe_get(item, "question")
            if extra_kw and not any(k in q.lower() for k in extra_kw):
                continue
            answers = item.get("answers", []) or []
            good = sorted(
                [a for a in answers
                 if (a.get("pm_score", 0) or 0) > 0
                 and len(a.get("text", "") or "") >= 500],
                key=lambda a: a.get("pm_score", 0), reverse=True
            )
            if not good:
                good = [a for a in answers
                        if len(a.get("text", "") or "") >= 500][:1]
            if good and len(q) > 80:
                tags = ", ".join(item.get("tags", []) or [])
                ans_block = "\n\n---\n\n".join(
                    f"Answer (score {a.get('pm_score', 0)}):\n"
                    f"{a.get('text', '').strip()[:4000]}"
                    for a in good[:3]
                )
                docs.append(
                    f"{context_header(domain, subfield, tags or dom, 'SE-QA')}\n\n"
                    f"[StackExchange/{dom}]\n\n"
                    f"Question: {q.strip()}\n\nTags: {tags}\n\n{ans_block}"
                )
            if len(docs) >= max_docs:
                break
        return write_docs(bucket_dir, docs, tag, topic_kws)
    except Exception as e:
        record_exclusion(f"se:{tag}", str(e))
        return 0

def arxiv_cats(cats, bucket_dir, tag, domain, subfield,
               extra_kw=None, max_docs=5000):
    try:
        from datasets import load_dataset
        arxiv = load_dataset("Cornell-University/arxiv",
                             split="train", streaming=True)
        cat_set = set(cats)
        docs = []
        for i, item in enumerate(arxiv):
            if i > 7_000_000:
                break
            c = set((item.get("categories", "") or "").split())
            if not c.intersection(cat_set):
                continue
            title = (item.get("title", "") or "").replace("\n", " ").strip()
            abst  = (item.get("abstract", "") or "").replace("\n", " ").strip()
            year  = item.get("update_date", "")[:4]
            if len(abst) < 150:
                continue
            if extra_kw:
                combo = (title + " " + abst).lower()
                if not any(k in combo for k in extra_kw):
                    continue
            docs.append(
                f"{context_header(domain, subfield, title[:60], 'Research-Abstract')}\n\n"
                f"[arXiv | {' '.join(sorted(c))} | {year}]\n\n"
                f"Title: {title}\n\nAbstract:\n{abst}"
            )
            if len(docs) >= max_docs:
                break
        return write_docs(bucket_dir, docs, tag)
    except Exception as e:
        record_exclusion(f"arxiv:{tag}", str(e))
        return 0

def survey_arxiv(cats, bucket_dir, tag, domain, subfield, max_docs=2500):
    SURVEY_KW = {"survey", "review", "tutorial", "overview", "introduction to",
                 "comprehensive", "systematic review", "state of the art",
                 "state-of-the-art", "primer", "advances in"}
    try:
        from datasets import load_dataset
        arxiv = load_dataset("Cornell-University/arxiv",
                             split="train", streaming=True)
        cat_set = set(cats)
        docs = []
        for i, item in enumerate(arxiv):
            if i > 7_000_000:
                break
            c = set((item.get("categories", "") or "").split())
            if not c.intersection(cat_set):
                continue
            title = (item.get("title", "") or "").replace("\n", " ").strip()
            abst  = (item.get("abstract", "") or "").replace("\n", " ").strip()
            year  = item.get("update_date", "")[:4]
            if not any(k in title.lower() for k in SURVEY_KW):
                continue
            if len(abst) < 150:
                continue
            docs.append(
                f"{context_header(domain, subfield, title[:60], 'Survey-Paper', 'Expert')}\n\n"
                f"[arXiv SURVEY | {' '.join(sorted(c))} | {year}]\n\n"
                f"Title: {title}\n\nAbstract:\n{abst}\n\n"
                f"[Note: This is a survey/review paper synthesizing broad field knowledge.]"
            )
            if len(docs) >= max_docs:
                break
        return write_docs(bucket_dir, docs, tag)
    except Exception as e:
        record_exclusion(f"survey_arxiv:{tag}", str(e))
        return 0

def pubmed(queries, bucket_dir, tag, domain, subfield, max_per_query=100):
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    docs = []
    for q in queries:
        try:
            search = requests.get(f"{BASE_URL}/esearch.fcgi", params={
                "db": "pubmed", "term": q, "retmax": max_per_query,
                "retmode": "json", "sort": "relevance"
            }, timeout=15).json()
            ids = search.get("esearchresult", {}).get("idlist", [])
            if not ids:
                continue
            fetch = requests.get(f"{BASE_URL}/efetch.fcgi", params={
                "db": "pubmed", "id": ",".join(ids),
                "rettype": "abstract", "retmode": "text"
            }, timeout=30)
            if fetch.status_code == 200:
                for block in fetch.text.split("\n\n\n"):
                    block = block.strip()
                    if len(block) > 400:
                        docs.append(
                            f"{context_header(domain, subfield, q[:50], 'PubMed-Abstract')}\n\n"
                            f"[Source: PubMed | Query: {q[:60]}]\n\n{block}"
                        )
            time.sleep(0.5)
        except Exception as e:
            log.warning(f"    PubMed '{q[:35]}': {e}")
    return write_docs(bucket_dir, docs, tag)

def pmc_fulltext(queries, bucket_dir, tag, domain, subfield, max_per_query=30):
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    docs = []
    for q in queries:
        try:
            search = requests.get(f"{BASE_URL}/esearch.fcgi", params={
                "db": "pmc", "term": q + " AND open access[filter]",
                "retmax": max_per_query, "retmode": "json", "sort": "relevance"
            }, timeout=15).json()
            ids = search.get("esearchresult", {}).get("idlist", [])
            if not ids:
                continue
            fetch = requests.get(f"{BASE_URL}/efetch.fcgi", params={
                "db": "pmc", "id": ",".join(ids),
                "rettype": "text", "retmode": "text"
            }, timeout=45)
            if fetch.status_code == 200 and len(fetch.text) > 500:
                articles = re.split(r'\n={10,}\n', fetch.text)
                for art in articles:
                    art = art.strip()
                    if len(art) > 600:
                        docs.extend(
                            chunk_text_smart(
                                f"PMC: {q[:50]}", art, "PMC-fulltext",
                                domain, subfield))
            time.sleep(0.6)
        except Exception as e:
            log.warning(f"    PMC '{q[:35]}': {e}")
    return write_docs(bucket_dir, docs, tag)

def semantic_scholar(queries, bucket_dir, tag, domain, subfield,
                     max_per_query=80, min_citations=5):
    docs = []
    for q in queries:
        try:
            r = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": q, "limit": max_per_query,
                    "fields": "title,abstract,year,citationCount,fieldsOfStudy"
                },
                timeout=20, headers={"User-Agent": "TitanAI/1.0"}
            )
            if r.status_code != 200:
                time.sleep(2)
                continue
            results = sorted(
                [p for p in r.json().get("data", [])
                 if (p.get("citationCount") or 0) >= min_citations],
                key=lambda p: p.get("citationCount") or 0, reverse=True
            )
            for p in results:
                title  = (p.get("title", "") or "").strip()
                abst   = (p.get("abstract", "") or "").strip()
                year   = p.get("year", "")
                cites  = p.get("citationCount", 0)
                fields = ", ".join(p.get("fieldsOfStudy", []) or [])
                if len(abst) > 200:
                    docs.append(
                        f"{context_header(domain, subfield, title[:60], 'Peer-Reviewed', 'Expert')}\n\n"
                        f"[Semantic Scholar | Fields: {fields} | Year: {year} | Citations: {cites}]\n\n"
                        f"Title: {title}\n\nAbstract:\n{abst}"
                    )
            time.sleep(1.2)
        except Exception as e:
            log.warning(f"    SemanticScholar '{q[:35]}': {e}")
    return write_docs(bucket_dir, docs, tag)

def openalex(queries, bucket_dir, tag, domain, subfield, label="Research"):
    docs = []
    for q in queries:
        try:
            r = requests.get(
                f"https://api.openalex.org/works?search={requests.utils.quote(q)}"
                "&per-page=100&filter=open_access.is_oa:true"
                "&select=title,abstract_inverted_index,publication_year,cited_by_count",
                timeout=20, headers={"User-Agent": "TitanAI/1.0"}
            )
            if r.status_code != 200:
                continue
            results = sorted(
                r.json().get("results", []),
                key=lambda w: w.get("cited_by_count") or 0, reverse=True
            )
            for w in results:
                title = (w.get("title", "") or "").strip()
                year  = w.get("publication_year", "")
                cites = w.get("cited_by_count", 0)
                inv   = w.get("abstract_inverted_index") or {}
                if not inv:
                    continue
                mx = max((p for ps in inv.values() for p in ps), default=-1)
                if mx < 0:
                    continue
                wl = [""] * (mx + 1)
                for word, ps in inv.items():
                    for p in ps:
                        wl[p] = word
                abst = " ".join(x for x in wl if x)
                if len(abst) > 150:
                    docs.append(
                        f"{context_header(domain, subfield, title[:60], 'OpenAlex-OA')}\n\n"
                        f"[OpenAlex | {label} | Year: {year} | Citations: {cites}]\n\n"
                        f"Title: {title}\n\nAbstract:\n{abst}"
                    )
        except Exception as e:
            log.warning(f"    OpenAlex '{q[:35]}': {e}")
        time.sleep(0.35)
    return write_docs(bucket_dir, docs, tag)

def gutenberg_fetch(ids, bucket_dir, tag, domain, subfield):
    all_chunks = []
    for gid, desc in ids:
        for url in [
            f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
            f"https://gutenberg.org/files/{gid}/{gid}-0.txt",
        ]:
            try:
                r = requests.get(url, timeout=40)
                if r.status_code == 200 and len(r.text) > 1000:
                    text = gutenberg_strip(r.text)
                    if len(text) > 500:
                        all_chunks.extend(
                            chunk_text_smart(desc, text, "Gutenberg",
                                             domain, subfield))
                    break
            except Exception as e:
                log.warning(f"    Gutenberg {gid}: {e}")
        time.sleep(0.5)
    return write_docs(bucket_dir, all_chunks, tag)


# ── Domain keyword sets ───────────────────────────────────────────────────────

CYBER_SYS_KW = {
    "network", "protocol", "firewall", "vpn", "router", "switch",
    "tcp", "ethernet", "vlan", "bgp", "ospf", "dns", "tls",
    "embedded", "firmware", "fpga", "microcontroller", "iot",
    "devsecops", "container", "kubernetes", "docker", "zero trust",
    "soc", "siem", "microsegmentation", "sdn", "nfv"
}

PENTEST_KW = {
    "penetration", "vulnerability", "exploit", "payload",
    "metasploit", "nmap", "reconnaissance", "enumeration",
    "privilege escalation", "lateral movement", "red team",
    "sql injection", "xss", "buffer overflow", "shellcode",
    "attack", "hacking", "ctf", "mitre"
}

MALWARE_KW = {
    "malware", "ransomware", "trojan", "rootkit", "botnet",
    "reverse engineering", "disassembly", "ghidra",
    "obfuscation", "evasion", "sandbox", "dynamic analysis",
    "static analysis", "yara", "ioc", "apt", "threat actor",
    "ida pro", "decompile"
}

CRYPTO_EXPLOIT_KW = {
    "cryptography", "encryption", "aes", "rsa", "elliptic curve",
    "hash function", "sha", "tls", "certificate", "pki",
    "exploitation", "rop", "heap spray", "use-after-free",
    "kernel exploit", "syscall", "privilege escalation",
    "cve", "aslr", "dep", "nx bit"
}

WEB_ARCH_KW = {
    "api", "rest", "graphql", "microservices", "distributed",
    "load balancer", "cdn", "caching", "redis", "kafka",
    "kubernetes", "docker", "serverless", "websocket",
    "database", "sql", "nosql", "sharding", "replication",
    "message queue", "event-driven", "cap theorem"
}

WEB_DEV_KW = {
    "javascript", "typescript", "react", "vue", "angular",
    "node", "python", "django", "flask", "postgresql",
    "html", "css", "frontend", "backend", "fullstack",
    "webpack", "browser", "dom", "cors", "oauth", "jwt",
    "web assembly", "pwa", "responsive"
}

ALGO_KW = {
    "algorithm", "data structure", "complexity", "big o",
    "sorting", "searching", "graph", "tree", "dynamic programming",
    "recursion", "hash table", "heap", "queue", "stack",
    "competitive programming", "binary search", "memoization",
    "greedy", "divide and conquer", "np-complete"
}

PROG_KW = {
    "programming language", "compiler", "interpreter", "runtime",
    "python", "rust", "c++", "java", "go", "functional",
    "object-oriented", "design pattern", "refactoring",
    "testing", "debugging", "profiling", "memory management",
    "garbage collection", "type system", "llvm"
}

EQ_KW = {
    "emotional intelligence", "empathy", "self-awareness",
    "self-regulation", "motivation", "social skills",
    "goleman", "mindfulness", "emotional regulation",
    "compassion", "active listening", "interpersonal",
    "resilience", "emotional literacy", "affect",
    "salovey", "mayer", "eq", "emotion"
}

BEHAV_KW = {
    "behavioural psychology", "conditioning", "reinforcement",
    "pavlov", "skinner", "cognitive bias", "heuristic",
    "operant conditioning", "classical conditioning",
    "behaviour modification", "cognitive behavioural therapy",
    "cbt", "aba", "applied behaviour analysis",
    "schema", "cognitive distortion", "bandura"
}

ICE_KW = {
    "engine", "combustion", "cylinder", "piston", "crankshaft",
    "camshaft", "valve", "turbocharger", "fuel injection",
    "carburetor", "torque", "horsepower", "transmission",
    "gearbox", "differential", "clutch", "brake", "suspension",
    "exhaust", "timing belt", "ignition"
}

EV_KW = {
    "electric motor", "battery", "bms", "inverter", "regenerative",
    "electric vehicle", "lithium ion", "permanent magnet",
    "induction motor", "motor controller", "powertrain",
    "range", "charging", "onboard charger", "ev", "tesla",
    "sic mosfet", "igbt", "brushless"
}

MACHINERY_KW = {
    "hydraulic", "pneumatic", "crane", "excavator", "bulldozer",
    "forklift", "loader", "heavy machinery", "construction",
    "agricultural", "tractor", "combine harvester",
    "plc", "automation", "safety", "operator",
    "control system", "actuator", "solenoid"
}

VEHICLE_KW = {
    "truck", "commercial vehicle", "hgv", "bus",
    "driving", "road safety", "logistics",
    "fleet", "cargo", "axle", "trailer",
    "brake system", "abs", "tachograph",
    "transport", "cdl", "articulated"
}

AVIATION_KW = {
    "aircraft", "airplane", "aerodynamics", "lift", "drag",
    "thrust", "wing", "airfoil", "avionics", "autopilot",
    "navigation", "atc", "runway", "landing gear",
    "turbofan", "flight control", "angle of attack",
    "stall", "mach", "pitot"
}

ROTORCRAFT_KW = {
    "helicopter", "rotor", "collective", "cyclic", "autorotation",
    "tail rotor", "blade", "hover", "vtol",
    "swashplate", "retreating blade stall",
    "vortex ring state", "ground effect", "turboshaft",
    "rotor wash", "tandem rotor", "coaxial"
}

MARINE_KW = {
    "yacht", "boat", "sailing", "hull", "keel", "rudder",
    "rigging", "sail", "buoyancy", "displacement",
    "marine engine", "outboard", "inboard", "diesel",
    "navigation", "colregs", "seamanship", "tide",
    "chart", "anchor", "marina", "knot"
}

HYDRO_KW = {
    "fluid mechanics", "hydraulics", "bernoulli", "reynolds",
    "viscosity", "turbulence", "laminar", "flow rate",
    "pressure", "pump", "turbine", "cavitation",
    "wave mechanics", "hydrodynamics", "drag", "lift",
    "navier-stokes", "continuity equation", "pipe flow",
    "vortex", "boundary layer"
}

CONFLICT_KW = {
    "negotiation", "conflict resolution", "mediation",
    "arbitration", "batna", "interest-based", "principled",
    "game theory", "diplomacy", "de-escalation",
    "active listening", "compromise", "win-win",
    "hostage negotiation", "restorative justice",
    "facilitation", "dispute", "settlement"
}

ETIQUETTE_KW = {
    "etiquette", "manners", "politeness", "courtesy",
    "table manners", "social norms", "protocol",
    "formal", "decorum", "civility", "respect",
    "greeting", "dress code", "hospitality",
    "business etiquette", "netiquette", "social skills"
}

AI_KW = {
    "artificial intelligence", "machine learning", "deep learning",
    "neural network", "transformer", "attention mechanism",
    "reinforcement learning", "natural language processing",
    "computer vision", "convolutional", "recurrent",
    "generative", "gan", "large language model", "llm",
    "bert", "gpt", "training", "inference"
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. ADVANCED CYBER SYSTEMS — Network Architecture & Protocol Design
# ══════════════════════════════════════════════════════════════════════════════
def load_cyber_systems_network():
    log.info("=" * 65)
    log.info("CYBER-SYS-1: Network Architecture & Protocol Design")
    log.info("=" * 65)
    D, SF = "Cyber Systems", "Network Architecture & Protocol Design"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "TCP/IP protocol stack OSI model network layers",
        "BGP border gateway protocol autonomous system routing",
        "OSPF link state routing algorithm Dijkstra SPF",
        "VLAN 802.1Q trunk port access port segmentation",
        "DNS DNSSEC domain name resolution security",
        "TLS 1.3 handshake certificate encryption forward secrecy",
        "VPN IPSec IKEv2 tunnel mode transport mode",
        "software-defined networking SDN OpenFlow controller",
        "network function virtualization NFV ETSI framework",
        "zero trust network architecture ZTNA microsegmentation",
        "firewall stateful packet inspection deep packet inspection",
        "intrusion detection system IDS Snort Suricata rules",
        "802.11ax Wi-Fi 6E OFDMA MU-MIMO spatial reuse",
        "5G network architecture RAN core network slicing",
        "IPv6 transition dual stack NAT64 6to4 tunneling",
        "MPLS traffic engineering label switched path",
        "Ethernet physical layer 10G 40G 100G fiber optic",
        "network topology star mesh bus ring hybrid design",
        "load balancer L4 L7 round robin health check",
        "content delivery network CDN edge caching Anycast",
        "QUIC HTTP/3 UDP multiplexing stream 0-RTT",
        "BGP hijacking route leak AS path manipulation",
        "network monitoring SNMP NetFlow sFlow traffic analysis",
        "DMZ architecture perimeter network dual firewall",
        "proxy server forward reverse transparent Squid",
        "network access control NAC 802.1X RADIUS EAP",
        "WireGuard VPN modern protocol cryptography performance",
        "VXLAN overlay network virtualization encapsulation",
        "SD-WAN branch connectivity policy QoS MPLS",
        "network time protocol NTP stratum clock synchronization",
    ]
    n = wiki_api(queries, bucket, "csys_net_wiki", D, SF, CYBER_SYS_KW); total += n
    log.info(f"  [CSYS-NET-1] Wiki API: {n}")
    n = wiki_stream(CYBER_SYS_KW, bucket, "csys_net_stream", D, SF,
                    CYBER_SYS_KW, max_docs=4000); total += n
    log.info(f"  [CSYS-NET-2] Wiki stream: {n}")
    n = se_qa(["networking", "security", "serverfault", "unix"],
              bucket, "csys_net_se", D, SF, CYBER_SYS_KW,
              extra_kw={"network", "tcp", "routing", "firewall", "vpn",
                        "vlan", "dns", "tls", "bgp", "ospf", "protocol",
                        "packet", "subnet", "nat", "iptables", "switch"}); total += n
    log.info(f"  [CSYS-NET-3] SE Q&A: {n}")
    n = arxiv_cats(["cs.NI", "cs.CR", "cs.DC"],
                   bucket, "csys_net_arxiv", D, SF,
                   extra_kw={"network", "protocol", "routing", "sdn",
                             "5g", "wireless", "congestion", "latency",
                             "security", "firewall", "vpn", "dns"},
                   max_docs=4000); total += n
    log.info(f"  [CSYS-NET-4] arXiv: {n}")
    n = survey_arxiv(["cs.NI", "cs.CR"],
                     bucket, "csys_net_survey", D, SF, max_docs=1500); total += n
    log.info(f"  [CSYS-NET-5] arXiv surveys: {n}")
    n = semantic_scholar([
        "software-defined networking SDN controller scalability performance",
        "zero trust architecture microsegmentation enterprise network",
        "5G network slicing resource management QoS",
        "BGP routing security hijacking detection mitigation",
        "TLS 1.3 performance analysis handshake latency",
        "Wi-Fi 6 802.11ax OFDMA throughput latency dense",
        "QUIC HTTP/3 web performance comparison TCP",
        "intrusion detection machine learning anomaly network traffic",
        "DNS over HTTPS privacy security performance",
        "network function virtualization NFV cloud performance",
        "IPv6 deployment adoption transition security challenges",
        "content delivery network CDN cache performance edge",
    ], bucket, "csys_net_ss", D, SF); total += n
    log.info(f"  [CSYS-NET-6] Semantic Scholar: {n}")
    n = openalex([
        "network protocol design architecture security",
        "software defined networking performance analysis",
        "5G wireless network resource management",
        "zero trust network security enterprise",
    ], bucket, "csys_net_oa", D, SF, "Network Systems"); total += n
    log.info(f"  [CSYS-NET-7] OpenAlex: {n}")
    log.info(f"  >>> Cyber Systems Network total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 2. ADVANCED CYBER SYSTEMS — Hardware Security, Embedded & DevSecOps
# ══════════════════════════════════════════════════════════════════════════════
def load_cyber_systems_hardware_devsecops():
    log.info("=" * 65)
    log.info("CYBER-SYS-2: Hardware Security, Embedded Systems & DevSecOps")
    log.info("=" * 65)
    D, SF = "Cyber Systems", "Hardware Security & DevSecOps"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "FPGA design Verilog VHDL synthesis place and route",
        "ARM Cortex-M microcontroller architecture bare metal",
        "RISC-V open ISA instruction set architecture",
        "secure boot chain of trust signature verification",
        "hardware security module HSM cryptographic key management",
        "trusted platform module TPM 2.0 attestation PCR",
        "side-channel attack power analysis timing attack",
        "differential power analysis DPA countermeasures",
        "fault injection glitching electromagnetic voltage",
        "IoT firmware security UART JTAG debug interface",
        "embedded Linux Yocto Buildroot kernel hardening",
        "RTOS real-time operating system FreeRTOS Zephyr",
        "CAN bus automotive OBD-II ECU communication",
        "DevSecOps shift left security CI/CD pipeline",
        "container security Docker image scanning vulnerability",
        "Kubernetes RBAC pod security policy network policy",
        "infrastructure as code Terraform Ansible security",
        "SAST static application security testing CodeQL",
        "DAST dynamic application security testing OWASP ZAP",
        "software composition analysis SCA dependency CVE",
        "supply chain security SLSA provenance attestation",
        "secrets management HashiCorp Vault environment variables",
        "security as code policy OPA conftest GitOps",
        "firmware reverse engineering binwalk Ghidra extraction",
        "hardware trojan detection IC supply chain integrity",
        "UEFI BIOS security secure boot measured boot",
        "PCIe DMA attack IOMMU protection memory isolation",
        "physical unclonable function PUF device fingerprint",
        "secure element NFC contactless smart card",
        "threat modeling STRIDE DREAD architecture review",
    ]
    n = wiki_api(queries, bucket, "csys_hw_wiki", D, SF, CYBER_SYS_KW); total += n
    log.info(f"  [CSYS-HW-1] Wiki API: {n}")
    n = wiki_stream({"fpga", "embedded", "firmware", "iot", "microcontroller",
                     "devsecops", "container", "kubernetes", "secure boot",
                     "hardware security", "side-channel", "tpm", "hsm"},
                    bucket, "csys_hw_stream", D, SF, CYBER_SYS_KW,
                    max_docs=4000); total += n
    log.info(f"  [CSYS-HW-2] Wiki stream: {n}")
    n = se_qa(["security", "embedded", "electronics", "unix", "stackoverflow"],
              bucket, "csys_hw_se", D, SF, CYBER_SYS_KW,
              extra_kw={"firmware", "embedded", "fpga", "microcontroller",
                        "iot", "secure boot", "hardware", "tpm", "hsm",
                        "docker", "kubernetes", "ci/cd", "devsecops",
                        "pipeline", "container", "ansible"}); total += n
    log.info(f"  [CSYS-HW-3] SE Q&A: {n}")
    n = arxiv_cats(["cs.CR", "cs.AR", "cs.SE", "eess.SY"],
                   bucket, "csys_hw_arxiv", D, SF,
                   extra_kw={"hardware security", "embedded", "fpga",
                             "side-channel", "iot", "firmware",
                             "devsecops", "container", "supply chain"},
                   max_docs=4000); total += n
    log.info(f"  [CSYS-HW-4] arXiv: {n}")
    n = survey_arxiv(["cs.CR", "cs.AR"],
                     bucket, "csys_hw_survey", D, SF, max_docs=1500); total += n
    log.info(f"  [CSYS-HW-5] arXiv surveys: {n}")
    n = semantic_scholar([
        "hardware security side-channel power analysis countermeasure",
        "IoT firmware vulnerability analysis static dynamic",
        "FPGA hardware trojan detection mitigation",
        "trusted execution environment TEE ARM TrustZone",
        "DevSecOps continuous integration security automation",
        "container security Docker Kubernetes runtime policy",
        "software supply chain attack dependency package",
        "SAST DAST security testing CI/CD pipeline shift-left",
        "secure boot UEFI TPM firmware integrity attestation",
        "physical unclonable function PUF authentication key",
        "fault injection attack microcontroller glitch",
        "infrastructure as code security misconfiguration cloud",
    ], bucket, "csys_hw_ss", D, SF); total += n
    log.info(f"  [CSYS-HW-6] Semantic Scholar: {n}")
    n = openalex([
        "embedded systems security hardware vulnerability",
        "IoT device security firmware analysis",
        "DevSecOps CI/CD pipeline security automation",
        "hardware security module cryptographic implementation",
    ], bucket, "csys_hw_oa", D, SF, "Hardware Security"); total += n
    log.info(f"  [CSYS-HW-7] OpenAlex: {n}")
    log.info(f"  >>> Cyber Systems Hardware/DevSecOps total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 3. ADVANCED CYBER SECURITY — Offensive Security & Penetration Testing
# ══════════════════════════════════════════════════════════════════════════════
def load_cyber_security_pentest():
    log.info("=" * 65)
    log.info("CYBER-SEC-1: Offensive Security & Penetration Testing")
    log.info("=" * 65)
    D, SF = "Cyber Security", "Offensive Security & Penetration Testing"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "penetration testing methodology PTES OWASP NIST",
        "OWASP Top 10 web application vulnerabilities 2021",
        "SQL injection union-based blind time-based error",
        "cross-site scripting XSS reflected stored DOM",
        "SSRF server-side request forgery cloud metadata",
        "insecure deserialization Java PHP gadget chain",
        "XML external entity XXE SSRF blind OOB",
        "broken authentication session management JWT",
        "CSRF cross-site request forgery token bypass",
        "directory traversal path traversal LFI RFI",
        "buffer overflow stack heap overwrite EIP",
        "format string vulnerability printf FSB attack",
        "privilege escalation Linux SUID sudo PATH",
        "Windows privilege escalation unquoted service path",
        "pass the hash NTLM relay responder poisoning",
        "Kerberoasting AS-REP roasting golden ticket",
        "Active Directory enumeration BloodHound SharpHound",
        "lateral movement PsExec WMI SMB credential reuse",
        "command and control C2 Cobalt Strike Empire Sliver",
        "metasploit framework payload meterpreter post-exploit",
        "Burp Suite proxy scanner intruder repeater",
        "nmap port scanning OS detection service version",
        "social engineering phishing spear phishing vishing",
        "MITRE ATT&CK framework tactics techniques procedures",
        "Cyber Kill Chain Lockheed Martin reconnaissance",
        "red team operations purple team adversary simulation",
        "living off the land LOLBins fileless attack",
        "web application firewall WAF bypass evasion",
        "password cracking hashcat dictionary brute force",
        "recon-ng OSINT framework reconnaissance automation",
    ]
    n = wiki_api(queries, bucket, "csec_pt_wiki", D, SF, PENTEST_KW); total += n
    log.info(f"  [CSEC-PT-1] Wiki API: {n}")
    n = wiki_stream(PENTEST_KW, bucket, "csec_pt_stream", D, SF,
                    PENTEST_KW, max_docs=4000); total += n
    log.info(f"  [CSEC-PT-2] Wiki stream: {n}")
    n = se_qa(["security", "reverseengineering"],
              bucket, "csec_pt_se", D, SF, PENTEST_KW,
              extra_kw={"penetration", "exploit", "vulnerability", "xss",
                        "sql injection", "buffer overflow", "privilege",
                        "metasploit", "nmap", "kerberos", "active directory",
                        "red team", "payload", "shellcode", "bypass"}); total += n
    log.info(f"  [CSEC-PT-3] SE Q&A: {n}")
    n = arxiv_cats(["cs.CR"],
                   bucket, "csec_pt_arxiv", D, SF,
                   extra_kw={"penetration testing", "vulnerability", "exploit",
                             "attack", "offensive", "red team", "web security",
                             "fuzzing", "injection", "privilege escalation"},
                   max_docs=5000); total += n
    log.info(f"  [CSEC-PT-4] arXiv: {n}")
    n = survey_arxiv(["cs.CR"],
                     bucket, "csec_pt_survey", D, SF, max_docs=2000); total += n
    log.info(f"  [CSEC-PT-5] arXiv surveys: {n}")
    n = semantic_scholar([
        "automated penetration testing machine learning vulnerability",
        "SQL injection detection prevention web application",
        "cross-site scripting XSS detection mitigation browser",
        "Active Directory attack Kerberos ticket Golden Silver",
        "fuzzing coverage guided AFL libFuzzer vulnerability",
        "MITRE ATT&CK technique detection hunting SIEM",
        "phishing detection machine learning email URL",
        "privilege escalation Linux Windows kernel exploit",
        "web application firewall bypass evasion machine learning",
        "lateral movement detection endpoint network behavior",
        "password cracking GPU hash rainbow table dictionary",
        "social engineering human factor security awareness",
    ], bucket, "csec_pt_ss", D, SF); total += n
    log.info(f"  [CSEC-PT-6] Semantic Scholar: {n}")
    n = openalex([
        "penetration testing vulnerability assessment security",
        "web application security OWASP attack defense",
        "network intrusion detection offensive techniques",
        "malware analysis reverse engineering techniques",
    ], bucket, "csec_pt_oa", D, SF, "Offensive Security"); total += n
    log.info(f"  [CSEC-PT-7] OpenAlex: {n}")
    log.info(f"  >>> Cyber Security Pentest total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 4. ADVANCED CYBER SECURITY — Malware Analysis & Reverse Engineering
# ══════════════════════════════════════════════════════════════════════════════
def load_cyber_security_malware():
    log.info("=" * 65)
    log.info("CYBER-SEC-2: Malware Analysis & Reverse Engineering")
    log.info("=" * 65)
    D, SF = "Cyber Security", "Malware Analysis & Reverse Engineering"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "malware classification taxonomy trojan worm virus",
        "ransomware encryption key management Bitcoin payment",
        "WannaCry EternalBlue SMB exploit propagation",
        "NotPetya Petya destructive wiper MBR overwrite",
        "rootkit kernel mode user mode hook DKOM",
        "bootkit MBR VBR infection persistence UEFI",
        "APT advanced persistent threat nation-state espionage",
        "Stuxnet SCADA ICS industrial malware PLC",
        "Emotet banking trojan malspam botnet loader",
        "TrickBot modular malware credential theft",
        "IDA Pro disassembly decompilation Hex-Rays",
        "Ghidra reverse engineering NSA open source",
        "dynamic analysis sandbox Cuckoo Any.run behavioral",
        "static analysis PE header import table strings",
        "YARA rule writing malware family signature",
        "indicators of compromise IOC hash domain IP",
        "anti-analysis anti-VM detection evasion technique",
        "code obfuscation packing UPX custom packer",
        "fileless malware living-off-the-land PowerShell",
        "process injection DLL hollowing process hollowing",
        "reflective DLL injection LoadLibrary memory",
        "lateral tool transfer PowerShell WMI BITS",
        "command and control beacon domain generation algorithm",
        "polymorphic metamorphic virus code mutation",
        "threat intelligence MISP OpenCTI sharing platform",
        "malware deobfuscation unpacking Python scripting",
        "mobile malware Android APK smali Dalvik",
        "macOS malware dylib hijacking LaunchAgent",
        "firmware malware UEFI persistence implant",
        "memory forensics Volatility process artifact",
    ]
    n = wiki_api(queries, bucket, "csec_mal_wiki", D, SF, MALWARE_KW); total += n
    log.info(f"  [CSEC-MAL-1] Wiki API: {n}")
    n = wiki_stream(MALWARE_KW, bucket, "csec_mal_stream", D, SF,
                    MALWARE_KW, max_docs=4000); total += n
    log.info(f"  [CSEC-MAL-2] Wiki stream: {n}")
    n = se_qa(["security", "reverseengineering"],
              bucket, "csec_mal_se", D, SF, MALWARE_KW,
              extra_kw={"malware", "ransomware", "rootkit", "reverse",
                        "ida", "ghidra", "disassemble", "decompile",
                        "sandbox", "yara", "apt", "botnet", "trojan",
                        "obfuscation", "unpacking", "shellcode"}); total += n
    log.info(f"  [CSEC-MAL-3] SE Q&A: {n}")
    n = arxiv_cats(["cs.CR"],
                   bucket, "csec_mal_arxiv", D, SF,
                   extra_kw={"malware", "ransomware", "detection",
                             "classification", "reverse engineering",
                             "botnet", "apt", "evasion", "analysis"},
                   max_docs=5000); total += n
    log.info(f"  [CSEC-MAL-4] arXiv: {n}")
    n = survey_arxiv(["cs.CR"],
                     bucket, "csec_mal_survey", D, SF, max_docs=1500); total += n
    log.info(f"  [CSEC-MAL-5] arXiv surveys: {n}")
    n = semantic_scholar([
        "malware detection machine learning deep learning features",
        "ransomware detection prevention encryption behavior",
        "Android malware analysis static dynamic classification",
        "botnet detection traffic analysis peer-to-peer",
        "APT detection threat intelligence lateral movement",
        "fileless malware detection PowerShell memory forensics",
        "code obfuscation deobfuscation program analysis",
        "sandbox evasion anti-analysis malware detection",
        "malware family classification clustering graph neural",
        "vulnerability exploitation memory corruption ROP heap",
        "threat intelligence sharing IOC STIX TAXII",
        "adversarial malware evasion anti-virus ML bypass",
    ], bucket, "csec_mal_ss", D, SF); total += n
    log.info(f"  [CSEC-MAL-6] Semantic Scholar: {n}")
    n = openalex([
        "malware detection analysis machine learning",
        "ransomware encryption detection prevention",
        "reverse engineering binary analysis vulnerability",
        "threat intelligence attribution APT campaign",
    ], bucket, "csec_mal_oa", D, SF, "Malware Research"); total += n
    log.info(f"  [CSEC-MAL-7] OpenAlex: {n}")
    log.info(f"  >>> Cyber Security Malware total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 5. ADVANCED CYBER SECURITY — Cryptography & OS Exploitation
# ══════════════════════════════════════════════════════════════════════════════
def load_cyber_security_crypto_exploit():
    log.info("=" * 65)
    log.info("CYBER-SEC-3: Cryptography & OS Exploitation")
    log.info("=" * 65)
    D, SF = "Cyber Security", "Cryptography & OS Exploitation"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "AES advanced encryption standard block cipher modes",
        "RSA public key cryptography factoring modular exponentiation",
        "elliptic curve cryptography ECDSA ECDH secp256k1",
        "Diffie-Hellman key exchange discrete logarithm",
        "SHA-256 SHA-3 hash function collision resistance",
        "HMAC message authentication code integrity",
        "X.509 certificate PKI chain of trust CA",
        "TLS handshake certificate pinning HSTS",
        "zero-knowledge proof zk-SNARK zk-STARK",
        "homomorphic encryption FHE computation on ciphertext",
        "post-quantum cryptography lattice NIST CRYSTALS",
        "Diffie-Hellman logjam FREAK BEAST POODLE attack",
        "heap overflow exploitation ptmalloc tcache bin",
        "use-after-free exploitation type confusion browser",
        "return-oriented programming ROP chain gadget",
        "stack buffer overflow SEH overwrite canary bypass",
        "ASLR bypass information leak entropy reduction",
        "kernel exploitation LPE privilege escalation Linux",
        "Windows kernel exploit token stealing ring0",
        "browser exploitation V8 JIT SpiderMonkey sandbox",
        "CVE common vulnerabilities exposures CVSS scoring",
        "fuzzing AFL++ libFuzzer coverage guided feedback",
        "symbolic execution angr Manticore constraint solving",
        "race condition TOCTOU time of check use",
        "integer overflow underflow sign extension wrap",
        "format string vulnerability stack arbitrary write",
        "null pointer dereference kernel panic crash",
        "heap spray exploitation JIT spray BSTR",
        "memory safe language Rust ownership borrow",
        "exploit mitigation CFI shadow stack CET Intel",
    ]
    n = wiki_api(queries, bucket, "csec_cry_wiki", D, SF, CRYPTO_EXPLOIT_KW); total += n
    log.info(f"  [CSEC-CRY-1] Wiki API: {n}")
    n = wiki_stream(CRYPTO_EXPLOIT_KW, bucket, "csec_cry_stream", D, SF,
                    CRYPTO_EXPLOIT_KW, max_docs=4000); total += n
    log.info(f"  [CSEC-CRY-2] Wiki stream: {n}")
    n = se_qa(["security", "cryptography", "reverseengineering"],
              bucket, "csec_cry_se", D, SF, CRYPTO_EXPLOIT_KW,
              extra_kw={"aes", "rsa", "elliptic", "hash", "tls",
                        "certificate", "encryption", "exploit",
                        "rop", "heap", "kernel", "privilege",
                        "overflow", "aslr", "dep", "bypass"}); total += n
    log.info(f"  [CSEC-CRY-3] SE Q&A: {n}")
    n = arxiv_cats(["cs.CR", "cs.IT"],
                   bucket, "csec_cry_arxiv", D, SF,
                   extra_kw={"cryptography", "encryption", "hash",
                             "protocol", "exploit", "kernel",
                             "vulnerability", "post-quantum"},
                   max_docs=5000); total += n
    log.info(f"  [CSEC-CRY-4] arXiv: {n}")
    n = survey_arxiv(["cs.CR", "cs.IT"],
                     bucket, "csec_cry_survey", D, SF, max_docs=1500); total += n
    log.info(f"  [CSEC-CRY-5] arXiv surveys: {n}")
    n = semantic_scholar([
        "post-quantum cryptography lattice signature NIST standardization",
        "TLS protocol analysis vulnerability attack BEAST POODLE",
        "heap exploitation tcache bin glibc allocator",
        "return-oriented programming ROP defense CFI",
        "kernel privilege escalation exploit mitigation",
        "fuzzing vulnerability discovery coverage guided",
        "symbolic execution constraint solving vulnerability",
        "side-channel attack cache timing AES",
        "browser JavaScript engine JIT exploit V8",
        "zero-knowledge proof privacy blockchain zk-SNARK",
        "homomorphic encryption practical scheme FHE performance",
        "memory safety Rust ownership exploit prevention",
    ], bucket, "csec_cry_ss", D, SF); total += n
    log.info(f"  [CSEC-CRY-6] Semantic Scholar: {n}")
    n = openalex([
        "applied cryptography protocol security analysis",
        "software vulnerability exploitation mitigation",
        "post-quantum cryptography algorithm implementation",
        "OS kernel security exploit privilege escalation",
    ], bucket, "csec_cry_oa", D, SF, "Cryptography & Exploitation"); total += n
    log.info(f"  [CSEC-CRY-7] OpenAlex: {n}")
    log.info(f"  >>> Cyber Security Crypto/Exploit total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 6. ADVANCED WEB SYSTEMS — Architecture, APIs & Distributed Systems
# ══════════════════════════════════════════════════════════════════════════════
def load_web_systems_architecture():
    log.info("=" * 65)
    log.info("WEB-SYS-1: Web Architecture, APIs & Distributed Systems")
    log.info("=" * 65)
    D, SF = "Web Systems", "Architecture & Distributed Systems"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "microservices architecture decomposition domain-driven design",
        "RESTful API design principles HATEOAS versioning",
        "GraphQL schema resolver mutation subscription",
        "gRPC protocol buffers streaming bidirectional",
        "event-driven architecture pub/sub event sourcing",
        "CQRS command query responsibility segregation",
        "API gateway rate limiting authentication JWT",
        "service mesh Istio Envoy proxy sidecar pattern",
        "distributed systems CAP theorem consistency availability",
        "eventual consistency saga pattern distributed transaction",
        "Apache Kafka message queue topic partition consumer",
        "RabbitMQ AMQP exchange routing key queue binding",
        "Redis in-memory cache pub/sub data structures",
        "Elasticsearch full-text search indexing mapping",
        "PostgreSQL ACID transactions indexing query planning",
        "database sharding horizontal partitioning routing",
        "database replication master-slave primary-replica",
        "Kubernetes pod deployment service ingress HPA",
        "Docker containerization image layer registry",
        "serverless Lambda function as a service cold start",
        "load balancing sticky session health check circuit breaker",
        "content delivery network CDN cache invalidation purge",
        "WebSocket real-time bidirectional server push",
        "OAuth 2.0 OpenID Connect authorization code PKCE",
        "rate limiting throttling token bucket leaky bucket",
        "circuit breaker pattern resilience Hystrix Resilience4j",
        "distributed tracing OpenTelemetry Jaeger Zipkin",
        "monitoring observability metrics logs traces",
        "chaos engineering fault injection Chaos Monkey",
        "twelve-factor app methodology cloud-native principles",
    ]
    n = wiki_api(queries, bucket, "web_arch_wiki", D, SF, WEB_ARCH_KW); total += n
    log.info(f"  [WEB-ARCH-1] Wiki API: {n}")
    n = wiki_stream(WEB_ARCH_KW, bucket, "web_arch_stream", D, SF,
                    WEB_ARCH_KW, max_docs=4000); total += n
    log.info(f"  [WEB-ARCH-2] Wiki stream: {n}")
    n = se_qa(["stackoverflow", "softwareengineering", "devops", "dba"],
              bucket, "web_arch_se", D, SF, WEB_ARCH_KW,
              extra_kw={"microservices", "api", "rest", "graphql",
                        "kafka", "redis", "kubernetes", "docker",
                        "load balancer", "distributed", "sharding",
                        "caching", "event driven", "saga", "cqrs",
                        "cap theorem", "eventual consistency"}); total += n
    log.info(f"  [WEB-ARCH-3] SE Q&A: {n}")
    n = arxiv_cats(["cs.DC", "cs.SE", "cs.NI"],
                   bucket, "web_arch_arxiv", D, SF,
                   extra_kw={"microservices", "distributed", "api",
                             "serverless", "kubernetes", "consistency",
                             "performance", "scalability", "fault"},
                   max_docs=4000); total += n
    log.info(f"  [WEB-ARCH-4] arXiv: {n}")
    n = survey_arxiv(["cs.DC", "cs.SE"],
                     bucket, "web_arch_survey", D, SF, max_docs=1500); total += n
    log.info(f"  [WEB-ARCH-5] arXiv surveys: {n}")
    n = semantic_scholar([
        "microservices architecture decomposition performance scalability",
        "Kubernetes container orchestration scheduling resource",
        "distributed database consistency replication partition",
        "event sourcing CQRS eventual consistency implementation",
        "API gateway rate limiting security authentication",
        "service mesh Istio Envoy observability traffic management",
        "serverless computing performance cold start FaaS",
        "Apache Kafka stream processing throughput latency",
        "distributed tracing observability microservices",
        "chaos engineering resilience fault injection",
        "GraphQL performance REST comparison adoption",
        "CAP theorem BASE ACID distributed trade-off",
    ], bucket, "web_arch_ss", D, SF); total += n
    log.info(f"  [WEB-ARCH-6] Semantic Scholar: {n}")
    n = openalex([
        "microservices distributed systems architecture cloud",
        "API design REST GraphQL performance comparison",
        "Kubernetes orchestration container scheduling",
        "distributed database consistency availability tradeoff",
    ], bucket, "web_arch_oa", D, SF, "Web Architecture"); total += n
    log.info(f"  [WEB-ARCH-7] OpenAlex: {n}")
    log.info(f"  >>> Web Systems Architecture total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 7. ADVANCED WEB SYSTEMS — Frontend & Backend Engineering
# ══════════════════════════════════════════════════════════════════════════════
def load_web_systems_development():
    log.info("=" * 65)
    log.info("WEB-SYS-2: Frontend & Backend Engineering")
    log.info("=" * 65)
    D, SF = "Web Systems", "Frontend & Backend Engineering"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "JavaScript ES2023 modules async await promise",
        "TypeScript type system generics decorators",
        "React hooks useState useEffect context API",
        "Next.js server-side rendering static generation ISR",
        "Vue.js composition API Pinia reactive system",
        "Angular dependency injection zone.js RxJS",
        "Node.js event loop libuv non-blocking I/O",
        "Express.js middleware routing error handling",
        "Django REST framework serializer viewset",
        "FastAPI async Python type hints OpenAPI",
        "PostgreSQL query optimization EXPLAIN ANALYZE",
        "MongoDB aggregation pipeline index Atlas",
        "Redis data structures expiry eviction policy",
        "Webpack module bundling code splitting tree shaking",
        "Vite ES modules HMR fast dev build",
        "CSS Grid Flexbox responsive layout media query",
        "Web accessibility ARIA WCAG 2.1 screen reader",
        "CORS cross-origin resource sharing preflight",
        "OAuth 2.0 JWT refresh token session management",
        "WebAssembly WASM binary format memory linear",
        "Progressive Web App service worker cache manifest",
        "browser rendering critical rendering path paint",
        "virtual DOM reconciliation diffing React",
        "CSS-in-JS styled-components emotion performance",
        "GraphQL N+1 problem DataLoader batching",
        "SQL query optimization index B-tree covering",
        "database connection pooling pgBouncer transaction",
        "API versioning strategy URL header content negotiation",
        "web performance LCP FID CLS Core Web Vitals",
        "HTTP/2 multiplexing header compression server push",
    ]
    n = wiki_api(queries, bucket, "web_dev_wiki", D, SF, WEB_DEV_KW); total += n
    log.info(f"  [WEB-DEV-1] Wiki API: {n}")
    n = wiki_stream(WEB_DEV_KW, bucket, "web_dev_stream", D, SF,
                    WEB_DEV_KW, max_docs=4000); total += n
    log.info(f"  [WEB-DEV-2] Wiki stream: {n}")
    n = se_qa(["stackoverflow", "webmasters", "softwareengineering",
               "codereview"],
              bucket, "web_dev_se", D, SF, WEB_DEV_KW,
              extra_kw={"javascript", "typescript", "react", "vue",
                        "angular", "node", "python", "django",
                        "postgresql", "css", "html", "api",
                        "webpack", "browser", "performance", "oauth"}); total += n
    log.info(f"  [WEB-DEV-3] SE Q&A: {n}")
    n = arxiv_cats(["cs.SE", "cs.PL", "cs.HC"],
                   bucket, "web_dev_arxiv", D, SF,
                   extra_kw={"web", "javascript", "framework",
                             "performance", "browser", "frontend",
                             "backend", "database", "orm"},
                   max_docs=3000); total += n
    log.info(f"  [WEB-DEV-4] arXiv: {n}")
    n = semantic_scholar([
        "JavaScript framework React Vue Angular performance comparison",
        "TypeScript static typing developer productivity bug reduction",
        "web performance optimization Core Web Vitals LCP",
        "progressive web app service worker offline capability",
        "Node.js performance scalability event loop async",
        "PostgreSQL query optimization indexing performance",
        "GraphQL vs REST API performance complexity tradeoff",
        "web accessibility WCAG compliance screen reader testing",
        "WebAssembly performance near-native browser compute",
        "CSS architecture maintainability BEM methodology",
        "OAuth OpenID Connect security implementation best practice",
        "HTTP/2 HTTP/3 QUIC web performance comparison",
    ], bucket, "web_dev_ss", D, SF); total += n
    log.info(f"  [WEB-DEV-5] Semantic Scholar: {n}")
    n = openalex([
        "web development framework JavaScript TypeScript",
        "frontend performance optimization browser rendering",
        "backend API database scalability architecture",
        "progressive web application service worker",
    ], bucket, "web_dev_oa", D, SF, "Web Engineering"); total += n
    log.info(f"  [WEB-DEV-6] OpenAlex: {n}")
    log.info(f"  >>> Web Systems Development total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 8. ADVANCED CODING — Algorithms & Data Structures
# ══════════════════════════════════════════════════════════════════════════════
def load_coding_algorithms():
    log.info("=" * 65)
    log.info("CODE-1: Algorithms, Data Structures & Competitive Programming")
    log.info("=" * 65)
    D, SF = "Advanced Coding", "Algorithms & Data Structures"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "binary search algorithm sorted array mid pivot",
        "merge sort divide and conquer stable O(n log n)",
        "quicksort pivot partition lomuto hoare scheme",
        "heapsort binary heap max-heap sift down",
        "radix sort counting sort bucket sort linear time",
        "BFS breadth-first search queue level order",
        "DFS depth-first search stack topological sort",
        "Dijkstra shortest path priority queue relaxation",
        "Bellman-Ford negative edge shortest path SSSP",
        "A* heuristic algorithm admissible consistent",
        "Floyd-Warshall all-pairs shortest path DP",
        "Prim Kruskal minimum spanning tree greedy",
        "dynamic programming memoization tabulation",
        "knapsack problem 0-1 fractional branch bound",
        "longest common subsequence edit distance DP",
        "red-black tree balanced BST rotation recolor",
        "AVL tree height balanced rotation insertion",
        "B-tree B+ tree database index multiway",
        "segment tree range query lazy propagation",
        "Fenwick tree binary indexed tree prefix sum",
        "union-find disjoint set path compression rank",
        "trie prefix tree string search autocomplete",
        "hash table chaining open addressing probing",
        "bloom filter probabilistic set membership",
        "LRU cache least recently used eviction O(1)",
        "two pointers sliding window technique string",
        "NP-complete NP-hard reduction polynomial",
        "P vs NP Millennium Prize computational complexity",
        "amortized analysis aggregate accounting potential",
        "competitive programming LeetCode Codeforces",
    ]
    n = wiki_api(queries, bucket, "code_algo_wiki", D, SF, ALGO_KW); total += n
    log.info(f"  [CODE-ALGO-1] Wiki API: {n}")
    n = wiki_stream(ALGO_KW, bucket, "code_algo_stream", D, SF,
                    ALGO_KW, max_docs=4000); total += n
    log.info(f"  [CODE-ALGO-2] Wiki stream: {n}")
    n = se_qa(["stackoverflow", "cs", "codereview", "softwareengineering"],
              bucket, "code_algo_se", D, SF, ALGO_KW,
              extra_kw={"algorithm", "data structure", "complexity",
                        "sorting", "graph", "tree", "dynamic programming",
                        "recursion", "hash", "binary search",
                        "big o", "time complexity", "space complexity",
                        "greedy", "dp", "bfs", "dfs"}); total += n
    log.info(f"  [CODE-ALGO-3] SE Q&A: {n}")
    n = arxiv_cats(["cs.DS", "cs.CC", "cs.CG"],
                   bucket, "code_algo_arxiv", D, SF,
                   extra_kw={"algorithm", "data structure", "complexity",
                             "graph", "tree", "sorting", "approximation"},
                   max_docs=4000); total += n
    log.info(f"  [CODE-ALGO-4] arXiv: {n}")
    n = survey_arxiv(["cs.DS", "cs.CC"],
                     bucket, "code_algo_survey", D, SF, max_docs=1500); total += n
    log.info(f"  [CODE-ALGO-5] arXiv surveys: {n}")
    n = semantic_scholar([
        "graph neural network algorithm combinatorial optimization",
        "approximation algorithm NP-hard greedy local search",
        "randomized algorithm Monte Carlo Las Vegas probability",
        "online algorithm competitive ratio streaming algorithm",
        "parallel algorithm PRAM distributed computation",
        "cache-oblivious algorithm memory hierarchy locality",
        "string matching KMP Rabin-Karp suffix array",
        "geometric algorithm convex hull Voronoi Delaunay",
        "dynamic programming optimization substructure",
        "network flow max-flow min-cut Ford-Fulkerson",
        "data structure augmentation interval tree range",
        "hash function universal collision resistance",
    ], bucket, "code_algo_ss", D, SF); total += n
    log.info(f"  [CODE-ALGO-6] Semantic Scholar: {n}")
    n = openalex([
        "algorithm design analysis complexity optimization",
        "data structure performance implementation comparison",
        "graph algorithm shortest path network flow",
        "dynamic programming optimization problem solving",
    ], bucket, "code_algo_oa", D, SF, "Algorithm Research"); total += n
    log.info(f"  [CODE-ALGO-7] OpenAlex: {n}")
    log.info(f"  >>> Coding Algorithms total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 9. ADVANCED CODING — Programming Languages, Compilers & Architecture
# ══════════════════════════════════════════════════════════════════════════════
def load_coding_languages_architecture():
    log.info("=" * 65)
    log.info("CODE-2: Programming Languages, Compilers & Software Architecture")
    log.info("=" * 65)
    D, SF = "Advanced Coding", "Languages, Compilers & Architecture"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "compiler design lexical analysis tokenizer regex",
        "syntax analysis parser LL LR LALR grammar",
        "abstract syntax tree AST semantic analysis",
        "intermediate representation IR SSA form",
        "LLVM backend code generation optimization passes",
        "register allocation graph coloring spilling",
        "garbage collection mark-sweep reference counting",
        "Rust ownership borrow checker lifetime safety",
        "Python CPython GIL interpreter bytecode",
        "JVM bytecode JIT HotSpot performance",
        "functional programming Haskell monad functor",
        "type theory dependent types Hindley-Milner",
        "object-oriented SOLID principles cohesion coupling",
        "design patterns Gang of Four creational structural",
        "SOLID single responsibility open closed Liskov",
        "clean architecture hexagonal ports adapters",
        "domain-driven design bounded context aggregate",
        "test-driven development TDD red green refactor",
        "behavior-driven development BDD Cucumber Gherkin",
        "code smell technical debt refactoring",
        "concurrency parallelism race condition mutex",
        "async programming coroutines event loop",
        "memory management stack heap malloc free",
        "profiling performance bottleneck flamegraph",
        "static analysis linting type checking mypy",
        "metaprogramming reflection macros code generation",
        "WebAssembly Rust compilation target WASM",
        "Go goroutine channel concurrent CSP model",
        "Rust trait generic monomorphization zero-cost",
        "C++ template metaprogramming constexpr SFINAE",
    ]
    n = wiki_api(queries, bucket, "code_lang_wiki", D, SF, PROG_KW); total += n
    log.info(f"  [CODE-LANG-1] Wiki API: {n}")
    n = wiki_stream(PROG_KW, bucket, "code_lang_stream", D, SF,
                    PROG_KW, max_docs=4000); total += n
    log.info(f"  [CODE-LANG-2] Wiki stream: {n}")
    n = se_qa(["stackoverflow", "softwareengineering", "codereview",
               "programmers"],
              bucket, "code_lang_se", D, SF, PROG_KW,
              extra_kw={"compiler", "parser", "type system", "garbage",
                        "rust", "haskell", "functional", "design pattern",
                        "solid", "clean", "refactor", "tdd", "llvm",
                        "jit", "concurrency", "async"}); total += n
    log.info(f"  [CODE-LANG-3] SE Q&A: {n}")
    n = arxiv_cats(["cs.PL", "cs.SE"],
                   bucket, "code_lang_arxiv", D, SF,
                   extra_kw={"programming language", "compiler", "type",
                             "analysis", "verification", "synthesis",
                             "concurrency", "memory"},
                   max_docs=4000); total += n
    log.info(f"  [CODE-LANG-4] arXiv: {n}")
    n = survey_arxiv(["cs.PL", "cs.SE"],
                     bucket, "code_lang_survey", D, SF, max_docs=1500); total += n
    log.info(f"  [CODE-LANG-5] arXiv surveys: {n}")
    n = semantic_scholar([
        "Rust ownership memory safety systems programming",
        "type inference Hindley-Milner polymorphism",
        "LLVM compiler infrastructure optimization backend",
        "garbage collection pause latency throughput GC",
        "software architecture microservices clean hexagonal",
        "technical debt code smell refactoring detection",
        "test-driven development effectiveness productivity study",
        "concurrency bugs race condition deadlock detection",
        "program synthesis specification constraint neural",
        "static analysis bug detection precision recall",
        "JIT compilation V8 HotSpot performance optimization",
        "domain-specific language DSL embedding host",
    ], bucket, "code_lang_ss", D, SF); total += n
    log.info(f"  [CODE-LANG-6] Semantic Scholar: {n}")
    n = openalex([
        "programming language design type system implementation",
        "compiler optimization code generation performance",
        "software architecture patterns quality metrics",
        "static analysis program verification correctness",
    ], bucket, "code_lang_oa", D, SF, "Programming Languages"); total += n
    log.info(f"  [CODE-LANG-7] OpenAlex: {n}")
    log.info(f"  >>> Coding Languages/Architecture total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 10. EMOTIONAL INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
def load_emotional_intelligence():
    log.info("=" * 65)
    log.info("EQ-1: Emotional Intelligence — Full Spectrum")
    log.info("=" * 65)
    D, SF = "Emotional Intelligence", "EQ Theory & Applied Practice"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "emotional intelligence Daniel Goleman five components",
        "Salovey Mayer emotional intelligence model four branch",
        "self-awareness emotional introspection metacognition",
        "self-regulation impulse control emotional discipline",
        "intrinsic motivation achievement drive optimism",
        "empathy cognitive affective compassionate types",
        "social skills communication rapport influence",
        "amygdala hijack emotional flooding prefrontal cortex",
        "limbic system emotion brain structure function",
        "emotional regulation strategies reappraisal suppression",
        "mindfulness meditation emotion regulation practice",
        "active listening paraphrasing reflection empathy",
        "nonviolent communication NVC needs feelings",
        "emotional labeling affect labeling HALT technique",
        "emotional intelligence workplace leadership outcomes",
        "psychological safety team performance Amy Edmondson",
        "empathy map design thinking user experience",
        "attachment theory secure anxious avoidant Bowlby",
        "emotional contagion mirror neurons social emotion",
        "resilience adversity post-traumatic growth Tedeschi",
        "growth mindset fixed mindset Carol Dweck",
        "positive psychology PERMA Seligman flourishing",
        "emotional intelligence measurement EQ-i MSCEIT",
        "alexithymia emotional awareness deficit",
        "interpersonal effectiveness DBT skills DEAR MAN",
        "assertiveness communication passive aggressive",
        "conflict de-escalation emotional management",
        "emotional exhaustion burnout compassion fatigue",
        "gratitude practice journaling emotional wellbeing",
        "social-emotional learning SEL school children",
    ]
    n = wiki_api(queries, bucket, "eq_wiki", D, SF, EQ_KW); total += n
    log.info(f"  [EQ-1] Wiki API: {n}")
    n = wiki_stream(EQ_KW, bucket, "eq_stream", D, SF, EQ_KW,
                    max_docs=4000); total += n
    log.info(f"  [EQ-2] Wiki stream: {n}")
    n = se_qa(["psychology", "interpersonal", "workplace",
               "parenting"],
              bucket, "eq_se", D, SF, EQ_KW,
              extra_kw={"emotion", "empathy", "self-awareness",
                        "emotional intelligence", "mindfulness",
                        "resilience", "social skills", "eq",
                        "active listening", "compassion",
                        "communication", "relationship"}); total += n
    log.info(f"  [EQ-3] SE Q&A: {n}")
    n = arxiv_cats(["cs.HC", "q-bio.NC"],
                   bucket, "eq_arxiv", D, SF,
                   extra_kw={"emotion", "empathy", "affective",
                             "social", "wellbeing", "mindfulness"},
                   max_docs=2000); total += n
    log.info(f"  [EQ-4] arXiv: {n}")
    n = pubmed([
        "emotional intelligence training intervention outcomes",
        "empathy development neuroscience mirror neurons",
        "mindfulness emotional regulation clinical outcomes",
        "emotional intelligence workplace leadership performance",
        "self-regulation executive function prefrontal cortex",
        "social-emotional learning children school outcomes",
        "psychological safety team performance organizational",
        "compassion fatigue burnout healthcare emotional",
        "emotional intelligence nursing medicine clinical",
        "attachment theory adult relationships emotion regulation",
        "positive psychology intervention wellbeing flourishing",
        "alexithymia emotional awareness assessment treatment",
    ], bucket, "eq_pubmed", D, SF); total += n
    log.info(f"  [EQ-5] PubMed: {n}")
    n = pmc_fulltext([
        "emotional intelligence training effectiveness review",
        "mindfulness emotion regulation randomized trial",
        "empathy intervention clinical psychology outcome",
        "social emotional learning school mental health",
        "psychological safety team innovation performance",
    ], bucket, "eq_pmc", D, SF); total += n
    log.info(f"  [EQ-6] PMC full-text: {n}")
    n = semantic_scholar([
        "emotional intelligence job performance leadership meta-analysis",
        "mindfulness emotion regulation stress reduction outcome",
        "empathy neural basis fMRI mirror neuron",
        "psychological safety learning team performance",
        "social-emotional learning SEL intervention effectiveness",
        "emotional intelligence measurement construct validity",
        "self-regulation executive function impulse control",
        "positive psychology flourishing wellbeing intervention",
        "compassion meditation loving-kindness brain",
        "nonviolent communication conflict relationship",
        "emotional contagion affect organizational behavior",
        "growth mindset learning outcomes achievement",
    ], bucket, "eq_ss", D, SF, min_citations=3); total += n
    log.info(f"  [EQ-7] Semantic Scholar: {n}")
    n = openalex([
        "emotional intelligence workplace leadership outcomes",
        "empathy compassion psychological wellbeing",
        "mindfulness meditation emotion regulation brain",
        "social emotional learning child development",
    ], bucket, "eq_oa", D, SF, "Psychology"); total += n
    log.info(f"  [EQ-8] OpenAlex: {n}")
    log.info(f"  >>> Emotional Intelligence total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 11. BEHAVIOURAL PSYCHOLOGY
# ══════════════════════════════════════════════════════════════════════════════
def load_behavioural_psychology():
    log.info("=" * 65)
    log.info("BEHAV-1: Behavioural Psychology — Full Spectrum")
    log.info("=" * 65)
    D, SF = "Behavioural Psychology", "Conditioning, Cognition & Applied Practice"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "classical conditioning Pavlov conditioned reflex stimulus",
        "operant conditioning Skinner reinforcement punishment",
        "positive negative reinforcement extinction schedule",
        "variable ratio interval schedule gambling behavior",
        "social learning theory Bandura observation modeling",
        "self-efficacy belief outcome expectation Bandura",
        "cognitive bias heuristic decision making Kahneman",
        "confirmation bias availability anchoring framing",
        "prospect theory loss aversion risk Kahneman Tversky",
        "cognitive dissonance Festinger attitude change",
        "learned helplessness Seligman depression attributional",
        "locus of control internal external Rotter",
        "attribution theory fundamental error dispositional",
        "habit formation cue routine reward loop",
        "behavior modification token economy reward system",
        "cognitive behavioural therapy CBT Beck thought record",
        "dialectical behaviour therapy DBT Linehan skills",
        "exposure therapy systematic desensitization phobia",
        "applied behaviour analysis ABA discrete trial",
        "autism ABA intervention functional behavior analysis",
        "behavioral activation depression scheduling activity",
        "motivational interviewing ambivalence change talk",
        "behavior change transtheoretical model stages",
        "nudge theory libertarian paternalism Thaler Sunstein",
        "placebo nocebo effect expectation behavior health",
        "aggression frustration-aggression Bandura Bobo doll",
        "conformity Milgram Asch Zimbardo obedience authority",
        "group behavior mob psychology deindividuation",
        "behavioral economics irrational decision utility",
        "habit reversal competing response awareness training",
    ]
    n = wiki_api(queries, bucket, "behav_wiki", D, SF, BEHAV_KW); total += n
    log.info(f"  [BEHAV-1] Wiki API: {n}")
    n = wiki_stream(BEHAV_KW, bucket, "behav_stream", D, SF, BEHAV_KW,
                    max_docs=4000); total += n
    log.info(f"  [BEHAV-2] Wiki stream: {n}")
    n = se_qa(["psychology", "cogsci", "philosophy", "parenting"],
              bucket, "behav_se", D, SF, BEHAV_KW,
              extra_kw={"behaviour", "behavior", "conditioning",
                        "reinforcement", "cognitive", "bias", "cbt",
                        "aba", "habit", "motivation", "learning",
                        "reward", "punishment", "therapy"}); total += n
    log.info(f"  [BEHAV-3] SE Q&A: {n}")
    n = arxiv_cats(["q-bio.NC", "cs.HC", "econ.GN"],
                   bucket, "behav_arxiv", D, SF,
                   extra_kw={"behavior", "cognitive", "decision",
                             "learning", "reinforcement", "bias"},
                   max_docs=2000); total += n
    log.info(f"  [BEHAV-4] arXiv: {n}")
    n = pubmed([
        "cognitive behavioural therapy CBT depression anxiety RCT",
        "applied behaviour analysis ABA autism spectrum disorder",
        "operant conditioning behavior modification clinical",
        "habit formation neural basis basal ganglia reward",
        "motivational interviewing substance use outcomes",
        "exposure therapy phobia PTSD randomized trial",
        "behavioral activation depression efficacy review",
        "cognitive bias modification training anxiety depression",
        "dialectical behavior therapy DBT borderline outcomes",
        "nudge behavior change public health intervention",
        "behavioral economics health decision making",
        "social learning observational aggression children",
    ], bucket, "behav_pubmed", D, SF); total += n
    log.info(f"  [BEHAV-5] PubMed: {n}")
    n = pmc_fulltext([
        "cognitive behavioural therapy systematic review meta-analysis",
        "ABA autism intervention effectiveness",
        "habit formation neuroscience striatum reward",
        "motivational interviewing clinical outcomes review",
    ], bucket, "behav_pmc", D, SF); total += n
    log.info(f"  [BEHAV-6] PMC full-text: {n}")
    n = semantic_scholar([
        "cognitive behavioral therapy effectiveness meta-analysis",
        "applied behavior analysis autism intervention evidence",
        "habit loop formation basal ganglia dopamine reward",
        "behavioral economics nudge choice architecture",
        "cognitive bias decision making heuristic error",
        "classical conditioning fear acquisition extinction",
        "operant conditioning reinforcement schedule behavior",
        "social learning theory Bandura aggression modeling",
        "motivational interviewing behavior change effectiveness",
        "learned helplessness depression attributional style",
        "behavior modification token economy institutional",
        "transtheoretical model stages of change readiness",
    ], bucket, "behav_ss", D, SF, min_citations=3); total += n
    log.info(f"  [BEHAV-7] Semantic Scholar: {n}")
    n = openalex([
        "behavioural psychology conditioning learning theory",
        "cognitive bias heuristic decision making",
        "CBT cognitive therapy treatment outcomes",
        "applied behaviour analysis autism intervention",
    ], bucket, "behav_oa", D, SF, "Psychology"); total += n
    log.info(f"  [BEHAV-8] OpenAlex: {n}")
    log.info(f"  >>> Behavioural Psychology total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 12. MOTOR MECHANICS — Internal Combustion Engines & Drivetrain
# ══════════════════════════════════════════════════════════════════════════════
def load_motor_mechanics_ice():
    log.info("=" * 65)
    log.info("MOTOR-1: Internal Combustion Engines & Drivetrain")
    log.info("=" * 65)
    D, SF = "Motor Mechanics", "Internal Combustion Engines & Drivetrain"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "four-stroke engine cycle intake compression power exhaust",
        "two-stroke engine lubrication porting transfer",
        "diesel engine compression ignition glow plug",
        "petrol gasoline engine spark ignition stoichiometric",
        "crankshaft connecting rod big end main bearing",
        "camshaft timing lobe lift duration overlap",
        "timing belt chain tensioner interference engine",
        "fuel injection direct indirect common rail pressure",
        "turbocharger compressor turbine wastegate boost",
        "intercooler charge air cooling density temperature",
        "exhaust manifold header scavenging back pressure",
        "catalytic converter three-way lambda sensor",
        "diesel particulate filter DPF regeneration soot",
        "engine management ECU fuel map ignition timing",
        "variable valve timing VVT VTEC BMW Valvetronic",
        "torque horsepower power band RPM curve",
        "manual gearbox synchromesh gear ratio layshaft",
        "automatic transmission planetary gear torque converter",
        "CVT continuously variable transmission belt pulley",
        "DSG dual clutch transmission wet dry clutch",
        "differential limited slip open locking torque",
        "driveshaft constant velocity joint CV boot",
        "clutch pressure plate friction disc flywheel",
        "disc brake caliper rotor ABS wheel speed sensor",
        "drum brake wheel cylinder handbrake cable",
        "MacPherson strut double wishbone suspension geometry",
        "coilover shock absorber spring rate damping",
        "wheel alignment camber caster toe geometry",
        "steering rack pinion power electric assistance",
        "engine diagnostics OBD-II fault codes scan tool",
    ]
    n = wiki_api(queries, bucket, "motor_ice_wiki", D, SF, ICE_KW); total += n
    log.info(f"  [MOTOR-ICE-1] Wiki API: {n}")
    n = wiki_stream(ICE_KW, bucket, "motor_ice_stream", D, SF, ICE_KW,
                    max_docs=4000); total += n
    log.info(f"  [MOTOR-ICE-2] Wiki stream: {n}")
    n = se_qa(["mechanics", "engineering"],
              bucket, "motor_ice_se", D, SF, ICE_KW,
              extra_kw={"engine", "cylinder", "piston", "crankshaft",
                        "camshaft", "turbo", "fuel injection", "gearbox",
                        "clutch", "brake", "suspension", "differential",
                        "timing belt", "diagnostic", "obd"}); total += n
    log.info(f"  [MOTOR-ICE-3] SE Q&A: {n}")
    n = arxiv_cats(["physics.app-ph", "eess.SY"],
                   bucket, "motor_ice_arxiv", D, SF,
                   extra_kw={"combustion", "engine", "turbo", "emission",
                             "fuel", "thermal", "cylinder"},
                   max_docs=2500); total += n
    log.info(f"  [MOTOR-ICE-4] arXiv: {n}")
    n = semantic_scholar([
        "internal combustion engine efficiency emissions improvement",
        "turbocharger compressor map surge stall performance",
        "diesel common rail injection pressure spray atomization",
        "variable valve timing lift cam phaser performance",
        "engine knock detection ion current combustion",
        "catalytic converter light-off temperature efficiency",
        "DPF diesel particulate filter regeneration strategy",
        "automatic transmission shift control planetary gearset",
        "limited slip differential torque vectoring handling",
        "suspension geometry kinematics camber roll compliance",
        "disc brake thermal fade NVH squeal performance",
        "engine oil viscosity degradation condition monitoring",
    ], bucket, "motor_ice_ss", D, SF); total += n
    log.info(f"  [MOTOR-ICE-5] Semantic Scholar: {n}")
    n = openalex([
        "internal combustion engine combustion emissions efficiency",
        "automotive transmission drivetrain performance",
        "vehicle suspension braking dynamics control",
        "engine diagnostics fault detection OBD",
    ], bucket, "motor_ice_oa", D, SF, "Automotive Engineering"); total += n
    log.info(f"  [MOTOR-ICE-6] OpenAlex: {n}")
    log.info(f"  >>> Motor Mechanics ICE total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 13. MOTOR MECHANICS — Electric Motors, EVs & Automotive Electronics
# ══════════════════════════════════════════════════════════════════════════════
def load_motor_mechanics_ev():
    log.info("=" * 65)
    log.info("MOTOR-2: Electric Motors, EVs & Automotive Electronics")
    log.info("=" * 65)
    D, SF = "Motor Mechanics", "Electric Motors & EV Systems"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "brushless DC motor BLDC commutation hall sensor",
        "permanent magnet synchronous motor PMSM FOC control",
        "induction motor slip rotor stator squirrel cage",
        "switched reluctance motor SRM torque ripple",
        "motor controller FOC field-oriented vector control",
        "IGBT SiC MOSFET GaN power semiconductor inverter",
        "three-phase inverter PWM modulation voltage",
        "battery management system BMS cell balancing",
        "lithium-ion battery chemistry cathode anode electrolyte",
        "NMC LFP solid-state battery energy density",
        "battery thermal management cooling heating TMS",
        "regenerative braking energy recovery efficiency",
        "EV range energy consumption efficiency EPA WLTP",
        "fast charging CCS CHAdeMO 150kW 350kW DC",
        "onboard charger OBC AC level 1 2 type 1 2",
        "vehicle-to-grid V2G bidirectional charging",
        "EV powertrain single-speed gearbox torque",
        "Tesla Model S P100D dual motor all-wheel drive",
        "CAN bus LIN FlexRay automotive communication",
        "AUTOSAR software architecture ECU OEM supplier",
        "ADAS adaptive cruise control lane departure",
        "lidar radar camera sensor fusion autonomous",
        "OTA over-the-air update software-defined vehicle",
        "MOST bus Ethernet automotive 100BASE-T1",
        "automotive cybersecurity ISO 21434 TARA",
        "hybrid electric vehicle HEV PHEV mild parallel",
        "regenerative energy kinetic KERS flywheel",
        "hydrogen fuel cell FCEV Toyota Mirai stack",
        "electric motor efficiency map peak continuous torque",
        "EV charging infrastructure OCPP smart grid",
    ]
    n = wiki_api(queries, bucket, "motor_ev_wiki", D, SF, EV_KW); total += n
    log.info(f"  [MOTOR-EV-1] Wiki API: {n}")
    n = wiki_stream(EV_KW, bucket, "motor_ev_stream", D, SF, EV_KW,
                    max_docs=4000); total += n
    log.info(f"  [MOTOR-EV-2] Wiki stream: {n}")
    n = se_qa(["electronics", "engineering", "mechanics"],
              bucket, "motor_ev_se", D, SF, EV_KW,
              extra_kw={"electric motor", "battery", "bms", "inverter",
                        "ev", "electric vehicle", "charging", "brushless",
                        "pmsm", "foc", "regenerative", "bldc",
                        "igbt", "sic", "can bus", "adas"}); total += n
    log.info(f"  [MOTOR-EV-3] SE Q&A: {n}")
    n = arxiv_cats(["eess.SY", "physics.app-ph", "cs.SY"],
                   bucket, "motor_ev_arxiv", D, SF,
                   extra_kw={"electric vehicle", "battery", "motor",
                             "charging", "inverter", "bms", "adas",
                             "autonomous", "powertrain"},
                   max_docs=4000); total += n
    log.info(f"  [MOTOR-EV-4] arXiv: {n}")
    n = survey_arxiv(["eess.SY", "cs.SY"],
                     bucket, "motor_ev_survey", D, SF, max_docs=1500); total += n
    log.info(f"  [MOTOR-EV-5] arXiv surveys: {n}")
    n = semantic_scholar([
        "PMSM FOC field oriented control efficiency",
        "lithium-ion battery degradation aging SOH estimation",
        "BMS battery management balancing thermal",
        "EV fast charging infrastructure grid impact",
        "SiC MOSFET power loss inverter efficiency",
        "regenerative braking energy recovery control",
        "vehicle-to-grid V2G demand response charging",
        "fuel cell hydrogen FCEV performance durability",
        "automotive CAN bus cybersecurity attack intrusion",
        "ADAS sensor fusion lidar camera radar deep learning",
        "electric motor efficiency map thermal loss model",
        "solid-state battery electrolyte conductivity",
    ], bucket, "motor_ev_ss", D, SF); total += n
    log.info(f"  [MOTOR-EV-6] Semantic Scholar: {n}")
    n = openalex([
        "electric vehicle battery motor powertrain",
        "EV charging infrastructure grid integration",
        "automotive electronics control system",
        "ADAS autonomous vehicle sensor fusion",
    ], bucket, "motor_ev_oa", D, SF, "EV Technology"); total += n
    log.info(f"  [MOTOR-EV-7] OpenAlex: {n}")
    log.info(f"  >>> Motor Mechanics EV total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 14. OPERATING MACHINERY — Heavy Equipment & Construction
# ══════════════════════════════════════════════════════════════════════════════
def load_machinery_heavy():
    log.info("=" * 65)
    log.info("MACH-1: Heavy Machinery — Hydraulics, Construction & Agriculture")
    log.info("=" * 65)
    D, SF = "Operating Machinery", "Heavy Equipment & Construction"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "hydraulic system pump motor actuator pressure relief",
        "hydraulic cylinder rod piston seal stroke force",
        "hydraulic fluid viscosity contamination filtration",
        "pneumatic system compressor valve actuator FRL",
        "excavator crawler tracked operation digging",
        "hydraulic excavator arm boom bucket crowd",
        "bulldozer blade ripper dozing track undercarriage",
        "tower crane slewing luffing erection capacity",
        "mobile crane outrigger load chart radius lift",
        "crawler crane lifting load stability derating",
        "telehandler telescoping boom forklift attachment",
        "rough terrain forklift mast tilt side shift",
        "skid steer loader hydraulic drive attachment",
        "wheel loader bucket capacity Z-bar linkage",
        "agricultural tractor PTO three-point hitch",
        "combine harvester threshing separation cleaning",
        "ploughing tillage soil preparation cultivation",
        "irrigation system drip sprinkler pivot scheduling",
        "PLC programmable logic controller ladder diagram",
        "SCADA industrial control system HMI automation",
        "hydraulic press forming stamping bending force",
        "concrete pump boom pump line pump output",
        "piling rig driven bored CFA auger installation",
        "OSHA safety standards lockout tagout LOTO",
        "crane lift plan rigging sling angle load",
        "machinery guarding safety interlock proximity",
        "vibration analysis condition monitoring bearing",
        "hydraulic troubleshooting fault cavitation noise",
        "GPS machine control grading excavation accuracy",
        "telematics fleet management maintenance OEM",
    ]
    n = wiki_api(queries, bucket, "mach_heavy_wiki", D, SF, MACHINERY_KW); total += n
    log.info(f"  [MACH-HVY-1] Wiki API: {n}")
    n = wiki_stream(MACHINERY_KW, bucket, "mach_heavy_stream", D, SF,
                    MACHINERY_KW, max_docs=4000); total += n
    log.info(f"  [MACH-HVY-2] Wiki stream: {n}")
    n = se_qa(["engineering", "mechanics", "electronics", "diy"],
              bucket, "mach_heavy_se", D, SF, MACHINERY_KW,
              extra_kw={"hydraulic", "pneumatic", "crane", "excavator",
                        "forklift", "loader", "tractor", "plc",
                        "actuator", "pump", "valve", "cylinder",
                        "safety", "rigging", "automation"}); total += n
    log.info(f"  [MACH-HVY-3] SE Q&A: {n}")
    n = arxiv_cats(["eess.SY", "cs.RO", "physics.app-ph"],
                   bucket, "mach_heavy_arxiv", D, SF,
                   extra_kw={"hydraulic", "construction", "robot",
                             "automation", "control", "actuator",
                             "agricultural", "machine"},
                   max_docs=3000); total += n
    log.info(f"  [MACH-HVY-4] arXiv: {n}")
    n = semantic_scholar([
        "hydraulic excavator control energy efficiency optimization",
        "construction machinery automation GPS grade control",
        "agricultural tractor precision farming GPS GNSS",
        "crane stability load monitoring safety system",
        "PLC SCADA industrial automation Industry 4.0",
        "hydraulic system fault diagnosis condition monitoring",
        "combine harvester crop loss optimization adjustment",
        "forklift safety stability dynamic load lateral",
        "construction site safety accident prevention OSHA",
        "machinery vibration bearing fault detection FFT",
        "hydraulic fluid contamination filtration cleanliness",
        "telerobotic excavator remote operator haptic",
    ], bucket, "mach_heavy_ss", D, SF); total += n
    log.info(f"  [MACH-HVY-5] Semantic Scholar: {n}")
    n = openalex([
        "hydraulic machinery control efficiency performance",
        "construction equipment automation GPS control",
        "agricultural machinery precision farming",
        "crane lifting safety load monitoring",
    ], bucket, "mach_heavy_oa", D, SF, "Heavy Machinery"); total += n
    log.info(f"  [MACH-HVY-6] OpenAlex: {n}")
    log.info(f"  >>> Heavy Machinery total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 15. OPERATING MACHINERY — Commercial Vehicles & Road Transport
# ══════════════════════════════════════════════════════════════════════════════
def load_machinery_vehicles():
    log.info("=" * 65)
    log.info("MACH-2: Commercial Vehicles, HGV & Road Transport")
    log.info("=" * 65)
    D, SF = "Operating Machinery", "Commercial Vehicles & Road Transport"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "HGV heavy goods vehicle Class 1 2 articulated",
        "CDL commercial driver license endorsement test",
        "tachograph digital driver hours rest EU rules",
        "air brake system dual circuit anti-lock ABS",
        "fifth wheel coupling king pin landing legs",
        "trailer types curtainsider flatbed refrigerated",
        "tanker vehicle chemical hazmat ADR dangerous",
        "vehicle inspection daily walk-around checklist",
        "truck wheel alignment steering tracking geometry",
        "def diesel exhaust fluid SCR AdBlue NOx",
        "fleet management telematics GPS tracking fuel",
        "driver hours regulations EU 561/2006 WTD",
        "hazardous materials transport placard label class",
        "load securing lashing point stacking dunnage",
        "road freight logistics delivery routing optimisation",
        "bus coach operation route schedule passenger",
        "minibus driver CPC certificate periodic training",
        "driver CPC certificate periodic training EU",
        "road traffic law highway code stopping distances",
        "defensive driving technique hazard perception",
        "skid control recovery oversteer understeer",
        "vehicle dynamics weight transfer stability",
        "articulated vehicle reversing manoeuvring technique",
        "night driving fatigue management rest breaks",
        "winter driving tyre chains snow mud terrain",
        "vehicle weight limits axle GVW payload rating",
        "roadside check enforcement DVSA ANPR weighbridge",
        "fleet maintenance preventive schedule service",
        "truck cab comfort ergonomics sleep berth",
        "last-mile delivery urban consolidation EV van",
    ]
    n = wiki_api(queries, bucket, "mach_veh_wiki", D, SF, VEHICLE_KW); total += n
    log.info(f"  [MACH-VEH-1] Wiki API: {n}")
    n = wiki_stream(VEHICLE_KW, bucket, "mach_veh_stream", D, SF, VEHICLE_KW,
                    max_docs=3000); total += n
    log.info(f"  [MACH-VEH-2] Wiki stream: {n}")
    n = se_qa(["mechanics", "driving", "travel"],
              bucket, "mach_veh_se", D, SF, VEHICLE_KW,
              extra_kw={"truck", "hgv", "lorry", "trailer", "cdl",
                        "tachograph", "air brake", "commercial",
                        "driving", "hazmat", "fleet", "transport"}); total += n
    log.info(f"  [MACH-VEH-3] SE Q&A: {n}")
    n = semantic_scholar([
        "heavy vehicle stability rollover prevention control",
        "truck driver fatigue monitoring detection system",
        "fleet management telematics fuel consumption routing",
        "commercial vehicle air brake performance ABS EBS",
        "logistics last-mile delivery urban route optimization",
        "truck platooning cooperative adaptive cruise control",
        "freight transport emissions reduction electric truck",
        "hazardous materials transport safety regulation",
        "tachograph driver behavior monitoring compliance",
        "commercial vehicle inspection defect detection",
        "vehicle dynamics heavy truck handling stability",
        "autonomous trucking highway platooning freight",
    ], bucket, "mach_veh_ss", D, SF); total += n
    log.info(f"  [MACH-VEH-4] Semantic Scholar: {n}")
    n = openalex([
        "commercial vehicle transport logistics safety",
        "truck driver fatigue road safety",
        "freight transport emission optimization",
        "vehicle dynamics stability heavy truck",
    ], bucket, "mach_veh_oa", D, SF, "Transport Engineering"); total += n
    log.info(f"  [MACH-VEH-5] OpenAlex: {n}")
    log.info(f"  >>> Commercial Vehicles total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 16. AVIATION — Fixed-Wing Aircraft
# ══════════════════════════════════════════════════════════════════════════════
def load_aviation_fixed_wing():
    log.info("=" * 65)
    log.info("AVIA-1: Fixed-Wing Aircraft — Aerodynamics, Avionics & Systems")
    log.info("=" * 65)
    D, SF = "Aviation", "Fixed-Wing Aircraft"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "aerodynamics lift generation Bernoulli Newtonian",
        "airfoil NACA profile chord camber thickness",
        "angle of attack stall critical CLmax",
        "drag types parasite induced profile form",
        "wing sweep taper aspect ratio efficiency",
        "aircraft control surfaces aileron elevator rudder",
        "trim tab elevator authority balance",
        "flap slat leading edge trailing edge lift",
        "high-lift device fowler slot double slotted",
        "aircraft stability longitudinal lateral directional",
        "weight and balance CG envelope moment arm",
        "aircraft performance takeoff climb cruise range",
        "airspeed IAS TAS EAS CAS Mach number",
        "altimeter barometric pressure altitude encoding",
        "attitude indicator artificial horizon gyroscope",
        "VOR navigation radial bearing station passage",
        "ILS instrument landing glideslope localizer",
        "GPS GNSS aviation WAAS SBAS LPV approach",
        "instrument flight rules IFR clearance procedure",
        "visual flight rules VFR weather minimum visibility",
        "air traffic control ATC clearance squawk transponder",
        "turbojet thermodynamic Brayton cycle intake compressor",
        "turbofan bypass ratio fan core thrust",
        "turboprop shaft power propeller gearbox",
        "piston aircraft Lycoming Continental magneto",
        "aircraft pressurization cabin altitude differential",
        "aircraft fuel system tank sump drain crossfeed",
        "glass cockpit EFIS MFD PFD avionics suite",
        "TCAS traffic collision avoidance resolution advisory",
        "aircraft maintenance airworthiness log Part 145",
    ]
    n = wiki_api(queries, bucket, "avia_fw_wiki", D, SF, AVIATION_KW); total += n
    log.info(f"  [AVIA-FW-1] Wiki API: {n}")
    n = wiki_stream(AVIATION_KW, bucket, "avia_fw_stream", D, SF, AVIATION_KW,
                    max_docs=4000); total += n
    log.info(f"  [AVIA-FW-2] Wiki stream: {n}")
    n = se_qa(["aviation", "engineering", "physics"],
              bucket, "avia_fw_se", D, SF, AVIATION_KW,
              extra_kw={"aircraft", "airplane", "aerodynamics", "pilot",
                        "flight", "airfoil", "avionics", "ils",
                        "vor", "ifr", "atc", "turbine", "piston",
                        "stall", "lift", "thrust", "drag"}); total += n
    log.info(f"  [AVIA-FW-3] SE Q&A: {n}")
    n = arxiv_cats(["physics.flu-dyn", "cs.RO", "eess.SY"],
                   bucket, "avia_fw_arxiv", D, SF,
                   extra_kw={"aircraft", "aerodynamics", "wing",
                             "flight", "control", "avionics",
                             "turbulence", "airfoil"},
                   max_docs=3000); total += n
    log.info(f"  [AVIA-FW-4] arXiv: {n}")
    n = survey_arxiv(["physics.flu-dyn", "eess.SY"],
                     bucket, "avia_fw_survey", D, SF, max_docs=1000); total += n
    log.info(f"  [AVIA-FW-5] arXiv surveys: {n}")
    n = semantic_scholar([
        "laminar turbulent flow transition airfoil wing drag",
        "aircraft flutter aeroelastic instability composite wing",
        "fly-by-wire control law envelope protection flight",
        "glass cockpit EFIS situational awareness pilot",
        "aircraft icing anti-ice de-ice system certification",
        "turbofan engine noise reduction chevron nozzle",
        "aircraft fuel efficiency aerodynamic optimization",
        "UAV unmanned aircraft system flight control autopilot",
        "aircraft accident investigation human factors CRM",
        "avionics architecture federated integrated modular",
        "runway excursion approach stabilized landing safety",
        "winglet blended split scimitar induced drag reduction",
    ], bucket, "avia_fw_ss", D, SF); total += n
    log.info(f"  [AVIA-FW-6] Semantic Scholar: {n}")
    n = openalex([
        "aircraft aerodynamics wing design performance",
        "aviation safety human factors accident",
        "avionics flight management navigation",
        "turbofan engine performance efficiency emissions",
    ], bucket, "avia_fw_oa", D, SF, "Aerospace Engineering"); total += n
    log.info(f"  [AVIA-FW-7] OpenAlex: {n}")
    log.info(f"  >>> Aviation Fixed-Wing total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 17. AVIATION — Rotorcraft & Helicopters
# ══════════════════════════════════════════════════════════════════════════════
def load_aviation_rotorcraft():
    log.info("=" * 65)
    log.info("AVIA-2: Helicopters & Rotorcraft — Theory, Systems & Operations")
    log.info("=" * 65)
    D, SF = "Aviation", "Rotorcraft & Helicopters"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "helicopter rotor theory induced velocity actuator disk",
        "main rotor blade lift generation rotation flapping",
        "collective pitch control vertical climb descent",
        "cyclic pitch control longitudinal lateral tilt",
        "tail rotor torque reaction yaw pedal anti-torque",
        "NOTAR no tail rotor fenestron ducted fan",
        "autorotation entry flare power-off landing",
        "retreating blade stall high-speed translational",
        "vortex ring state power settling avoiding recovery",
        "ground resonance mechanical instability landing gear",
        "dynamic rollover slope landing lateral force",
        "translational lift ETL effective transitional",
        "helicopter hover ground effect OGE IGE",
        "swashplate pitch links control rods mechanics",
        "rotor head fully articulated semi-rigid teetering",
        "blade lead lag drag hinge flapping angle",
        "turboshaft engine power turbine gas generator",
        "helicopter transmission gearbox planetary bevel",
        "freewheeling unit sprag clutch one-way drive",
        "rotor wash downwash slope landing technique",
        "tandem rotor chinook CH-47 synchronized blades",
        "coaxial rotor Kamov Ka-52 intermeshing Kaman",
        "tiltrotor V-22 Osprey conversion mode transition",
        "Robinson R22 R44 light training helicopter",
        "Bell 206 JetRanger Longranger commercial utility",
        "helicopter emergency autorotation height-velocity",
        "EMS helicopter sling load cargo external load",
        "night vision NVG goggle helicopter operations",
        "helicopter performance density altitude power",
        "instrument helicopter IFR operation IMC",
    ]
    n = wiki_api(queries, bucket, "avia_rot_wiki", D, SF, ROTORCRAFT_KW); total += n
    log.info(f"  [AVIA-ROT-1] Wiki API: {n}")
    n = wiki_stream(ROTORCRAFT_KW, bucket, "avia_rot_stream", D, SF,
                    ROTORCRAFT_KW, max_docs=3000); total += n
    log.info(f"  [AVIA-ROT-2] Wiki stream: {n}")
    n = se_qa(["aviation", "engineering", "physics"],
              bucket, "avia_rot_se", D, SF, ROTORCRAFT_KW,
              extra_kw={"helicopter", "rotor", "collective", "cyclic",
                        "autorotation", "hover", "tail rotor",
                        "vortex ring", "retreating blade", "swashplate",
                        "turboshaft", "blade", "rotorcraft"}); total += n
    log.info(f"  [AVIA-ROT-3] SE Q&A: {n}")
    n = arxiv_cats(["physics.flu-dyn", "cs.RO", "eess.SY"],
                   bucket, "avia_rot_arxiv", D, SF,
                   extra_kw={"rotor", "helicopter", "blade", "vtol",
                             "hover", "aerodynamics", "uav"},
                   max_docs=2500); total += n
    log.info(f"  [AVIA-ROT-4] arXiv: {n}")
    n = semantic_scholar([
        "helicopter rotor aerodynamics CFD blade vortex",
        "autorotation energy management glide descent",
        "vortex ring state power settling helicopter recovery",
        "tiltrotor V-22 conversion transition performance",
        "helicopter vibration active noise control IBC",
        "rotor blade composite material design fatigue",
        "helicopter turboshaft engine performance altitude",
        "coaxial rotor aerodynamic interaction performance",
        "helicopter flight mechanics trim simulation",
        "maritime helicopter winching rescue operations",
        "helicopter pilot training simulator fidelity",
        "UAV multirotor quadcopter control stability",
    ], bucket, "avia_rot_ss", D, SF); total += n
    log.info(f"  [AVIA-ROT-5] Semantic Scholar: {n}")
    n = openalex([
        "helicopter rotor aerodynamics performance",
        "rotorcraft dynamics control stability",
        "helicopter safety accident human factors",
        "VTOL aircraft hover transition control",
    ], bucket, "avia_rot_oa", D, SF, "Rotorcraft Engineering"); total += n
    log.info(f"  [AVIA-ROT-6] OpenAlex: {n}")
    log.info(f"  >>> Aviation Rotorcraft total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 18. MARINE — Boats, Yachts & Seamanship
# ══════════════════════════════════════════════════════════════════════════════
def load_marine():
    log.info("=" * 65)
    log.info("MARINE-1: Boats, Yachts, Seamanship & Marine Engineering")
    log.info("=" * 65)
    D, SF = "Marine", "Boats, Yachts & Seamanship"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "yacht design hull form displacement planing semi-planing",
        "monohull catamaran trimaran multihull stability",
        "keel types fin bulb full swing lifting",
        "hull buoyancy Archimedes displacement weight",
        "freeboard draft waterplane area stability",
        "metacentric height GM righting lever GZ curve",
        "rigging standing running masthead fractional",
        "sail types mainsail genoa jib spinnaker gennaker",
        "sailing points of sail close-hauled beam reach",
        "tacking gybing sail trim trim tab telltale",
        "apparent wind true wind VMG velocity made good",
        "celestial navigation sextant sight reduction",
        "chart navigation pilotage GPS waypoint bearing",
        "tidal calculation secondary port range springs",
        "COLREGs rules of the road give-way stand-on",
        "VHF radio DSC GMDSS distress Mayday Pan-Pan",
        "marine diesel engine maintenance injection pump",
        "outboard motor two-stroke four-stroke impeller",
        "inboard saildrive shaft seal anodes",
        "antifouling hull coating osmosis blister repair",
        "anchoring technique scope holding power dragging",
        "mooring lines spring breast stern bow",
        "marina berth alongside approach wind tide",
        "passage planning weather routing pilot book",
        "heavy weather sailing storm jib trysail heaving",
        "man overboard procedure lifebuoy recovery",
        "EPIRB liferaft flare distress signal equipment",
        "container ship tanker bulker commercial vessel",
        "RYA Day Skipper Yachtmaster certificate",
        "marine surveyor condition survey insurance",
    ]
    n = wiki_api(queries, bucket, "marine_wiki", D, SF, MARINE_KW); total += n
    log.info(f"  [MARINE-1] Wiki API: {n}")
    n = wiki_stream(MARINE_KW, bucket, "marine_stream", D, SF, MARINE_KW,
                    max_docs=4000); total += n
    log.info(f"  [MARINE-2] Wiki stream: {n}")
    n = se_qa(["boating", "outdoors", "travel", "engineering"],
              bucket, "marine_se", D, SF, MARINE_KW,
              extra_kw={"yacht", "boat", "sailing", "sail", "keel",
                        "hull", "diesel", "anchor", "mooring",
                        "colregs", "vhf", "navigation", "tide",
                        "marine", "outboard", "antifouling"}); total += n
    log.info(f"  [MARINE-3] SE Q&A: {n}")
    n = arxiv_cats(["physics.flu-dyn", "eess.SY", "cs.RO"],
                   bucket, "marine_arxiv", D, SF,
                   extra_kw={"ship", "marine", "hull", "vessel",
                             "wave", "sailing", "ocean", "maritime"},
                   max_docs=2500); total += n
    log.info(f"  [MARINE-4] arXiv: {n}")
    n = semantic_scholar([
        "yacht hull design resistance CFD wave making",
        "sailing yacht performance polar VMG upwind",
        "ship stability righting moment metacentric height",
        "marine diesel engine emissions NOx IMO MARPOL",
        "autonomous surface vessel navigation obstacle",
        "weather routing ship optimization fuel consumption",
        "maritime COLREGs collision avoidance algorithm",
        "composite boat building carbon fibre epoxy infusion",
        "anti-fouling coating biocide drag performance",
        "offshore sailing safety equipment heavy weather",
        "tidal energy stream turbine marine renewable",
        "marine corrosion cathodic protection zinc anode",
    ], bucket, "marine_ss", D, SF); total += n
    log.info(f"  [MARINE-5] Semantic Scholar: {n}")
    n = openalex([
        "yacht sailing hydrodynamics performance design",
        "marine vessel navigation safety seamanship",
        "ship resistance propulsion efficiency",
        "maritime safety COLREGs collision avoidance",
    ], bucket, "marine_oa", D, SF, "Marine Engineering"); total += n
    log.info(f"  [MARINE-6] OpenAlex: {n}")
    n = gutenberg_fetch([
        (6701,  "Sailing Alone Around the World — Joshua Slocum"),
        (4352,  "The Seaman's Friend — R.H. Dana"),
        (30637, "A Voyage Round the World — Francis Drake"),
    ], bucket, "marine_gutenberg", D, SF); total += n
    log.info(f"  [MARINE-7] Gutenberg: {n}")
    log.info(f"  >>> Marine total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 19. HYDROMECHANICS
# ══════════════════════════════════════════════════════════════════════════════
def load_hydromechanics():
    log.info("=" * 65)
    log.info("HYDRO-1: Hydromechanics — Fluid Mechanics & Hydraulics")
    log.info("=" * 65)
    D, SF = "Hydromechanics", "Fluid Mechanics & Applied Hydraulics"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "fluid mechanics statics dynamics kinematics",
        "Bernoulli equation streamline energy pressure velocity",
        "Navier-Stokes equations viscous incompressible",
        "Reynolds number laminar turbulent transition",
        "boundary layer Blasius Prandtl viscous thickness",
        "drag coefficient form friction pressure body",
        "lift generation Kutta-Joukowski circulation",
        "turbulent flow Kolmogorov energy cascade scale",
        "pipe flow Hagen-Poiseuille Darcy-Weisbach friction",
        "Moody chart friction factor relative roughness",
        "pump centrifugal axial mixed flow impeller",
        "pump curve system curve operating point",
        "cavitation NPSH bubble collapse erosion",
        "hydraulic turbine Pelton Francis Kaplan runner",
        "hydraulic jump subcritical supercritical Froude",
        "open channel flow Manning Chezy wetted perimeter",
        "weir orifice flow measurement discharge coefficient",
        "continuity equation mass conservation flow rate",
        "momentum equation control volume force thrust",
        "surface tension capillary action contact angle",
        "wave mechanics gravity wave celerity period",
        "shallow water wave deep water dispersion",
        "tsunami generation propagation run-up",
        "ship hull drag resistance wave-making Froude",
        "computational fluid dynamics CFD finite volume",
        "RANS k-epsilon turbulence model closure",
        "large eddy simulation LES direct numerical DNS",
        "hydraulic accumulator energy storage nitrogen",
        "Pascal law pressure transmission hydrostatic",
        "venturi meter flow measurement pressure drop",
    ]
    n = wiki_api(queries, bucket, "hydro_wiki", D, SF, HYDRO_KW); total += n
    log.info(f"  [HYDRO-1] Wiki API: {n}")
    n = wiki_stream(HYDRO_KW, bucket, "hydro_stream", D, SF, HYDRO_KW,
                    max_docs=4000); total += n
    log.info(f"  [HYDRO-2] Wiki stream: {n}")
    n = se_qa(["engineering", "physics", "mechanics", "scicomp"],
              bucket, "hydro_se", D, SF, HYDRO_KW,
              extra_kw={"fluid", "hydraulic", "bernoulli", "reynolds",
                        "viscosity", "turbulence", "laminar", "pump",
                        "pipe flow", "navier", "cavitation", "cfd",
                        "wave", "pressure", "velocity", "flow"}); total += n
    log.info(f"  [HYDRO-3] SE Q&A: {n}")
    n = arxiv_cats(["physics.flu-dyn", "physics.class-ph", "cond-mat.soft"],
                   bucket, "hydro_arxiv", D, SF,
                   extra_kw={"fluid", "flow", "turbulence", "wave",
                             "hydraulic", "viscous", "drag", "lift",
                             "cavitation", "pipe"},
                   max_docs=5000); total += n
    log.info(f"  [HYDRO-4] arXiv: {n}")
    n = survey_arxiv(["physics.flu-dyn"],
                     bucket, "hydro_survey", D, SF, max_docs=1500); total += n
    log.info(f"  [HYDRO-5] arXiv surveys: {n}")
    n = semantic_scholar([
        "turbulent pipe flow Reynolds stress DNS simulation",
        "boundary layer transition laminar turbulent control",
        "centrifugal pump performance cavitation NPSH",
        "computational fluid dynamics OpenFOAM ANSYS Fluent",
        "wave breaking shoaling nearshore coastal",
        "ship resistance hull form optimization drag",
        "hydraulic fracturing fracking fluid mechanics",
        "microfluidics lab on chip droplet flow",
        "LES large eddy simulation turbulence resolved",
        "drag reduction polymer additive turbulent pipe",
        "tidal wave energy converter hydrokinetic turbine",
        "sloshing liquid tank seismic dynamic response",
    ], bucket, "hydro_ss", D, SF); total += n
    log.info(f"  [HYDRO-6] Semantic Scholar: {n}")
    n = openalex([
        "fluid mechanics turbulence flow simulation",
        "hydraulic pump turbine performance cavitation",
        "wave hydrodynamics coastal ocean",
        "computational fluid dynamics CFD validation",
    ], bucket, "hydro_oa", D, SF, "Fluid Engineering"); total += n
    log.info(f"  [HYDRO-7] OpenAlex: {n}")
    log.info(f"  >>> Hydromechanics total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 20. CONFLICT RESOLUTION & NEGOTIATION
# ══════════════════════════════════════════════════════════════════════════════
def load_conflict_negotiation():
    log.info("=" * 65)
    log.info("CONF-1: Conflict Resolution & Negotiation")
    log.info("=" * 65)
    D, SF = "Conflict Resolution", "Negotiation Theory & Applied Practice"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "negotiation theory principled interest-based Fisher Ury",
        "BATNA best alternative negotiated agreement leverage",
        "ZOPA zone possible agreement overlap range",
        "distributive negotiation positional zero-sum",
        "integrative negotiation win-win mutual gain",
        "mediation process stages shuttle Caucus",
        "arbitration binding non-binding procedure",
        "facilitation group process neutral third party",
        "conflict styles Thomas-Kilmann competing collaborating",
        "de-escalation technique verbal non-verbal calm",
        "game theory Nash equilibrium cooperative payoff",
        "prisoner dilemma iterated cooperation defection",
        "hostage negotiation crisis FBI behavioral",
        "crisis negotiation active listening rapport building",
        "international diplomacy treaty multilateral bilateral",
        "UN mediation peacebuilding ceasefire agreement",
        "restorative justice circle harm repair offender",
        "peer mediation school bullying conflict resolution",
        "workplace conflict HR grievance procedure",
        "collective bargaining union management wage",
        "negotiation anchoring first offer concession pattern",
        "cross-cultural negotiation Hofstede face saving",
        "salary negotiation BATNA market research offer",
        "divorce mediation family separation child custody",
        "commercial dispute ADR alternative resolution",
        "active listening paraphrase reflect summarize",
        "reframing interest position technique",
        "DARVO manipulation accountability deflection",
        "manipulation tactics recognition response",
        "peace psychology reconciliation post-conflict",
    ]
    n = wiki_api(queries, bucket, "conf_wiki", D, SF, CONFLICT_KW); total += n
    log.info(f"  [CONF-1] Wiki API: {n}")
    n = wiki_stream(CONFLICT_KW, bucket, "conf_stream", D, SF, CONFLICT_KW,
                    max_docs=4000); total += n
    log.info(f"  [CONF-2] Wiki stream: {n}")
    n = se_qa(["interpersonal", "workplace", "law",
               "politics", "philosophy"],
              bucket, "conf_se", D, SF, CONFLICT_KW,
              extra_kw={"negotiation", "conflict", "mediation",
                        "batna", "dispute", "resolution",
                        "arbitration", "compromise", "de-escalation",
                        "win-win", "hostage", "workplace"}); total += n
    log.info(f"  [CONF-3] SE Q&A: {n}")
    n = arxiv_cats(["econ.GN", "cs.GT"],
                   bucket, "conf_arxiv", D, SF,
                   extra_kw={"negotiation", "conflict", "game theory",
                             "cooperation", "mediation", "bargaining"},
                   max_docs=2500); total += n
    log.info(f"  [CONF-4] arXiv: {n}")
    n = pubmed([
        "conflict resolution intervention effectiveness workplace",
        "mediation dispute resolution mental health outcome",
        "negotiation training skill development study",
        "restorative justice juvenile recidivism outcome",
        "hostage crisis negotiation psychology behavioral",
        "family mediation divorce child outcome wellbeing",
        "peer mediation school conflict bullying reduction",
        "cross-cultural negotiation communication outcome",
    ], bucket, "conf_pubmed", D, SF); total += n
    log.info(f"  [CONF-5] PubMed: {n}")
    n = semantic_scholar([
        "negotiation BATNA reservation value anchoring",
        "game theory Nash equilibrium cooperative strategy",
        "mediation effectiveness dispute resolution outcome",
        "cross-cultural negotiation face concern individualism",
        "crisis negotiation hostage behavioral change",
        "restorative justice harm repair reoffending",
        "workplace conflict management organizational behavior",
        "integrative negotiation interest-based joint gain",
        "international diplomacy mediation peace agreement",
        "negotiation training skill development transfer",
        "conflict escalation de-escalation emotion regulation",
        "salary negotiation gender gap strategy outcome",
    ], bucket, "conf_ss", D, SF, min_citations=3); total += n
    log.info(f"  [CONF-6] Semantic Scholar: {n}")
    n = openalex([
        "negotiation strategy outcome bargaining",
        "conflict resolution mediation effectiveness",
        "game theory cooperation bargaining Nash",
        "workplace dispute resolution HR management",
    ], bucket, "conf_oa", D, SF, "Conflict Studies"); total += n
    log.info(f"  [CONF-7] OpenAlex: {n}")
    log.info(f"  >>> Conflict Resolution & Negotiation total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 21. MANNERS, POLITENESS & SOCIAL ETIQUETTE
# ══════════════════════════════════════════════════════════════════════════════
def load_etiquette_manners():
    log.info("=" * 65)
    log.info("ETIQ-1: Manners, Politeness & Social Etiquette")
    log.info("=" * 65)
    D, SF = "Social Etiquette", "Manners, Politeness & Cultural Protocol"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "etiquette history development social rules civility",
        "table manners formal dining fork knife placement",
        "Western table setting formal place setting crystal",
        "French dining etiquette cheese wine bread",
        "Japanese etiquette bow omotenashi chopsticks",
        "British etiquette afternoon tea queuing custom",
        "American etiquette tipping handshake greeting",
        "Middle Eastern etiquette hospitality right hand",
        "Chinese etiquette banquet toast face mianzi",
        "Indian etiquette namaste right hand hospitality",
        "business etiquette card exchange meeting protocol",
        "email etiquette subject line reply formality",
        "phone etiquette voicemail hold introduction",
        "meeting etiquette punctuality agenda minutes",
        "dress code black tie white tie business formal",
        "wedding etiquette guest host gift registry",
        "funeral etiquette condolence dress behavior",
        "gift giving etiquette wrapping thank-you note",
        "forms of address title honorific Dr Mr Ms",
        "social media netiquette online behavior civility",
        "dinner party hosting invitation seating",
        "introductions formal informal hierarchy order",
        "personal space proxemics cultural variation",
        "eye contact cultural meaning direct indirect",
        "handshake greeting hug bow kiss cheek",
        "politeness theory face positive negative Brown",
        "linguistic politeness indirect request hedge",
        "social norms informal rules violation sanction",
        "chivalry modern manners gender neutral courtesy",
        "restaurant etiquette ordering tipping napkin",
    ]
    n = wiki_api(queries, bucket, "etiq_wiki", D, SF, ETIQUETTE_KW); total += n
    log.info(f"  [ETIQ-1] Wiki API: {n}")
    n = wiki_stream(ETIQUETTE_KW, bucket, "etiq_stream", D, SF, ETIQUETTE_KW,
                    max_docs=3000); total += n
    log.info(f"  [ETIQ-2] Wiki stream: {n}")
    n = se_qa(["interpersonal", "travel", "expatriates",
               "workplace", "parenting"],
              bucket, "etiq_se", D, SF, ETIQUETTE_KW,
              extra_kw={"etiquette", "manners", "polite", "courtesy",
                        "formal", "rude", "dress code", "table",
                        "greeting", "culture", "respect",
                        "protocol", "business etiquette"}); total += n
    log.info(f"  [ETIQ-3] SE Q&A: {n}")
    n = pubmed([
        "politeness theory face linguistic Brown Levinson",
        "social norms etiquette behavior cultural variation",
        "civility incivility workplace behavior outcome",
        "prosocial behavior courtesy helping social norm",
        "cultural etiquette cross-cultural communication",
    ], bucket, "etiq_pubmed", D, SF); total += n
    log.info(f"  [ETIQ-4] PubMed: {n}")
    n = semantic_scholar([
        "politeness theory face threatening act linguistic",
        "social norms enforcement informal sanction",
        "cross-cultural communication etiquette misunderstanding",
        "netiquette online civility social media behavior",
        "workplace civility incivility organizational outcome",
        "dining etiquette cultural variation behavior study",
        "gift exchange reciprocity social norm culture",
        "dress code social signal identity perception",
        "proxemics personal space cross-cultural Hall",
        "eye contact gaze culture communication signal",
        "greeting ritual handshake hug bow cultural",
        "chivalry courtesy gender contemporary norms",
    ], bucket, "etiq_ss", D, SF, min_citations=2); total += n
    log.info(f"  [ETIQ-5] Semantic Scholar: {n}")
    n = openalex([
        "social etiquette politeness norms cultural",
        "manners courtesy social behavior civility",
        "cross-cultural communication etiquette protocol",
        "workplace behavior professionalism norms",
    ], bucket, "etiq_oa", D, SF, "Social Sciences"); total += n
    log.info(f"  [ETIQ-6] OpenAlex: {n}")
    n = gutenberg_fetch([
        (17921, "Etiquette — Emily Post"),
        (14977, "The Book of Good Manners — W.C. Green"),
        (13547, "Manners and Rules of Good Society — Frances Countess of Airlie"),
    ], bucket, "etiq_gutenberg", D, SF); total += n
    log.info(f"  [ETIQ-7] Gutenberg: {n}")
    log.info(f"  >>> Manners & Etiquette total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 22. ARTIFICIAL INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
def load_artificial_intelligence():
    log.info("=" * 65)
    log.info("AI-1: Artificial Intelligence — Full Spectrum")
    log.info("=" * 65)
    D, SF = "Artificial Intelligence", "ML, DL, NLP, CV & AI Systems"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "machine learning supervised unsupervised reinforcement",
        "deep learning neural network layer activation",
        "gradient descent backpropagation chain rule",
        "convolutional neural network CNN image recognition",
        "recurrent neural network LSTM GRU sequence",
        "transformer architecture self-attention multi-head",
        "BERT bidirectional encoder representations language",
        "GPT generative pre-trained transformer autoregressive",
        "large language model LLM prompt engineering",
        "fine-tuning transfer learning domain adaptation",
        "reinforcement learning policy reward Q-learning",
        "deep Q-network DQN experience replay target",
        "policy gradient REINFORCE actor critic PPO",
        "AlphaGo AlphaZero MCTS self-play chess Go",
        "generative adversarial network GAN discriminator",
        "variational autoencoder VAE latent space generation",
        "diffusion model DDPM score matching image synthesis",
        "stable diffusion latent diffusion CLIP conditioning",
        "object detection YOLO SSD Faster RCNN anchor",
        "semantic segmentation FCN DeepLab SegFormer",
        "natural language processing tokenization embedding",
        "word2vec GloVe FastText semantic embedding",
        "BERT fine-tuning classification NER QA",
        "GPT-4 Claude Gemini capability alignment safety",
        "AI safety alignment RLHF constitutional AI",
        "hallucination grounding retrieval augmented generation",
        "knowledge graph reasoning embedding RDF SPARQL",
        "explainable AI XAI SHAP LIME interpretability",
        "AI ethics bias fairness accountability transparency",
        "federated learning privacy distributed training",
    ]
    n = wiki_api(queries, bucket, "ai_wiki", D, SF, AI_KW); total += n
    log.info(f"  [AI-1] Wiki API: {n}")
    n = wiki_stream(AI_KW, bucket, "ai_stream", D, SF, AI_KW,
                    max_docs=5000); total += n
    log.info(f"  [AI-2] Wiki stream: {n}")
    n = se_qa(["stackoverflow", "datascience", "stats",
               "ai", "softwareengineering"],
              bucket, "ai_se", D, SF, AI_KW,
              extra_kw={"machine learning", "deep learning", "neural",
                        "transformer", "lstm", "gradient",
                        "training", "overfitting", "regularization",
                        "pytorch", "tensorflow", "model", "nlp",
                        "classification", "regression", "cnn"}); total += n
    log.info(f"  [AI-3] SE Q&A: {n}")
    n = arxiv_cats(["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE",
                    "stat.ML"],
                   bucket, "ai_arxiv", D, SF,
                   extra_kw={"learning", "neural", "model",
                             "training", "attention", "generation",
                             "classification", "detection"},
                   max_docs=8000); total += n
    log.info(f"  [AI-4] arXiv: {n}")
    n = survey_arxiv(["cs.AI", "cs.LG", "cs.CL", "cs.CV"],
                     bucket, "ai_survey", D, SF, max_docs=3000); total += n
    log.info(f"  [AI-5] arXiv surveys: {n}")
    n = semantic_scholar([
        "transformer architecture attention mechanism NLP performance",
        "large language model GPT BERT scaling law",
        "reinforcement learning from human feedback RLHF alignment",
        "diffusion model image synthesis score matching DDPM",
        "GAN generative adversarial training stability mode",
        "federated learning privacy differential preserving",
        "graph neural network node classification link prediction",
        "few-shot learning meta-learning prompt in-context",
        "AI safety alignment specification reward hacking",
        "explainable AI interpretability SHAP attribution",
        "computer vision object detection segmentation benchmark",
        "neural architecture search NAS autoML efficiency",
        "continual learning catastrophic forgetting rehearsal",
        "multimodal learning vision language model CLIP",
        "AI ethics bias fairness dataset mitigation",
        "knowledge distillation teacher student compression",
    ], bucket, "ai_ss", D, SF); total += n
    log.info(f"  [AI-6] Semantic Scholar: {n}")
    n = openalex([
        "deep learning neural network architecture performance",
        "natural language processing transformer language model",
        "computer vision convolutional network image recognition",
        "reinforcement learning agent environment reward",
        "AI ethics bias fairness machine learning",
        "generative AI image synthesis diffusion GAN",
    ], bucket, "ai_oa", D, SF, "Artificial Intelligence"); total += n
    log.info(f"  [AI-7] OpenAlex: {n}")
    log.info(f"  >>> Artificial Intelligence total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("TitanAI Multi-Domain Corpus Loader v1 — 14 Subjects / 22 Domains")
    log.info("Sources: Wiki + Stream + SE + arXiv + SurveyArXiv + PubMed + "
             "PMC + SemanticScholar + OpenAlex + Gutenberg")
    log.info("Quality: section-aware chunking | 500-char min | dedup | "
             "keyword-density | citation gate | context headers | "
             "glossary | instruction-format")
    log.info(f"Output: {RAW}")
    log.info("")

    results = {}

    # Cyber Systems
    results["cyber_sys_network"]       = load_cyber_systems_network()
    results["cyber_sys_hw_devsecops"]  = load_cyber_systems_hardware_devsecops()

    # Cyber Security
    results["cyber_sec_pentest"]       = load_cyber_security_pentest()
    results["cyber_sec_malware"]       = load_cyber_security_malware()
    results["cyber_sec_crypto_exploit"]= load_cyber_security_crypto_exploit()

    # Web Systems
    results["web_architecture"]        = load_web_systems_architecture()
    results["web_development"]         = load_web_systems_development()

    # Coding
    results["coding_algorithms"]       = load_coding_algorithms()
    results["coding_languages"]        = load_coding_languages_architecture()

    # Psychology
    results["emotional_intelligence"]  = load_emotional_intelligence()
    results["behavioural_psychology"]  = load_behavioural_psychology()

    # Motor & Machinery
    results["motor_ice"]               = load_motor_mechanics_ice()
    results["motor_ev"]                = load_motor_mechanics_ev()
    results["machinery_heavy"]         = load_machinery_heavy()
    results["machinery_vehicles"]      = load_machinery_vehicles()

    # Aviation
    results["aviation_fixed_wing"]     = load_aviation_fixed_wing()
    results["aviation_rotorcraft"]     = load_aviation_rotorcraft()

    # Marine & Hydro
    results["marine"]                  = load_marine()
    results["hydromechanics"]          = load_hydromechanics()

    # Social & Soft Skills
    results["conflict_negotiation"]    = load_conflict_negotiation()
    results["etiquette_manners"]       = load_etiquette_manners()

    # AI
    results["artificial_intelligence"] = load_artificial_intelligence()

    with open(INVENTORY, "w") as f:
        json.dump(inventory, f, indent=2)
    with open(EXCLUSION_LOG, "w") as f:
        for e in exclusions:
            f.write(json.dumps(e) + "\n")

    log.info("")
    log.info("=" * 65)
    log.info("MULTI-DOMAIN CORPUS LOAD — COMPLETE")
    log.info("=" * 65)
    grand = 0
    for domain, n in results.items():
        log.info(f"  {domain:<35}: {n:>7,} docs")
        grand += n
    log.info(f"  {'GRAND TOTAL':<35}: {grand:>7,} docs")
    log.info("")
    log.info("Next: python scripts/generate_shards.py")


if __name__ == "__main__":
    main()
