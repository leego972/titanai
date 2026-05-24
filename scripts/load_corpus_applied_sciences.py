#!/usr/bin/env python3
"""
TitanAI Applied Sciences Corpus Loader  v2  — Maximum Depth
============================================================
Premium training data across 6 applied science domains.
Target: 80,000+ documents.

Quality standards:
  • Full Wikipedia articles (40,000 chars) with section-aware sentence-boundary chunking
  • SE Q&A — all 6 domains, multi-answer (top 3), ≥500-char answer gate
  • arXiv — standard + dedicated survey/review filter for synthesis papers
  • PMC full-text — PubMed Central open-access full papers
  • PubMed abstracts — NCBI E-utilities
  • Semantic Scholar — citation-ranked, ≥5 cite gate for quality
  • OpenAlex — citation-sorted open-access
  • NASA Technical Reports — public domain, expert aerospace content
  • Project Gutenberg — historical primary texts
  • Instruction-format wrapping — subset converted to Q&A instructional pairs
  • Context-aware headers — every doc tagged: domain, subfield, type, difficulty
  • Glossary extraction — defined terms from Wikipedia as standalone definition docs
  • Keyword-density filter — rejects off-topic hits (< 2 keyword occurrences)
  • MD5 deduplication — no repeated content across all sources
  • 500-char minimum threshold — eliminates stubs

Domains:
  1. Apothecary & Herbal Medicine
  2. Crystals, Metals & Materials Science
  3. Magnets & Applied Magnetism
  4. Complex Circuit Systems — Design & Build
  5. Lasers & Photonics
  6. Propulsion Mechanics
"""

import os, sys, json, time, logging, requests, hashlib, re
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent
RAW  = BASE / "data" / "raw"
LOG  = BASE / "data" / "corpus_applied_sciences.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(str(LOG))]
)
log = logging.getLogger("applied_sciences_loader")

INVENTORY     = BASE / "data" / "source_inventory.json"
EXCLUSION_LOG = BASE / "data" / "exclusions_applied_sciences.jsonl"
try:
    with open(INVENTORY) as f:
        inventory = json.load(f)
except Exception:
    inventory = {}
exclusions = []

_seen_hashes = set()   # Global dedup across all sources


# ── Core helpers ────────────────────────────────────────────────────────────────
def doc_hash(text):
    return hashlib.md5(text[:300].encode("utf-8", errors="ignore")).hexdigest()

def keyword_density(text, keywords, min_hits=2):
    """Reject docs where topic keywords appear fewer than min_hits times."""
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
    """Convert document to Q&A instruction-tuning format."""
    return (f"Question: Provide a comprehensive expert-level explanation of {title} "
            f"in the context of {domain}.\n\n"
            f"Answer:\n{text.strip()[:6000]}")

def chunk_text_smart(title, text, source_tag, domain, subfield,
                     chunk_size=5000, overlap=400):
    """Section-aware, sentence-boundary chunking with context headers."""
    # Split by section headers (## ...) if present
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
        # Sentence-boundary chunk
        sentences = re.split(r'(?<=[.!?])\s+', block)
        current, part = [], 0
        for sent in sentences:
            current.append(sent)
            joined = " ".join(current)
            if len(joined) >= chunk_size:
                full_chunk = hdr + joined.strip()
                chunks.append(full_chunk)
                # Keep overlap sentences
                words_back = " ".join(current).split()
                overlap_text = " ".join(words_back[-int(overlap/6):])
                current = [overlap_text]
                part += 1
        if current:
            remainder = hdr + " ".join(current).strip()
            if len(remainder) > 200:
                chunks.append(remainder)
    # Also add instruction-format version for the full text
    if len(text) > 1000:
        chunks.append(
            context_header(domain, subfield, title, "Instruction-QA") +
            "\n\n" + instruction_wrap(title, text[:6000], domain)
        )
    return chunks

def extract_glossary_terms(title, text, domain, subfield):
    """Extract bolded/defined terms as standalone definition documents."""
    docs = []
    # Find patterns like "Term is defined as..." or "**term** — definition"
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
    log.info(f"  Wrote {n} docs → {bucket_dir.name}/{tag}_*.txt")
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


# ── Source loaders ───────────────────────────────────────────────────────────────
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
                    "explaintext": True, "exsectionformat": "plain", "format": "json"
                }, timeout=15).json().get("query", {}).get("pages", {})
                for pid, pg in pages.items():
                    if pid == "-1":
                        continue
                    text = pg.get("extract", "").strip()
                    if len(text) > 800:
                        all_chunks.extend(
                            chunk_text_smart(t, text, "Wikipedia",
                                             domain, subfield)
                        )
                        # Also extract glossary terms
                        all_chunks.extend(
                            extract_glossary_terms(t, text, domain, subfield)
                        )
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
                                     domain, subfield)
                )
                all_chunks.extend(
                    extract_glossary_terms(title, text, domain, subfield)
                )
            if len(all_chunks) >= max_docs * 4:
                break
        return write_docs(bucket_dir, all_chunks, tag, domain_kws)
    except Exception as e:
        record_exclusion(f"wiki_stream:{tag}", str(e))
        return 0

def se_qa(domain_kws_se, bucket_dir, tag, domain, subfield, topic_kws,
          extra_kw=None, max_docs=5000):
    """Full Q&A with top-3 answers, ≥500-char gate, context headers."""
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
                    f"Answer (score {a.get('pm_score',0)}):\n"
                    f"{a.get('text','').strip()[:4000]}"
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

def survey_arxiv(cats, bucket_dir, tag, domain, subfield, max_docs=3000):
    """Dedicated filter for survey, review, and tutorial papers — gold for training."""
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
    """NCBI PubMed abstracts — high-quality peer-reviewed content."""
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
    """PubMed Central — full-text open-access papers via NCBI API."""
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
                # Split by article (PMC returns concatenated)
                articles = re.split(r'\n={10,}\n', fetch.text)
                for art in articles:
                    art = art.strip()
                    if len(art) > 600:
                        docs.extend(
                            chunk_text_smart(
                                f"PMC: {q[:50]}", art, "PMC-fulltext",
                                domain, subfield
                            )
                        )
            time.sleep(0.6)
        except Exception as e:
            log.warning(f"    PMC '{q[:35]}': {e}")
    return write_docs(bucket_dir, docs, tag)

def semantic_scholar(queries, bucket_dir, tag, domain, subfield,
                     max_per_query=80, min_citations=5):
    """Semantic Scholar — citation-gated (≥5 cites) academic papers."""
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

def gutenberg(ids, bucket_dir, tag, domain, subfield):
    all_chunks = []
    for gid, desc in ids:
        for url in [
            f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
            f"https://gutenberg.org/files/{gid}/{gid}-0.txt",
            f"https://gutenberg.org/files/{gid}/{gid}.txt",
        ]:
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200 and len(r.text) > 500:
                    text = gutenberg_strip(r.text)
                    all_chunks.extend(
                        chunk_text_smart(desc, text, f"Gutenberg#{gid}",
                                         domain, subfield)
                    )
                    log.info(f"    Gutenberg #{gid}: {len(text):,} chars")
                    break
            except Exception as e:
                log.warning(f"    Gutenberg #{gid}: {e}")
        time.sleep(0.6)
    return write_docs(bucket_dir, all_chunks, tag)

def nasa_reports(queries, bucket_dir, tag, domain="Propulsion", subfield="Aerospace"):
    """NASA Technical Reports Server — public domain, expert aerospace content."""
    docs = []
    base = "https://ntrs.nasa.gov/api/citations/search"
    for q in queries:
        try:
            r = requests.get(base, params={
                "q": q, "rows": 50, "sort": "relevance"
            }, timeout=20, headers={"User-Agent": "TitanAI/1.0"})
            if r.status_code != 200:
                continue
            for item in r.json().get("results", []):
                title = (item.get("title", "") or "").strip()
                abst  = (item.get("abstract", "") or "").strip()
                year  = (item.get("publicationDate", "") or "")[:4]
                if len(abst) > 200:
                    docs.append(
                        f"{context_header(domain, subfield, title[:60], 'NASA-TechReport', 'Expert')}\n\n"
                        f"[NASA Technical Reports | Year: {year}]\n\n"
                        f"Title: {title}\n\nAbstract:\n{abst}"
                    )
        except Exception as e:
            log.warning(f"    NASA '{q[:35]}': {e}")
        time.sleep(0.4)
    return write_docs(bucket_dir, docs, tag)


# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD SETS
# ══════════════════════════════════════════════════════════════════════════════
HERB_KW = {
    "herbal medicine", "medicinal plants", "ethnobotany", "pharmacognosy",
    "phytotherapy", "botanical medicine", "traditional medicine", "ayurveda",
    "traditional chinese medicine", "unani medicine", "apothecary",
    "materia medica", "plant alkaloids", "phytochemicals", "essential oils",
    "tincture", "decoction", "adaptogen", "echinacea", "valerian",
    "ginseng", "turmeric curcumin", "chamomile", "lavender", "peppermint",
    "elderberry", "ashwagandha", "plant secondary metabolites", "flavonoids",
    "terpenes", "saponins", "medicinal herb", "herbal pharmacopoeia",
}
CRYSTAL_KW = {
    "crystallography", "crystal structure", "crystal system", "unit cell",
    "x-ray diffraction", "bragg law", "crystal lattice", "miller indices",
    "gemology", "mineral properties", "quartz crystal", "diamond structure",
    "semiconductor crystal", "silicon crystal", "doping", "metallurgy",
    "alloy", "steel microstructure", "heat treatment", "titanium alloy",
    "nickel superalloy", "copper properties", "nanomaterials", "nanocrystal",
    "quantum dot", "graphene", "piezoelectric", "ferroelectric", "pyroelectric",
    "superconductor", "ceramic material", "biomaterial", "optical crystal",
    "nonlinear optics crystal", "liquid crystal",
}
MAGNET_KW = {
    "electromagnetism", "magnetic field", "permanent magnet",
    "neodymium magnet", "ferromagnetism", "paramagnetism", "diamagnetism",
    "superconducting magnet", "mri magnetic resonance", "electric motor",
    "generator electromagnetic", "transformer magnetic", "maglev",
    "magnetic storage", "spintronics", "hall effect", "electromagnet",
    "faraday law", "magnetic confinement fusion", "magnetohydrodynamics",
    "geomagnetic field", "magnetic nanoparticle",
}
CIRCUIT_KW = {
    "electronic circuit", "circuit design", "analog circuit", "digital circuit",
    "integrated circuit", "vlsi design", "fpga", "pcb layout",
    "operational amplifier", "filter design", "oscillator circuit",
    "power electronics", "switching power supply", "inverter",
    "rf circuit", "microwave circuit", "transistor", "logic gate",
    "adc dac", "microcontroller", "embedded systems", "spice simulation",
    "ic fabrication cmos", "thermal management", "signal integrity",
}
LASER_KW = {
    "laser physics", "stimulated emission", "laser cavity", "co2 laser",
    "nd yag laser", "fiber laser", "diode laser", "excimer laser",
    "femtosecond laser", "ultrafast laser", "laser surgery", "lasik",
    "photodynamic therapy", "lidar", "laser cutting", "laser welding",
    "optical fiber", "holography", "laser interferometry", "photonic",
    "optical amplifier", "nonlinear optics", "raman spectroscopy",
    "laser cooling", "optical tweezers",
}
PROPULSION_KW = {
    "rocket propulsion", "specific impulse", "thrust", "solid propellant",
    "liquid propellant", "ion thruster", "hall effect thruster",
    "nuclear thermal rocket", "jet engine", "turbojet", "turbofan",
    "ramjet", "scramjet", "pulse detonation", "hypersonic", "nozzle design",
    "rocket staging", "orbital mechanics", "spacecraft", "launch vehicle",
    "combustion chamber", "propeller aerodynamics", "marine propulsion",
    "electric aircraft propulsion",
}


# ══════════════════════════════════════════════════════════════════════════════
# SUBFIELD LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_apothecary_herbal_medicine():
    log.info("=" * 65)
    log.info("APP-1: Apothecary & Herbal Medicine")
    log.info("=" * 65)
    D, SF = "Applied Sciences", "Herbal Medicine & Apothecary"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "pharmacognosy medicinal plant active compounds mechanisms",
        "ethnobotany traditional plant medicine indigenous knowledge",
        "Ayurveda herbal formulations doshas pharmacology",
        "Traditional Chinese Medicine herbs phytochemistry mechanisms",
        "Unani Tibb Islamic medicine materia medica plants",
        "plant alkaloids morphine quinine berberine biosynthesis",
        "flavonoids polyphenols antioxidant mechanisms bioavailability",
        "essential oils terpenes antimicrobial therapeutic properties",
        "adaptogen ashwagandha ginseng rhodiola HPA axis cortisol",
        "turmeric curcumin NF-kB anti-inflammatory mechanisms",
        "echinacea immune modulation clinical trials",
        "valerian passionflower GABA sleep sedation mechanism",
        "St John's wort hyperforin serotonin reuptake depression",
        "milk thistle silymarin hepatoprotective liver oxidative stress",
        "garlic allicin cardiovascular antimicrobial mechanisms",
        "ginkgo biloba memory cerebral blood flow flavone terpene",
        "elderberry Sambucus antiviral neuraminidase immune",
        "cannabis cannabinoids endocannabinoid system therapeutic",
        "opium history morphine laudanum apothecary pharmaceutical",
        "phytochemical extraction isolation HPLC NMR methods",
        "herbal drug interactions cytochrome P450 metabolism",
        "WHO traditional medicine safety efficacy guidelines",
        "plant cell culture secondary metabolite elicitation",
        "natural product drug discovery antibiotic antifungal",
        "saponins glycosides tannins plant defense chemistry",
        "medicinal mushrooms reishi chaga immunomodulatory beta-glucan",
        "historical apothecary pharmacy Dioscorides Galen",
        "phytotherapy clinical evidence systematic review",
        "Amazonian medicinal plants rainforest ethnobotany",
        "African traditional medicine Ubuntu healing plants",
    ]
    n = wiki_api(queries, bucket, "app_herb_wiki", D, SF, HERB_KW); total += n
    log.info(f"  [HERB-1] Wiki API: {n}")
    n = wiki_stream(HERB_KW, bucket, "app_herb_stream", D, SF, HERB_KW,
                    max_docs=5000); total += n
    log.info(f"  [HERB-2] Wiki stream: {n}")
    n = se_qa(["biology", "chemistry", "medicalsciences", "health"],
              bucket, "app_herb_se", D, SF, HERB_KW,
              extra_kw={"herb", "plant", "medicinal", "ayurved", "tcm",
                        "phytochem", "alkaloid", "essential oil", "tincture",
                        "adaptogen", "traditional medicine", "supplement"}); total += n
    log.info(f"  [HERB-3] SE Q&A: {n}")
    n = gutenberg([
        (4698,  "Culpeper's Complete Herbal — Nicholas Culpeper"),
        (21214, "The Herbalist — Meyer"),
        (27948, "A Modern Herbal — Mrs. M. Grieve"),
        (36988, "Herbals: Their Origin and Evolution — Arber"),
        (22253, "The Secrets of the Plants — traditional herbalism"),
    ], bucket, "app_herb_gutenberg", D, SF); total += n
    log.info(f"  [HERB-4] Gutenberg: {n}")
    n = pubmed([
        "pharmacognosy medicinal plant active compound isolation",
        "ethnobotany traditional medicine biological activity",
        "curcumin anti-inflammatory clinical trial randomized",
        "adaptogen ashwagandha stress cortisol clinical",
        "herbal medicine drug interaction pharmacokinetics",
        "natural product antibiotic drug discovery",
        "phytotherapy systematic review clinical evidence",
        "traditional Chinese medicine herbs mechanism action",
        "cannabis cannabidiol therapeutic epilepsy pain",
        "plant alkaloid biosynthesis pathway elucidation",
    ], bucket, "app_herb_pubmed", D, SF); total += n
    log.info(f"  [HERB-5] PubMed: {n}")
    n = pmc_fulltext([
        "herbal medicine pharmacology mechanism review",
        "phytochemical bioactive compound anticancer",
        "ethnobotany medicinal plants Africa Asia",
        "adaptogen stress response HPA axis",
    ], bucket, "app_herb_pmc", D, SF); total += n
    log.info(f"  [HERB-6] PMC full-text: {n}")
    n = arxiv_cats(["q-bio.QM", "q-bio.MN", "q-bio.SC"], bucket,
                   "app_herb_arxiv", D, SF,
                   extra_kw={"herbal", "plant", "phytochem", "alkaloid",
                             "medicinal", "ethnobotany", "natural product",
                             "botanical", "traditional medicine"},
                   max_docs=2000); total += n
    log.info(f"  [HERB-7] arXiv: {n}")
    n = semantic_scholar([
        "pharmacognosy medicinal plant secondary metabolites mechanism",
        "ethnobotany traditional knowledge medicinal plants validation",
        "curcumin anti-inflammatory NF-kB mechanism bioavailability",
        "plant alkaloid biosynthesis pathway engineering",
        "herbal medicine drug interaction cytochrome P450",
        "natural product drug discovery antibiotic pipeline",
        "adaptogen ashwagandha cortisol clinical randomized",
        "traditional Chinese medicine phytochemistry clinical",
        "cannabis endocannabinoid therapeutic applications",
        "essential oil antimicrobial terpene mechanism",
    ], bucket, "app_herb_ss", D, SF); total += n
    log.info(f"  [HERB-8] Semantic Scholar: {n}")
    n = openalex([
        "medicinal plants anti-inflammatory antioxidant activity",
        "ethnobotany indigenous knowledge drug discovery",
        "phytotherapy clinical trial herbal medicine",
        "plant alkaloid pharmacology mechanism",
    ], bucket, "app_herb_oa", D, SF, "Herbal Medicine"); total += n
    log.info(f"  [HERB-9] OpenAlex: {n}")
    record_source("corpus_C_technical", "Herbal Medicine Multi-source", "",
                  0, "9 sources: Wiki+stream+SE+Gutenberg+PubMed+PMC+arXiv+SS+OA", total)
    log.info(f"  >>> Herbal Medicine total: {total:,}")
    return total


def load_crystals_metals_materials():
    log.info("=" * 65)
    log.info("APP-2: Crystals, Metals & Materials Science")
    log.info("=" * 65)
    D, SF = "Applied Sciences", "Crystals, Metals & Materials"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "crystal structure symmetry space group Bravais lattice",
        "X-ray diffraction crystallography Bragg law powder",
        "semiconductor silicon germanium band gap doping",
        "diamond crystal structure cubic hardness thermal",
        "quartz piezoelectric crystal oscillator properties",
        "gemstone mineralogy optical properties refraction",
        "metallurgy iron steel microstructure phase diagram",
        "titanium alloy aerospace biomedical osseointegration",
        "nickel superalloy turbine blade creep high temperature",
        "copper electrical conductivity annealing applications",
        "aluminum alloy lightweight aerospace automotive",
        "shape memory alloy nitinol superelasticity biomedical",
        "steel heat treatment quenching tempering martensite",
        "nanomaterials quantum confinement size-dependent properties",
        "graphene electronic thermal mechanical properties",
        "carbon nanotube chirality electrical mechanical",
        "superconductor BCS Cooper pairs high temperature cuprate",
        "ceramic alumina zirconia toughening mechanisms",
        "piezoelectric material sensor actuator MEMS",
        "ferroelectric perovskite barium titanate polarization",
        "liquid crystal display nematic cholesteric",
        "optical glass refractive index Abbe number",
        "nonlinear optical crystal KTP lithium niobate SHG",
        "biomaterial scaffold tissue engineering biocompatibility",
        "corrosion electrochemistry protection coatings",
        "powder metallurgy sintering hot pressing SPS",
        "thin film PVD CVD ALD deposition semiconductor",
        "transmission electron microscopy TEM crystal defects",
        "density functional theory DFT materials properties",
        "high entropy alloy multi-principal composition",
    ]
    n = wiki_api(queries, bucket, "app_cryst_wiki", D, SF, CRYSTAL_KW); total += n
    log.info(f"  [CRYST-1] Wiki API: {n}")
    n = wiki_stream(CRYSTAL_KW, bucket, "app_cryst_stream", D, SF, CRYSTAL_KW,
                    max_docs=5000); total += n
    log.info(f"  [CRYST-2] Wiki stream: {n}")
    n = se_qa(["chemistry", "physics", "materials", "engineering"],
              bucket, "app_cryst_se", D, SF, CRYSTAL_KW,
              extra_kw={"crystal", "metal", "alloy", "semiconductor", "material",
                        "ceramic", "polymer", "composite", "nano", "thin film",
                        "phase diagram", "diffraction", "lattice", "defect"}); total += n
    log.info(f"  [CRYST-3] SE Q&A: {n}")
    n = arxiv_cats(["cond-mat.mtrl-sci", "cond-mat.supr-con",
                    "cond-mat.mes-hall", "cond-mat.str-el",
                    "physics.app-ph"], bucket, "app_cryst_arxiv", D, SF,
                   max_docs=6000); total += n
    log.info(f"  [CRYST-4] arXiv: {n}")
    n = survey_arxiv(["cond-mat.mtrl-sci", "cond-mat.supr-con",
                      "physics.app-ph"], bucket,
                     "app_cryst_survey", D, SF, max_docs=2000); total += n
    log.info(f"  [CRYST-5] arXiv surveys: {n}")
    n = pubmed([
        "biomaterial implant osseointegration titanium clinical",
        "nanoparticle drug delivery tumor targeting cancer",
        "piezoelectric biosensor MEMS medical applications",
        "graphene biomedical applications toxicity review",
        "crystal structure protein drug target binding",
    ], bucket, "app_cryst_pubmed", D, SF); total += n
    log.info(f"  [CRYST-6] PubMed: {n}")
    n = semantic_scholar([
        "crystal structure prediction machine learning DFT",
        "high entropy alloy mechanical hardness microstructure",
        "2D materials graphene MoS2 electronic properties",
        "perovskite solar cell efficiency stability mechanism",
        "superconductor high temperature cuprate mechanism",
        "biomaterial scaffold tissue engineering stem cells",
        "corrosion inhibitor mechanism protection efficiency",
        "nanomaterial synthesis characterization biomedical",
        "piezoelectric energy harvesting wearable",
        "thin film deposition ALD semiconductor",
    ], bucket, "app_cryst_ss", D, SF); total += n
    log.info(f"  [CRYST-7] Semantic Scholar: {n}")
    n = openalex([
        "materials science crystal structure properties",
        "nanomaterials synthesis characterization applications",
        "biomaterials tissue engineering scaffold",
        "semiconductor doping electrical properties",
    ], bucket, "app_cryst_oa", D, SF, "Materials Science"); total += n
    log.info(f"  [CRYST-8] OpenAlex: {n}")
    record_source("corpus_C_technical", "Materials Science Multi-source", "", 0,
                  "8 sources", total)
    log.info(f"  >>> Crystals, Metals & Materials total: {total:,}")
    return total


def load_magnets_applied_magnetism():
    log.info("=" * 65)
    log.info("APP-3: Magnets & Applied Magnetism")
    log.info("=" * 65)
    D, SF = "Applied Sciences", "Magnets & Applied Magnetism"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "permanent magnet neodymium NdFeB coercivity remanence",
        "ferromagnetism Weiss domain theory exchange interaction",
        "magnetic hysteresis B-H curve coercivity saturation",
        "electromagnetic induction Faraday Lenz law applications",
        "electric motor torque speed flux winding design",
        "AC induction motor rotating magnetic field slip",
        "synchronous motor BLDC permanent magnet control",
        "transformer magnetic core losses lamination eddy current",
        "MRI magnetic resonance imaging RF pulses gradients",
        "superconducting magnet cryogenic ITER fusion",
        "magnetic levitation maglev Meissner effect",
        "hard disk drive magnetic recording perpendicular",
        "spintronics giant magnetoresistance GMR sensor",
        "magnetic tunnel junction TMR spin transfer torque MRAM",
        "Hall effect sensor current position measurement",
        "magnetohydrodynamics MHD propulsion electromagnetic",
        "magnetic confinement fusion tokamak coils plasma",
        "geomagnetic field dynamo paleomagnetic reversal",
        "magnetic nanoparticle hyperthermia targeted cancer",
        "electromagnetic compatibility shielding EMI",
        "eddy current nondestructive testing inspection",
        "magnetocaloric effect magnetic refrigeration",
        "wireless power transfer inductive coupling resonant",
        "Maxwell equations electromagnetic wave propagation",
        "magnetic monopole search physics",
    ]
    n = wiki_api(queries, bucket, "app_mag_wiki", D, SF, MAGNET_KW); total += n
    log.info(f"  [MAG-1] Wiki API: {n}")
    n = wiki_stream(MAGNET_KW, bucket, "app_mag_stream", D, SF, MAGNET_KW,
                    max_docs=4000); total += n
    log.info(f"  [MAG-2] Wiki stream: {n}")
    n = se_qa(["physics", "electronics", "engineering"],
              bucket, "app_mag_se", D, SF, MAGNET_KW,
              extra_kw={"magnet", "magnetic", "electromagnet", "motor",
                        "transformer", "induction", "mri", "spintronics",
                        "hall effect", "maglev", "faraday", "flux"}); total += n
    log.info(f"  [MAG-3] SE Q&A: {n}")
    n = arxiv_cats(["cond-mat.str-el", "cond-mat.supr-con",
                    "cond-mat.mes-hall", "physics.app-ph",
                    "eess.SY"], bucket, "app_mag_arxiv", D, SF,
                   extra_kw={"magnet", "spin", "ferromagnet", "electromagnet",
                             "mri", "motor", "maglev", "magnetic", "hall"},
                   max_docs=4000); total += n
    log.info(f"  [MAG-4] arXiv: {n}")
    n = survey_arxiv(["cond-mat.str-el", "cond-mat.mes-hall",
                      "physics.app-ph"], bucket,
                     "app_mag_survey", D, SF, max_docs=1500); total += n
    log.info(f"  [MAG-5] arXiv surveys: {n}")
    n = pubmed([
        "MRI physics image quality artifact reduction",
        "magnetic nanoparticle hyperthermia cancer therapy",
        "transcranial magnetic stimulation TMS neurology",
        "magnetic resonance spectroscopy metabolite brain",
        "magnetotherapy clinical evidence pain healing",
    ], bucket, "app_mag_pubmed", D, SF); total += n
    log.info(f"  [MAG-6] PubMed: {n}")
    n = semantic_scholar([
        "permanent magnet motor efficiency torque design",
        "spintronics magnetic memory MRAM spin orbit",
        "MRI gradient coil image reconstruction artifact",
        "magnetic nanoparticle biomedical hyperthermia cancer",
        "maglev levitation control stability Meissner",
        "tokamak magnetic confinement plasma fusion ITER",
        "giant magnetoresistance sensor applications automotive",
        "wireless power transfer magnetic resonance coupling",
        "magnetocaloric refrigeration near room temperature",
        "eddy current loss reduction amorphous core",
    ], bucket, "app_mag_ss", D, SF); total += n
    log.info(f"  [MAG-7] Semantic Scholar: {n}")
    n = openalex([
        "electromagnetism motors generators efficiency",
        "magnetic materials properties applications",
        "MRI physics biomedical imaging",
        "spintronics spintronic devices memory",
    ], bucket, "app_mag_oa", D, SF, "Applied Magnetism"); total += n
    log.info(f"  [MAG-8] OpenAlex: {n}")
    log.info(f"  >>> Magnets & Applied Magnetism total: {total:,}")
    return total


def load_complex_circuit_systems():
    log.info("=" * 65)
    log.info("APP-4: Complex Circuit Systems — Design & Build")
    log.info("=" * 65)
    D, SF = "Applied Sciences", "Complex Circuit Systems"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "analog circuit design operational amplifier feedback stability",
        "Butterworth Chebyshev Bessel filter design poles zeros",
        "oscillator crystal LC RC Colpitts Hartley design",
        "class A B AB D amplifier efficiency linearity bias",
        "CMOS inverter logic gate delay power consumption",
        "VLSI standard cell place route timing optimization",
        "FPGA LUT configurable logic block architecture Xilinx",
        "PCB layout signal integrity controlled impedance",
        "power MOSFET gate drive switching loss dead time",
        "buck boost flyback converter duty cycle regulation",
        "LLC resonant converter soft switching topology",
        "three-phase inverter SVPWM motor drive VFD",
        "RF impedance matching S-parameters Smith chart",
        "low noise amplifier LNA noise figure gain flatness",
        "phased array antenna beamforming digital analog",
        "ADC successive approximation sigma-delta pipeline",
        "phase-locked loop PLL VCO charge pump bandwidth",
        "SPICE netlist simulation AC DC transient analysis",
        "static timing analysis setup hold slack violation",
        "CMOS process node FinFET gate-all-around scaling",
        "power integrity PDN decoupling capacitor resonance",
        "EMI conducted radiated emission filter ferrite",
        "SerDes high-speed serial link equalization CDR",
        "thermal resistance junction ambient heat sink design",
        "system-on-chip SoC NoC bus protocol AXI",
        "neuromorphic computing spiking neural network chip",
        "in-memory computing resistive RAM compute-in-memory",
        "mixed-signal layout mismatch matching common centroid",
        "millimeter wave 5G 6G phased array beamforming IC",
        "functional safety IEC 61508 automotive ASIL",
    ]
    n = wiki_api(queries, bucket, "app_circ_wiki", D, SF, CIRCUIT_KW); total += n
    log.info(f"  [CIRC-1] Wiki API: {n}")
    n = wiki_stream(CIRCUIT_KW, bucket, "app_circ_stream", D, SF, CIRCUIT_KW,
                    max_docs=5000); total += n
    log.info(f"  [CIRC-2] Wiki stream: {n}")
    # Electronics StackExchange is the RICHEST source for circuit design
    n = se_qa(["electronics", "engineering", "physics"],
              bucket, "app_circ_se", D, SF, CIRCUIT_KW,
              extra_kw={"circuit", "amplifier", "filter", "oscillator",
                        "power supply", "transistor", "mosfet", "fpga",
                        "pcb", "signal", "impedance", "adc", "dac",
                        "rf", "microwave", "vlsi", "cmos", "op-amp",
                        "inverter", "converter", "pwm", "emi"}); total += n
    log.info(f"  [CIRC-3] SE Q&A (electronics.SE): {n}")
    n = arxiv_cats(["eess.SP", "eess.SY", "cs.ET", "physics.app-ph",
                    "eess.EE"], bucket, "app_circ_arxiv", D, SF,
                   extra_kw={"circuit", "vlsi", "fpga", "analog", "digital",
                             "amplifier", "power", "rf", "filter", "adc",
                             "cmos", "chip", "semiconductor", "pcb"},
                   max_docs=4000); total += n
    log.info(f"  [CIRC-4] arXiv: {n}")
    n = survey_arxiv(["eess.SP", "eess.SY", "cs.ET"],
                     bucket, "app_circ_survey", D, SF, max_docs=2000); total += n
    log.info(f"  [CIRC-5] arXiv surveys: {n}")
    n = semantic_scholar([
        "CMOS analog circuit noise power efficiency design",
        "FPGA architecture reconfigurable computing logic",
        "power converter switching loss efficiency topology",
        "RF transceiver integrated circuit 5G mmWave",
        "PCB signal integrity high-speed differential",
        "neuromorphic chip spiking neural network VLSI",
        "in-memory computing RRAM PIM architecture",
        "millimeter wave beamforming phased array CMOS",
        "ADC high speed low power resolution",
        "mixed signal layout mismatch offset correction",
        "electromigration reliability IC interconnect",
        "three-dimensional IC TSV stacking thermal",
    ], bucket, "app_circ_ss", D, SF); total += n
    log.info(f"  [CIRC-6] Semantic Scholar: {n}")
    n = openalex([
        "analog digital circuit design integrated",
        "power electronics converter efficiency",
        "VLSI CMOS chip fabrication design",
        "RF microwave circuit communication",
    ], bucket, "app_circ_oa", D, SF, "Circuit Engineering"); total += n
    log.info(f"  [CIRC-7] OpenAlex: {n}")
    log.info(f"  >>> Complex Circuit Systems total: {total:,}")
    return total


def load_lasers_photonics():
    log.info("=" * 65)
    log.info("APP-5: Lasers & Their Applications in Modern Technology")
    log.info("=" * 65)
    D, SF = "Applied Sciences", "Lasers & Photonics"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "laser physics stimulated emission population inversion gain",
        "laser cavity resonator longitudinal transverse modes",
        "CO2 laser 10.6um industrial cutting welding marking",
        "Nd:YAG laser pulsed Q-switch harmonic generation",
        "fiber laser Yb Er rare-earth doped amplification",
        "semiconductor diode laser Fabry-Perot DFB VCSEL",
        "excimer laser UV lithography 193nm ArF KrF",
        "femtosecond laser ultrafast nonlinear ablation",
        "Ti:Sapphire laser tunable broadband spectroscopy",
        "laser eye surgery LASIK PRK excimer correction",
        "photodynamic therapy photosensitizer ROS cancer",
        "LiDAR laser ranging SLAM autonomous vehicle",
        "laser cutting manufacturing precision kerf HAZ",
        "laser welding penetration keyhole automotive",
        "laser marking engraving surface material",
        "laser spectroscopy LIBS CARS absorption emission",
        "Raman spectroscopy Stokes anti-Stokes molecular",
        "DWDM wavelength division multiplexing fiber optic",
        "EDFA erbium doped fiber amplifier gain spectrum",
        "holography wavefront phase reconstruction",
        "laser interferometry LIGO gravitational wave detection",
        "OCT optical coherence tomography depth imaging",
        "silicon photonics waveguide modulator photodetector",
        "second harmonic generation sum frequency nonlinear",
        "optical tweezers gradient force trapping manipulation",
        "laser cooling Doppler sub-Doppler magneto-optical trap",
        "free electron laser synchrotron X-ray FEL",
        "directed energy high power laser atmospheric",
        "photonic crystal fiber hollow core bandgap",
        "frequency comb optical clock precision metrology",
    ]
    n = wiki_api(queries, bucket, "app_laser_wiki", D, SF, LASER_KW); total += n
    log.info(f"  [LASER-1] Wiki API: {n}")
    n = wiki_stream(LASER_KW, bucket, "app_laser_stream", D, SF, LASER_KW,
                    max_docs=5000); total += n
    log.info(f"  [LASER-2] Wiki stream: {n}")
    n = se_qa(["physics", "electronics", "engineering"],
              bucket, "app_laser_se", D, SF, LASER_KW,
              extra_kw={"laser", "photon", "optical", "fiber", "lidar",
                        "nonlinear", "coherent", "ultrafast", "spectroscopy",
                        "waveguide", "photonic", "holograph", "interferometer",
                        "raman", "excimer", "amplifier", "mode-lock"}); total += n
    log.info(f"  [LASER-3] SE Q&A: {n}")
    n = arxiv_cats(["physics.optics", "quant-ph", "eess.SP",
                    "physics.atom-ph", "physics.app-ph"],
                   bucket, "app_laser_arxiv", D, SF,
                   extra_kw={"laser", "photon", "optical", "fiber", "lidar",
                             "nonlinear", "coherent", "ultrafast", "spectroscopy"},
                   max_docs=6000); total += n
    log.info(f"  [LASER-4] arXiv: {n}")
    n = survey_arxiv(["physics.optics", "quant-ph", "physics.app-ph"],
                     bucket, "app_laser_survey", D, SF, max_docs=2000); total += n
    log.info(f"  [LASER-5] arXiv surveys: {n}")
    n = pubmed([
        "laser surgery LASIK corneal refractive outcome",
        "photodynamic therapy cancer clinical trial",
        "low-level laser therapy wound healing pain",
        "optical coherence tomography retinal diagnosis",
        "laser tissue interaction ablation photothermal",
        "laser dermatology skin treatment tattoo hair",
    ], bucket, "app_laser_pubmed", D, SF); total += n
    log.info(f"  [LASER-6] PubMed: {n}")
    n = pmc_fulltext([
        "laser photodynamic therapy cancer mechanism",
        "optical coherence tomography clinical application",
        "low level laser therapy healing review",
    ], bucket, "app_laser_pmc", D, SF); total += n
    log.info(f"  [LASER-7] PMC full-text: {n}")
    n = semantic_scholar([
        "ultrafast femtosecond laser material processing ablation",
        "LiDAR autonomous vehicle SLAM point cloud",
        "silicon photonics integrated circuit chip modulator",
        "optical coherence tomography retinal imaging diagnosis",
        "fiber laser high power beam quality brightness",
        "laser cooling ultracold atoms BEC quantum gas",
        "nonlinear photonics frequency comb microresonator",
        "directed energy laser weapon atmospheric propagation",
        "Raman spectroscopy tissue cancer in vivo diagnosis",
        "photonic quantum computing linear optical qubit",
        "laser metrology frequency standard atomic clock",
        "photovoltaic solar concentrator laser spectrum",
    ], bucket, "app_laser_ss", D, SF); total += n
    log.info(f"  [LASER-8] Semantic Scholar: {n}")
    n = openalex([
        "laser physics applications photonics",
        "optical fiber communication wavelength",
        "laser medical surgery therapeutic",
        "photonic integrated circuit chip",
    ], bucket, "app_laser_oa", D, SF, "Laser Physics"); total += n
    log.info(f"  [LASER-9] OpenAlex: {n}")
    log.info(f"  >>> Lasers & Photonics total: {total:,}")
    return total


def load_propulsion_mechanics():
    log.info("=" * 65)
    log.info("APP-6: Propulsion Mechanics")
    log.info("=" * 65)
    D, SF = "Applied Sciences", "Propulsion Mechanics"
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "rocket propulsion Tsiolkovsky rocket equation delta-v",
        "solid rocket motor propellant grain HTPB ammonium perchlorate",
        "liquid propellant bipropellant LOX kerosene LH2",
        "de Laval nozzle supersonic expansion thrust coefficient",
        "specific impulse Isp efficiency comparison propulsion",
        "ion thruster xenon Hall effect electrostatic acceleration",
        "Hall effect thruster plasma discharge magnetic field",
        "nuclear thermal rocket hydrogen propellant NERVA",
        "solar sail photon pressure trajectory deep space",
        "turbojet thermodynamic cycle Brayton compressor turbine",
        "turbofan high bypass ratio fan pressure ratio efficiency",
        "turboprop shaft power propeller gearbox turbine",
        "ramjet supersonic combustion inlet diffuser",
        "scramjet hypersonic inlet shock-induced combustion",
        "rotating detonation engine pressure gain combustion",
        "pulse detonation engine detonation wave cycle",
        "aerospike nozzle altitude compensating plug",
        "multi-stage rocket staging separation jettison mass",
        "orbital mechanics Hohmann transfer bi-elliptic",
        "spacecraft attitude control reaction wheels thrusters",
        "reentry vehicle ablative TPS heat shield materials",
        "hypersonic aerothermodynamics boundary layer heating",
        "propeller blade aerodynamics lift drag pitch twist",
        "marine propulsion cavitation thrust efficiency",
        "underwater propulsion pump jet submarine drag",
        "helicopter rotor blade vortex induced velocity",
        "turbine blade cooling film internal channel",
        "combustion instability chugging screaming Rayleigh",
        "electric aircraft propulsion distributed motor",
        "VASIMR variable Isp magnetoplasma rocket",
    ]
    n = wiki_api(queries, bucket, "app_prop_wiki", D, SF, PROPULSION_KW); total += n
    log.info(f"  [PROP-1] Wiki API: {n}")
    n = wiki_stream(PROPULSION_KW, bucket, "app_prop_stream", D, SF,
                    PROPULSION_KW, max_docs=5000); total += n
    log.info(f"  [PROP-2] Wiki stream: {n}")
    n = se_qa(["space", "physics", "engineering", "aviation"],
              bucket, "app_prop_se", D, SF, PROPULSION_KW,
              extra_kw={"rocket", "propulsion", "thrust", "nozzle", "turbine",
                        "jet engine", "combustion", "hypersonic", "ion",
                        "spacecraft", "launch", "propellant", "scramjet",
                        "ramjet", "specific impulse", "turbofan"}); total += n
    log.info(f"  [PROP-3] SE Q&A (space.SE): {n}")
    # NASA Technical Reports — public domain expert aerospace content
    n = nasa_reports([
        "rocket nozzle design thrust coefficient",
        "ion thruster xenon performance",
        "combustion instability liquid rocket",
        "hypersonic aerothermodynamics heat transfer",
        "turbofan engine performance optimization",
        "solid rocket motor grain design burning",
        "scramjet inlet combustion supersonic",
        "spacecraft attitude control thruster",
        "nuclear thermal propulsion NERVA",
        "electric propulsion Hall thruster",
        "turbine blade cooling effectiveness",
        "ramjet combustion supersonic flight",
    ], bucket, "app_prop_nasa"); total += n
    log.info(f"  [PROP-4] NASA Technical Reports: {n}")
    n = arxiv_cats(["physics.flu-dyn", "physics.app-ph", "eess.SY",
                    "astro-ph.IM"], bucket, "app_prop_arxiv", D, SF,
                   extra_kw={"rocket", "propulsion", "thrust", "nozzle",
                             "turbine", "jet", "combustion", "hypersonic",
                             "ion", "plasma", "spacecraft", "launch"},
                   max_docs=4000); total += n
    log.info(f"  [PROP-5] arXiv: {n}")
    n = survey_arxiv(["physics.flu-dyn", "physics.app-ph", "astro-ph.IM"],
                     bucket, "app_prop_survey", D, SF, max_docs=1500); total += n
    log.info(f"  [PROP-6] arXiv surveys: {n}")
    n = gutenberg([
        (4352,  "The Aeroplane Speaks — H. Barber (early aviation propulsion)"),
        (10108, "Flying Machines — W.J. Jackman (early aerodynamics)"),
    ], bucket, "app_prop_gutenberg", D, SF); total += n
    log.info(f"  [PROP-7] Gutenberg: {n}")
    n = semantic_scholar([
        "ion thruster xenon Hall effect specific impulse efficiency",
        "hypersonic scramjet combustion instability inlet",
        "solid rocket propellant burn rate aging additives",
        "turbofan engine performance efficiency blade cooling",
        "electric aircraft distributed propulsion motor efficiency",
        "rocket nozzle thrust optimization CFD erosion",
        "orbital mechanics low-thrust trajectory optimization",
        "nuclear thermal propulsion reactor core neutronics",
        "rotating detonation engine pressure gain propulsion",
        "propeller cavitation noise underwater submarine",
        "reentry vehicle ablator TPS heat flux aerothermal",
        "turbojet turbofan blade aerodynamics CFD cooling",
    ], bucket, "app_prop_ss", D, SF); total += n
    log.info(f"  [PROP-8] Semantic Scholar: {n}")
    n = openalex([
        "rocket propulsion combustion thermal efficiency",
        "jet engine turbine aerodynamics performance",
        "spacecraft orbital mechanics propulsion",
        "hypersonic vehicle aerothermal reentry",
    ], bucket, "app_prop_oa", D, SF, "Propulsion Engineering"); total += n
    log.info(f"  [PROP-9] OpenAlex: {n}")
    log.info(f"  >>> Propulsion Mechanics total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("TitanAI Applied Sciences Corpus Loader v2 — Maximum Depth")
    log.info("6 Domains × 9 Sources: Wiki+Stream+SE+arXiv+Survey-arXiv+"
             "PubMed+PMC+SemanticScholar+OpenAlex")
    log.info("Quality: full articles 40K | section-aware chunking | "
             "500-char min | dedup | keyword-density | citation gate | "
             "context headers | glossary | instruction-format | NASA TRS")
    log.info(f"Target directory: {RAW}")
    log.info("")

    results = {}
    results["herbal_medicine"]      = load_apothecary_herbal_medicine()
    results["crystals_metals"]      = load_crystals_metals_materials()
    results["magnets_applied"]      = load_magnets_applied_magnetism()
    results["circuit_systems"]      = load_complex_circuit_systems()
    results["lasers_photonics"]     = load_lasers_photonics()
    results["propulsion_mechanics"] = load_propulsion_mechanics()

    with open(INVENTORY, "w") as f:
        json.dump(inventory, f, indent=2)
    with open(EXCLUSION_LOG, "w") as f:
        for e in exclusions:
            f.write(json.dumps(e) + "\n")

    log.info("")
    log.info("=" * 65)
    log.info("APPLIED SCIENCES CORPUS LOAD — COMPLETE")
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
