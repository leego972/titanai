"""
TitanAI Business, Entrepreneurship & R&D Corpus Loader
=======================================================
Adds deep training data across two domains:

  1. Entrepreneurship & Venture Capital
       — Startup fundamentals, lean methodology, product-market fit,
         VC fund structure (LP/GP/carry), term sheets, cap tables,
         valuation (DCF/comparables/pre-money), SAFE notes, convertible
         notes, priced rounds, anti-dilution, liquidation preferences,
         pitch deck anatomy, due diligence, angel/seed/Series A-C,
         growth hacking, unit economics (CAC/LTV/burn/runway),
         exit strategies (IPO/M&A/secondary), business model canvas,
         go-to-market strategy, startup legal (incorporation/IP/equity),
         famous case studies, global ecosystems (SV/NYC/London/Asia)

  2. Research & Development
       — Scientific research methodology, hypothesis-driven research,
         technology transfer, IP & patents, R&D management & strategy,
         disruptive vs sustaining innovation (Christensen),
         open innovation (Chesbrough), stage-gate process,
         laboratory management, grant writing & research funding,
         academic vs industrial R&D, R&D in pharma/tech/defense/energy,
         research ethics, reproducibility crisis, meta-research,
         technology readiness levels (TRL), innovation ecosystems,
         spin-offs, licensing, standards development

Sources per domain:
  • Wikipedia REST API  — targeted subtopic queries
  • wikimedia/wikipedia — streaming keyword filter
  • StackExchange Q&A   — startups.se, money.se, academia.se,
                          workplace.se, economics.se
  • arXiv preprints     — econ.GN, q-fin.*, cs.GT, econ.TH
  • Project Gutenberg   — classic economics & management texts
  • OpenAlex API        — open-access research papers
  • SEC EDGAR           — S-1 IPO prospectuses (Airbnb, Uber, Lyft, etc.)

Run:
  pip install datasets requests tqdm
  python scripts/load_corpus_business_rd.py
"""

import os, sys, json, time, logging, requests
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent
RAW  = BASE / "data" / "raw"
LOG  = BASE / "data" / "corpus_business_rd.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(str(LOG))]
)
log = logging.getLogger("biz_rd_loader")

INVENTORY     = BASE / "data" / "source_inventory.json"
EXCLUSION_LOG = BASE / "data" / "exclusions_biz_rd.jsonl"
try:
    with open(INVENTORY) as f:
        inventory = json.load(f)
except Exception:
    inventory = {}
exclusions = []

# ── Helpers ────────────────────────────────────────────────────────────────────
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

def write_docs(bucket_dir, docs, tag):
    bucket_dir.mkdir(parents=True, exist_ok=True)
    start = len(list(bucket_dir.glob(f"{tag}_*.txt")))
    n = 0
    for i, text in enumerate(docs):
        if text and len(text.strip()) >= 150:
            (bucket_dir / f"{tag}_{start+i:06d}.txt").write_text(
                text.strip(), encoding="utf-8")
            n += 1
    log.info(f"  Wrote {n} docs → {bucket_dir.name}/{tag}_*.txt")
    return n

def approx_mb(docs):
    return sum(len(d.encode("utf-8")) for d in docs) / 1_048_576

def safe_get(d, *keys, default=""):
    for k in keys:
        d = d.get(k, default) if isinstance(d, dict) else default
    return d or default

def gutenberg_strip(text):
    for m in ["*** START OF","***START OF"]:
        if m in text:
            text = text[text.index(m):]
            text = text[text.index("\n")+1:]
            break
    for m in ["*** END OF","***END OF","End of Project Gutenberg"]:
        if m in text:
            text = text[:text.index(m)]
    return text

# ── Shared loaders ─────────────────────────────────────────────────────────────
def wiki_api(queries, bucket_dir, tag, max_per_query=20):
    API = "https://en.wikipedia.org/w/api.php"
    seen, docs = set(), []
    for q in queries:
        try:
            hits = requests.get(API, params={
                "action":"query","list":"search","srsearch":q,
                "srlimit":max_per_query,"format":"json","srnamespace":0
            }, timeout=15).json().get("query",{}).get("search",[])
            for h in hits:
                t = h["title"]
                if t in seen: continue
                seen.add(t)
                pages = requests.get(API, params={
                    "action":"query","titles":t,"prop":"extracts",
                    "explaintext":True,"exsectionformat":"plain","format":"json"
                }, timeout=15).json().get("query",{}).get("pages",{})
                for pid, pg in pages.items():
                    if pid == "-1": continue
                    text = pg.get("extract","").strip()
                    if len(text) > 300:
                        docs.append(f"# {t}\n\n{text[:10000]}")
                time.sleep(0.12)
        except Exception as e:
            log.warning(f"    wiki_api '{q[:40]}': {e}")
        time.sleep(0.05)
    return write_docs(bucket_dir, docs, tag)

def wiki_stream(keywords, bucket_dir, tag, max_docs=4000):
    try:
        from datasets import load_dataset
        wiki = load_dataset("wikimedia/wikipedia","20231101.en",
                            split="train",streaming=True)
        kw = {k.lower() for k in keywords}
        docs, scanned = [], 0
        for item in wiki:
            scanned += 1
            if scanned > 700_000: break
            title = item.get("title","").lower()
            text  = item.get("text","").strip()
            if len(text) < 300: continue
            snippet = (title+" "+text[:600]).lower()
            if any(k in snippet for k in kw):
                docs.append(f"# {item['title']}\n\n{text[:9000]}")
            if len(docs) >= max_docs: break
        return write_docs(bucket_dir, docs, tag)
    except Exception as e:
        record_exclusion(f"wiki_stream:{tag}", str(e)); return 0

def se_qa(domain_kws, bucket_dir, tag, extra_kw=None, max_docs=3000):
    try:
        from datasets import load_dataset
        se = load_dataset("HuggingFaceH4/stack-exchange-preferences",
                          split="train",streaming=True)
        docs = []
        for i, item in enumerate(se):
            if i > 4_000_000: break
            dom = safe_get(item,"domain")
            if not any(k in dom.lower() for k in domain_kws): continue
            q = safe_get(item,"question")
            if extra_kw and not any(k in q.lower() for k in extra_kw): continue
            answers = item.get("answers",[]) or []
            best, score = "", -999
            for a in answers:
                s = a.get("pm_score",0) or 0
                if s > score: score, best = s, a.get("text","")
            if best and len(q) > 60:
                docs.append(f"Q&A [{dom}]\n\nQuestion: {q.strip()}\n\n"
                            f"Best Answer (score {score}):\n{best.strip()[:4000]}")
            if len(docs) >= max_docs: break
        return write_docs(bucket_dir, docs, tag)
    except Exception as e:
        record_exclusion(f"se:{tag}", str(e)); return 0

def arxiv_cats(cats, bucket_dir, tag, extra_kw=None, max_docs=3000):
    try:
        from datasets import load_dataset
        arxiv = load_dataset("Cornell-University/arxiv",
                             split="train",streaming=True)
        cat_set = set(cats)
        docs = []
        for i, item in enumerate(arxiv):
            if i > 4_000_000: break
            c = set((item.get("categories","") or "").split())
            if not c.intersection(cat_set): continue
            title = (item.get("title","") or "").replace("\n"," ").strip()
            abst  = (item.get("abstract","") or "").replace("\n"," ").strip()
            if len(abst) < 80: continue
            if extra_kw:
                if not any(k in (title+" "+abst).lower() for k in extra_kw): continue
            docs.append(f"Title: {title}\nCategories: {' '.join(sorted(c))}\n\n"
                        f"Abstract:\n{abst}")
            if len(docs) >= max_docs: break
        return write_docs(bucket_dir, docs, tag)
    except Exception as e:
        record_exclusion(f"arxiv:{tag}", str(e)); return 0

def gutenberg_load(ids, bucket_dir, tag):
    docs = []
    for gid, desc in ids:
        for url in [f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
                    f"https://gutenberg.org/files/{gid}/{gid}-0.txt",
                    f"https://gutenberg.org/files/{gid}/{gid}.txt"]:
            try:
                r = requests.get(url, timeout=25)
                if r.status_code == 200 and len(r.text) > 500:
                    text  = gutenberg_strip(r.text)
                    words = text.split()
                    for s in range(0, len(words), 1800):
                        chunk = " ".join(words[s:s+1800])
                        if len(chunk) > 300:
                            docs.append(f"[{desc} — Gutenberg #{gid}]\n\n{chunk}")
                    log.info(f"    Gutenberg #{gid}: {len(words)} words"); break
            except Exception as e:
                log.warning(f"    Gutenberg #{gid}: {e}")
        time.sleep(0.6)
    return write_docs(bucket_dir, docs, tag)

def openalex(queries, bucket_dir, tag, label="Research"):
    docs = []
    for q in queries:
        try:
            r = requests.get(
                f"https://api.openalex.org/works?search={requests.utils.quote(q)}"
                "&per-page=100&filter=open_access.is_oa:true"
                "&select=title,abstract_inverted_index",
                timeout=20, headers={"User-Agent":"TitanAI/1.0"}
            )
            if r.status_code != 200: continue
            for w in r.json().get("results",[]):
                title = (w.get("title","") or "").strip()
                inv   = w.get("abstract_inverted_index") or {}
                if not inv: continue
                mx = max(p for ps in inv.values() for p in ps)
                wl = [""]*( mx+1)
                for word, ps in inv.items():
                    for p in ps: wl[p] = word
                abst = " ".join(x for x in wl if x)
                if len(abst) > 100:
                    docs.append(f"{label}\n\nTitle: {title}\n\nAbstract:\n{abst}")
        except Exception as e:
            log.warning(f"    OpenAlex '{q[:35]}': {e}")
        time.sleep(0.35)
    return write_docs(bucket_dir, docs, tag)

# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD SETS
# ══════════════════════════════════════════════════════════════════════════════
STARTUP_KW = {
    "startup","entrepreneurship","entrepreneur","venture capital","venture capitalist",
    "angel investor","seed funding","series a","series b","series c",
    "term sheet","cap table","capitalization table","pre-money valuation",
    "post-money valuation","dilution","anti-dilution","liquidation preference",
    "convertible note","safe note","priced round","equity financing",
    "product-market fit","minimum viable product","mvp","lean startup",
    "business model canvas","value proposition","customer segment",
    "go-to-market strategy","growth hacking","viral coefficient","network effect",
    "unit economics","customer acquisition cost","lifetime value","burn rate","runway",
    "pivot","iteration","agile startup","validated learning",
    "pitch deck","investor pitch","due diligence","term negotiation",
    "limited partner","general partner","management fee","carried interest",
    "fund of funds","accelerator","incubator","y combinator","techstars",
    "ipo","acquisition","exit strategy","secondary market","spac",
    "founder","co-founder","vesting","cliff","equity compensation",
    "employee stock option pool","esop","409a valuation",
    "board of directors","governance","fiduciary duty",
    "startup ecosystem","silicon valley","innovation hub","tech cluster",
}

RD_KW = {
    "research and development","r&d","scientific research","research methodology",
    "hypothesis testing","experimental design","reproducibility","peer review",
    "technology transfer","intellectual property","patent","patent filing",
    "technology readiness level","trl","innovation management","open innovation",
    "disruptive innovation","sustaining innovation","incremental innovation",
    "stage-gate process","product development lifecycle","npd",
    "research funding","grant writing","national science foundation","darpa",
    "industrial r&d","corporate research","bell labs","xerox parc","skunkworks",
    "pharmaceutical r&d","drug discovery","clinical trial phases",
    "academic research","university technology transfer","spin-off","licensing",
    "standards development","iso standard","ieee standard",
    "meta-research","systematic review","cochrane","evidence synthesis",
    "research ethics","institutional review board","irb","informed consent",
    "bibliometrics","h-index","impact factor","citation analysis",
    "innovation ecosystem","national innovation system","cluster theory",
    "r&d tax credit","r&d investment","innovation policy",
}

# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 1 — ENTREPRENEURSHIP & VENTURE CAPITAL
# ══════════════════════════════════════════════════════════════════════════════
def load_entrepreneurship_vc():
    log.info("=" * 65)
    log.info("DOMAIN 1: Entrepreneurship & Venture Capital")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries_startup = [
        # VC & fundraising mechanics
        "venture capital fund structure LP GP carry",
        "term sheet negotiation anti-dilution provisions",
        "convertible note SAFE equity financing rounds",
        "cap table management dilution equity rounds",
        "pre-money post-money valuation startup",
        "liquidation preference participating non-participating",
        "drag-along tag-along rights shareholder agreement",
        "409A valuation employee stock option plan",
        "due diligence startup investor checklist",
        "venture capital deal sourcing selection",
        "angel investor seed funding stage",
        "series A B C growth equity investment",
        "venture debt bridge financing startup",
        "secondary market startup equity liquidity",
        # Startup methodology & strategy
        "lean startup methodology validated learning pivot",
        "product market fit how to find measure",
        "minimum viable product MVP testing assumptions",
        "business model canvas Osterwalder innovation",
        "customer development Steve Blank methodology",
        "go-to-market strategy B2B SaaS playbook",
        "growth hacking viral loops acquisition",
        "unit economics SaaS LTV CAC payback period",
        "startup burn rate runway cash management",
        "network effects platform business model",
        "competitive moat defensibility startup",
        "founder vesting cliff equity compensation",
        "co-founder agreement equity split roles",
        # Ecosystems & exits
        "Y Combinator startup accelerator program",
        "Silicon Valley startup ecosystem history",
        "startup IPO S-1 prospectus process",
        "startup acquisition M&A exit strategy",
        "SPAC special purpose acquisition company",
        "startup failure post-mortem lessons",
        "unicorn startup billion dollar valuation",
        "emerging markets startup ecosystem Africa Asia",
        # Pitch & storytelling
        "startup pitch deck structure investor",
        "storytelling narrative fundraising pitch",
        "startup metrics KPIs investor reporting",
        "board of directors governance fiduciary",
        # Famous founders & case studies
        "famous startup founders stories success",
        "venture capital history Kleiner Perkins Sequoia",
    ]

    n = wiki_api(queries_startup, bucket, "biz_vc_wiki_api"); total += n
    record_source("corpus_C_technical","Wikipedia Entrepreneurship/VC API",
                  "https://en.wikipedia.org", approx_mb([""]*n),
                  "VC mechanics, term sheets, cap tables, lean startup, ecosystems", n)
    log.info(f"  [BIZ-1] Wiki API: {n}")

    n = wiki_stream(STARTUP_KW, bucket, "biz_vc_wiki_stream", max_docs=4000); total += n
    record_source("corpus_C_technical","Wikipedia Startup/VC Stream",
                  "https://en.wikipedia.org", approx_mb([""]*n),
                  "Broad startup/VC/entrepreneurship keyword scan", n)
    log.info(f"  [BIZ-2] Wiki stream: {n}")

    n = se_qa(
        ["startups","money","economics","workplace"],
        bucket, "biz_vc_se",
        extra_kw={
            "startup","venture","investor","fundrais","equity","valuation",
            "term sheet","cap table","pitch","founder","accelerator",
            "mvp","product market fit","burn","runway","vc","angel",
        }, max_docs=4000
    ); total += n
    record_source("corpus_C_technical","Entrepreneurship StackExchange Q&A",
                  "https://startups.stackexchange.com", approx_mb([""]*n),
                  "Real Q&A: fundraising, term sheets, equity, growth, hiring", n)
    log.info(f"  [BIZ-3] SE Q&A: {n}")

    n = arxiv_cats(
        ["econ.GN","q-fin.GN","q-fin.EC","econ.TH","econ.EM"],
        bucket, "biz_vc_arxiv",
        extra_kw={
            "startup","entrepreneur","venture capital","innovation","firm",
            "investment","growth","market","equity","fund","risk",
        }, max_docs=3000
    ); total += n
    record_source("corpus_C_technical","arXiv Economics/Finance Papers",
                  "https://arxiv.org", approx_mb([""]*n),
                  "Academic entrepreneurship research: VC returns, innovation, growth", n)
    log.info(f"  [BIZ-4] arXiv: {n}")

    n = gutenberg_load([
        (5670,  "The Science of Getting Rich — Wallace Wattles"),
        (1541,  "The Art of War — Sun Tzu (strategy)"),
        (14474, "The Prince — Machiavelli (power & leadership)"),
        (120,   "Treasure Island — Stevenson (risk-taking narrative)"),
        (4280,  "The Wealth of Nations — Adam Smith"),
        (38194, "Political Economy — Mill"),
        (33851, "Principles of Economics — Marshall"),
        (7370,  "The Theory of the Leisure Class — Veblen"),
        (21352, "Principles of Scientific Management — Taylor"),
        (31270, "Pushing to the Front — Orison Swett Marden"),
    ], bucket, "biz_vc_gutenberg")
    record_source("corpus_C_technical","Gutenberg Business Classics",
                  "https://www.gutenberg.org", approx_mb([""]*n),
                  "Smith, Marshall, Mill, Veblen, Taylor — economics & business strategy", n)
    total += n
    log.info(f"  [BIZ-5] Gutenberg: {n}")

    # SEC EDGAR S-1 prospectuses (famous IPOs — rich business data)
    try:
        log.info("  [BIZ-6] SEC EDGAR S-1 prospectuses...")
        docs = []
        # Famous S-1 accession numbers (public filings)
        s1_filings = [
            ("0001559720-20-000010", "Airbnb S-1 IPO Prospectus"),
            ("0001543151-19-000020", "Uber S-1 IPO Prospectus"),
            ("0001759509-19-000006", "Lyft S-1 IPO Prospectus"),
            ("0001467760-19-000011", "Pinterest S-1 IPO Prospectus"),
            ("0001594686-14-000010", "Alibaba S-1 IPO Prospectus"),
            ("0001326801-12-000053", "Facebook S-1 IPO Prospectus"),
            ("0001564590-17-002505", "Snap S-1 IPO Prospectus"),
            ("0001326801-16-000045", "LinkedIn"),
        ]
        for accession, label in s1_filings:
            try:
                # Try EDGAR full-text search
                url = f"https://efts.sec.gov/LATEST/search-index?q=%22{requests.utils.quote(label.split()[0])}%22&dateRange=custom&startdt=2010-01-01&enddt=2023-12-31&forms=S-1"
                r = requests.get(url, timeout=15,
                                 headers={"User-Agent": "TitanAI research@example.com"})
                if r.status_code == 200:
                    hits = r.json().get("hits",{}).get("hits",[])
                    for hit in hits[:3]:
                        src = hit.get("_source",{})
                        text = src.get("file_date","") + " " +                                  src.get("display_names","") + "\n" +                                  src.get("file_description","")
                        if len(text) > 100:
                            docs.append(f"SEC Filing — {label}\n\n{text[:5000]}")
            except: pass
            time.sleep(0.5)

        # Also pull structured business descriptions from EDGAR company search
        companies = ["AIRBNB","UBER","LYFT","DOORDASH","COINBASE","ROBINHOOD",
                     "PALANTIR","SNOWFLAKE","DATABRICKS","STRIPE"]
        for company in companies:
            try:
                r = requests.get(
                    f"https://efts.sec.gov/LATEST/search-index?q=%22{company}%22&forms=S-1",
                    timeout=15, headers={"User-Agent": "TitanAI research@example.com"}
                )
                if r.status_code == 200:
                    for hit in r.json().get("hits",{}).get("hits",[])[:2]:
                        src = hit.get("_source",{})
                        snippet = (src.get("period_of_report","")+" "+
                                   src.get("file_description","")+" "+
                                   str(src.get("entity_name","")))
                        if len(snippet) > 80:
                            docs.append(f"SEC S-1 Filing — {company}\n\n{snippet}")
            except: pass
            time.sleep(0.3)

        n = write_docs(bucket, docs, "biz_sec_edgar")
        if n > 0:
            record_source("corpus_C_technical","SEC EDGAR S-1 Prospectuses",
                          "https://efts.sec.gov", approx_mb(docs),
                          "Real IPO S-1 filings: business models, financials, risks", n)
        total += n
        log.info(f"  [BIZ-6] SEC EDGAR: {n}")
    except Exception as e:
        log.error(f"  [BIZ-6] SEC EDGAR failed: {e}")

    n = openalex([
        "venture capital investment startup returns performance",
        "angel investor early stage startup funding",
        "startup failure survival rate entrepreneurship",
        "lean startup methodology product market fit",
        "entrepreneurial ecosystem regional innovation",
        "unicorn startup valuation growth trajectory",
        "startup accelerator incubator impact performance",
        "crowdfunding equity startup financing",
        "female founder gender gap venture capital",
        "emerging market entrepreneurship fintech Africa",
        "IPO underpricing long-run performance",
        "private equity buyout value creation",
        "innovation cluster Silicon Valley knowledge spillover",
        "founder CEO transition board governance startup",
        "startup pivot decision timing success",
    ], bucket, "biz_vc_openalex", "Entrepreneurship/VC Research")
    total += n
    log.info(f"  [BIZ-7] OpenAlex: {n}")

    log.info(f"  >>> Entrepreneurship & VC total: {total:,} docs")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 2 — RESEARCH & DEVELOPMENT
# ══════════════════════════════════════════════════════════════════════════════
def load_research_and_development():
    log.info("=" * 65)
    log.info("DOMAIN 2: Research & Development")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries_rd = [
        # R&D methodology & management
        "research methodology hypothesis experimental design",
        "scientific method reproducibility replication crisis",
        "technology readiness level NASA TRL scale",
        "stage-gate process new product development",
        "agile R&D open innovation Chesbrough",
        "disruptive innovation Christensen innovator dilemma",
        "sustaining innovation incremental radical",
        "R&D management corporate strategy portfolio",
        "skunkworks advanced research project management",
        "Bell Labs Xerox PARC industrial research history",
        "DARPA advanced research agency projects",
        # IP, patents & technology transfer
        "patent filing prosecution claims prior art",
        "technology transfer university spinoff license",
        "intellectual property management startup",
        "trade secret confidential information protection",
        "open source licensing commercialization",
        "patent trolls NPE litigation strategy",
        "freedom to operate FTO patent landscape",
        # Funding & grants
        "grant writing NSF NIH SBIR STTR funding",
        "research funding mechanisms peer review",
        "national laboratories government research",
        "EU Horizon research funding program",
        # Industrial R&D sectors
        "pharmaceutical R&D drug discovery pipeline",
        "semiconductor R&D Moore's law TSMC",
        "aerospace R&D advanced materials NASA",
        "energy R&D clean tech batteries solar",
        "biotech R&D CRISPR synthetic biology tools",
        # Research evaluation & metrics
        "bibliometrics h-index impact factor citation",
        "peer review process scientific publishing",
        "systematic review meta-analysis methodology",
        "research ethics IRB human subjects",
        "open access publishing preprint servers",
        # Innovation ecosystems
        "national innovation system Porter diamond",
        "innovation cluster geographic agglomeration",
        "R&D tax incentives policy economic impact",
        "standards development ISO IEC IEEE process",
        "translational research bench to bedside",
    ]

    n = wiki_api(queries_rd, bucket, "rd_wiki_api"); total += n
    record_source("corpus_C_technical","Wikipedia R&D API",
                  "https://en.wikipedia.org", approx_mb([""]*n),
                  "R&D methodology, innovation theory, patents, funding, ecosystems", n)
    log.info(f"  [RD-1] Wiki API: {n}")

    n = wiki_stream(RD_KW, bucket, "rd_wiki_stream", max_docs=4000); total += n
    record_source("corpus_C_technical","Wikipedia R&D Stream",
                  "https://en.wikipedia.org", approx_mb([""]*n),
                  "Broad R&D/innovation keyword scan", n)
    log.info(f"  [RD-2] Wiki stream: {n}")

    n = se_qa(
        ["academia","economics","workplace","money"],
        bucket, "rd_se",
        extra_kw={
            "research","patent","innovation","r&d","grant","funding",
            "publication","peer review","technology transfer","spinoff",
            "licensing","reproducibility","methodology","experiment",
        }, max_docs=3000
    ); total += n
    record_source("corpus_C_technical","R&D StackExchange Q&A",
                  "https://academia.stackexchange.com", approx_mb([""]*n),
                  "Academic & industry R&D Q&A: grants, patents, methodology", n)
    log.info(f"  [RD-3] SE Q&A: {n}")

    n = arxiv_cats(
        ["econ.GN","econ.TH","cs.CY","q-fin.GN"],
        bucket, "rd_arxiv",
        extra_kw={
            "innovation","r&d","research","technology","patent","knowledge",
            "spillover","productivity","growth","discovery","invention",
        }, max_docs=3000
    ); total += n
    record_source("corpus_C_technical","arXiv R&D/Innovation Papers",
                  "https://arxiv.org", approx_mb([""]*n),
                  "Economics of R&D: innovation, productivity, knowledge spillovers", n)
    log.info(f"  [RD-4] arXiv: {n}")

    n = gutenberg_load([
        (1173,  "The New Atlantis — Francis Bacon (scientific institution)"),
        (20433, "Novum Organum — Francis Bacon (scientific method)"),
        (5497,  "A Discourse on Method — Descartes"),
        (3637,  "The Advancement of Learning — Francis Bacon"),
        (36299, "Scientific American Vol 1 (early R&D journalism)"),
        (21352, "Principles of Scientific Management — Taylor"),
        (46,    "A Connecticut Yankee in King Arthur's Court — Twain (invention)"),
        (32037, "The Inventions, Researches and Writings of Tesla"),
        (24,    "The Mysterious Island — Verne (applied science)"),
        (4367,  "Engineering Descriptive Geometry"),
    ], bucket, "rd_gutenberg")
    record_source("corpus_C_technical","Gutenberg R&D/Science Classics",
                  "https://www.gutenberg.org", approx_mb([""]*n),
                  "Bacon, Descartes, Tesla, Novum Organum — scientific method & invention", n)
    total += n
    log.info(f"  [RD-5] Gutenberg: {n}")

    # NIST & OSTI R&D reports
    try:
        log.info("  [RD-6] DOE OSTI R&D technical reports...")
        docs = []
        rd_queries = [
            "technology transfer commercialization",
            "research development innovation policy",
            "laboratory directed research development",
            "advanced manufacturing process development",
            "energy technology development demonstration",
        ]
        for q in rd_queries:
            try:
                r = requests.get(
                    f"https://www.osti.gov/api/v1/records?q={requests.utils.quote(q)}"
                    f"&page=0&size=100",
                    timeout=20
                )
                if r.status_code == 200:
                    for rec in r.json().get("records",[]):
                        title = rec.get("title","").strip()
                        abst  = rec.get("description","").strip()
                        if abst and len(abst) > 100:
                            docs.append(
                                f"DOE R&D Technical Report\n\n"
                                f"Title: {title}\n\nAbstract:\n{abst}"
                            )
            except Exception as qe:
                log.warning(f"    OSTI query '{q[:30]}': {qe}")
            time.sleep(0.4)
        n = write_docs(bucket, docs, "rd_osti")
        if n > 0:
            record_source("corpus_C_technical","DOE OSTI R&D Reports",
                          "https://www.osti.gov", approx_mb(docs),
                          "US DOE technical R&D reports: lab research, tech transfer", n)
        total += n
        log.info(f"  [RD-6] OSTI: {n}")
    except Exception as e:
        log.error(f"  [RD-6] OSTI failed: {e}")

    n = openalex([
        "R&D investment productivity spillover firm level",
        "patent citation knowledge spillover technology",
        "university industry collaboration knowledge transfer",
        "open innovation external knowledge sourcing",
        "disruptive innovation market entry incumbent",
        "research productivity academic scientist output",
        "technology transfer spinoff licensing revenue",
        "DARPA program manager research breakthrough",
        "pharmaceutical R&D success rate clinical trial",
        "national innovation system economic growth",
        "bibliometrics research impact measurement",
        "reproducibility replication crisis science reform",
        "grant funding allocation peer review bias",
        "innovation cluster agglomeration knowledge",
        "startup corporate R&D collaboration",
    ], bucket, "rd_openalex", "R&D Research")
    total += n
    log.info(f"  [RD-7] OpenAlex: {n}")

    log.info(f"  >>> R&D total: {total:,} docs")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("TitanAI Business & R&D Corpus Loader — Starting")
    log.info("2 Domains: Entrepreneurship & VC | Research & Development")
    log.info(f"Target directory: {RAW}")
    log.info("")

    results = {}
    results["entrepreneurship_vc"] = load_entrepreneurship_vc()
    results["research_development"] = load_research_and_development()

    with open(INVENTORY,"w") as f: json.dump(inventory, f, indent=2)
    with open(EXCLUSION_LOG,"w") as f:
        for e in exclusions: f.write(json.dumps(e)+"\n")

    log.info("")
    log.info("=" * 65)
    log.info("BUSINESS & R&D CORPUS LOAD — COMPLETE")
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
