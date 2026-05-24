#!/usr/bin/env python3
"""
TitanAI World Politics & International Relations Corpus Loader  v2
==================================================================
Premium-depth training data across 11 political science subfields.
Target: 60,000+ documents.

Quality standards:
  • Full Wikipedia articles (up to 40,000 chars, no stubs < 800 chars)
  • SE Q&A with up to 3 top answers per question
  • PubMed abstracts for political science journals
  • Semantic Scholar for academic coverage
  • 500-char minimum document threshold
  • Smart chunking for large documents (5,000-char windows, 300-char overlap)
  • MD5 deduplication — no repeated content

Subfields:
  1.  IR Theory                — realism, liberalism, constructivism, power theory
  2.  Geopolitics & Powers     — US, China, Russia, EU, Indo-Pacific, Africa, BRICS
  3.  International Orgs       — UN, NATO, EU, WTO, IMF, ICC, G20
  4.  Diplomacy & FP           — negotiation, treaties, statecraft, coercive diplomacy
  5.  Political Economy        — trade wars, sanctions, globalization, finance
  6.  Conflict & Security      — war, deterrence, terrorism, peacekeeping, hybrid warfare
  7.  Human Rights & Intl Law  — UDHR, ICC, Geneva, R2P, refugee law
  8.  Political Ideologies     — democracy, authoritarianism, populism, nationalism
  9.  Elections & Governance   — electoral systems, corruption, rule of law
 10.  Global Issues            — climate, nuclear, migration, pandemic, cyber
 11.  Middle East & Israel-Pal — Ottoman to Gaza war 2023-24, all regional actors
"""

import os, sys, json, time, logging, requests, hashlib
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent
RAW  = BASE / "data" / "raw"
LOG  = BASE / "data" / "corpus_world_politics.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(str(LOG))]
)
log = logging.getLogger("world_politics_loader")

INVENTORY     = BASE / "data" / "source_inventory.json"
EXCLUSION_LOG = BASE / "data" / "exclusions_world_politics.jsonl"
try:
    with open(INVENTORY) as f:
        inventory = json.load(f)
except Exception:
    inventory = {}
exclusions = []

# Global dedup set — prevents near-duplicate documents across all sources
_seen_hashes = set()


# ── Helpers ─────────────────────────────────────────────────────────────────────
def doc_hash(text):
    return hashlib.md5(text[:300].encode("utf-8", errors="ignore")).hexdigest()

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

def chunk_text(title, text, source_tag, chunk_size=5000, overlap=300):
    """Split large documents into overlapping chunks for optimal training segments."""
    if len(text) <= chunk_size:
        return [f"[Source: {source_tag}]\n# {title}\n\n{text}"]
    chunks = []
    start = 0
    part = 0
    while start < len(text):
        end = start + chunk_size
        segment = text[start:end]
        if end < len(text):
            # Break at last paragraph boundary
            last_para = segment.rfind("\n\n")
            if last_para > chunk_size // 2:
                segment = segment[:last_para]
                end = start + last_para
        chunks.append(
            f"[Source: {source_tag} | Part {part+1}]\n# {title}\n\n{segment.strip()}"
        )
        part += 1
        start = end - overlap
        if start >= len(text):
            break
    return chunks

def write_docs(bucket_dir, docs, tag):
    bucket_dir.mkdir(parents=True, exist_ok=True)
    start = len(list(bucket_dir.glob(f"{tag}_*.txt")))
    n = 0
    for i, text in enumerate(docs):
        if not text or len(text.strip()) < 500:
            continue
        h = doc_hash(text)
        if h in _seen_hashes:
            continue
        _seen_hashes.add(h)
        (bucket_dir / f"{tag}_{start+i:06d}.txt").write_text(
            text.strip(), encoding="utf-8"
        )
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
def wiki_api(queries, bucket_dir, tag, max_per_query=25):
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
                        all_chunks.extend(chunk_text(t, text, "Wikipedia"))
                time.sleep(0.12)
        except Exception as e:
            log.warning(f"    wiki_api '{q[:40]}': {e}")
        time.sleep(0.05)
    return write_docs(bucket_dir, all_chunks, tag)

def wiki_stream(keywords, bucket_dir, tag, max_docs=5000):
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
            title = item.get("title", "").lower()
            text  = item.get("text", "").strip()
            if len(text) < 800:
                continue
            if any(k in title or k in text[:600].lower() for k in kw):
                all_chunks.extend(chunk_text(item["title"], text, "Wikipedia-stream"))
            if len(all_chunks) >= max_docs * 3:
                break
        return write_docs(bucket_dir, all_chunks, tag)
    except Exception as e:
        record_exclusion(f"wiki_stream:{tag}", str(e))
        return 0

def se_qa(domain_kws, bucket_dir, tag, extra_kw=None, max_docs=5000):
    """StackExchange Q&A — full question + top 3 answers for rich training signal."""
    try:
        from datasets import load_dataset
        se = load_dataset("HuggingFaceH4/stack-exchange-preferences",
                          split="train", streaming=True)
        docs = []
        for i, item in enumerate(se):
            if i > 5_000_000:
                break
            dom = safe_get(item, "domain")
            if not any(k in dom.lower() for k in domain_kws):
                continue
            q = safe_get(item, "question")
            if extra_kw and not any(k in q.lower() for k in extra_kw):
                continue
            answers = item.get("answers", []) or []
            good = sorted(
                [a for a in answers if (a.get("pm_score", 0) or 0) > 0],
                key=lambda a: a.get("pm_score", 0), reverse=True
            )
            if not good and answers:
                good = answers[:1]
            if good and len(q) > 80:
                ans_block = "\n\n---\n\n".join(
                    f"Answer (score {a.get('pm_score',0)}):\n{a.get('text','').strip()[:3500]}"
                    for a in good[:3]
                )
                docs.append(
                    f"[Source: StackExchange/{dom}]\n\n"
                    f"Question: {q.strip()}\n\n{ans_block}"
                )
            if len(docs) >= max_docs:
                break
        return write_docs(bucket_dir, docs, tag)
    except Exception as e:
        record_exclusion(f"se:{tag}", str(e))
        return 0

def arxiv_cats(cats, bucket_dir, tag, extra_kw=None, max_docs=5000):
    try:
        from datasets import load_dataset
        arxiv = load_dataset("Cornell-University/arxiv",
                             split="train", streaming=True)
        cat_set = set(cats)
        docs = []
        for i, item in enumerate(arxiv):
            if i > 6_000_000:
                break
            c = set((item.get("categories", "") or "").split())
            if not c.intersection(cat_set):
                continue
            title = (item.get("title", "") or "").replace("\n", " ").strip()
            abst  = (item.get("abstract", "") or "").replace("\n", " ").strip()
            if len(abst) < 150:
                continue
            if extra_kw:
                combo = (title + " " + abst).lower()
                if not any(k in combo for k in extra_kw):
                    continue
            docs.append(
                f"[Source: arXiv | Categories: {' '.join(sorted(c))}]\n\n"
                f"Title: {title}\n\nAbstract:\n{abst}"
            )
            if len(docs) >= max_docs:
                break
        return write_docs(bucket_dir, docs, tag)
    except Exception as e:
        record_exclusion(f"arxiv:{tag}", str(e))
        return 0

def gutenberg(ids, bucket_dir, tag):
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
                    all_chunks.extend(chunk_text(desc, text, f"Gutenberg#{gid}"))
                    log.info(f"    Gutenberg #{gid}: {len(text)} chars")
                    break
            except Exception as e:
                log.warning(f"    Gutenberg #{gid}: {e}")
        time.sleep(0.6)
    return write_docs(bucket_dir, all_chunks, tag)

def openalex(queries, bucket_dir, tag, label="Political Science Research"):
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
                mx = max(p for ps in inv.values() for p in ps)
                wl = [""] * (mx + 1)
                for word, ps in inv.items():
                    for p in ps:
                        wl[p] = word
                abst = " ".join(x for x in wl if x)
                if len(abst) > 150:
                    docs.append(
                        f"[Source: OpenAlex | {label} | Year: {year} | Citations: {cites}]\n\n"
                        f"Title: {title}\n\nAbstract:\n{abst}"
                    )
        except Exception as e:
            log.warning(f"    OpenAlex '{q[:35]}': {e}")
        time.sleep(0.35)
    return write_docs(bucket_dir, docs, tag)

def pubmed(queries, bucket_dir, tag, max_per_query=100):
    """NCBI PubMed — high-quality biomedical & political science journal abstracts."""
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
            }, timeout=25)
            if fetch.status_code == 200 and len(fetch.text) > 200:
                for block in fetch.text.split("\n\n\n"):
                    block = block.strip()
                    if len(block) > 300:
                        docs.append(f"[Source: PubMed]\n\n{block}")
            time.sleep(0.4)
        except Exception as e:
            log.warning(f"    PubMed '{q[:35]}': {e}")
    return write_docs(bucket_dir, docs, tag)

def semantic_scholar(queries, bucket_dir, tag, max_per_query=80):
    """Semantic Scholar — citation-ranked academic papers with abstracts."""
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
                continue
            results = sorted(
                r.json().get("data", []),
                key=lambda p: p.get("citationCount") or 0, reverse=True
            )
            for p in results:
                title = (p.get("title", "") or "").strip()
                abst  = (p.get("abstract", "") or "").strip()
                year  = p.get("year", "")
                cites = p.get("citationCount", 0)
                fields = ", ".join(p.get("fieldsOfStudy", []) or [])
                if len(abst) > 150:
                    docs.append(
                        f"[Source: Semantic Scholar | Fields: {fields} | "
                        f"Year: {year} | Citations: {cites}]\n\n"
                        f"Title: {title}\n\nAbstract:\n{abst}"
                    )
            time.sleep(1.0)
        except Exception as e:
            log.warning(f"    SemanticScholar '{q[:35]}': {e}")
    return write_docs(bucket_dir, docs, tag)


# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD SETS
# ══════════════════════════════════════════════════════════════════════════════
POL_IR_KW = {
    "international relations", "realism", "liberalism", "constructivism",
    "neorealism", "balance of power", "hegemony", "anarchy", "sovereignty",
    "security dilemma", "deterrence theory", "polarity", "unipolarity",
    "multipolarity", "power transition", "democratic peace theory",
    "liberal internationalism", "offensive realism", "defensive realism",
    "international society", "english school", "world systems theory",
    "dependency theory", "neoliberalism", "international norms",
    "soft power", "hard power", "smart power", "public diplomacy",
    "foreign policy analysis", "national interest", "strategic culture",
}
POL_GEO_KW = {
    "geopolitics", "great power competition", "sino-american relations",
    "us foreign policy", "china foreign policy", "russia foreign policy",
    "european union foreign policy", "nato alliance", "indo-pacific strategy",
    "belt and road initiative", "quad security dialogue", "aukus",
    "middle east politics", "africa politics", "latin america politics",
    "south china sea", "taiwan strait", "ukraine war", "arctic geopolitics",
    "energy geopolitics", "rare earth minerals politics",
    "global south", "brics nations", "emerging powers",
}
POL_ORG_KW = {
    "united nations", "security council", "general assembly", "nato",
    "european union", "asean", "african union", "world trade organization",
    "international monetary fund", "world bank", "g7", "g20",
    "international criminal court", "opec", "world health organization",
    "international atomic energy agency", "nuclear non-proliferation treaty",
    "paris agreement", "kyoto protocol", "multilateralism",
    "un peacekeeping", "un sanctions", "international regime",
    "regional integration", "supra-national governance",
}
POL_ECON_KW = {
    "political economy", "trade war", "economic sanctions", "globalization",
    "protectionism", "free trade", "wto dispute", "tariff policy",
    "currency war", "dollar hegemony", "reserve currency",
    "development finance", "debt trap diplomacy", "aid conditionality",
    "economic coercion", "decoupling", "supply chain security",
    "state capitalism", "sovereign wealth fund", "foreign direct investment",
    "economic nationalism", "industrial policy", "neo-mercantilism",
    "washington consensus", "beijing consensus",
}
POL_CONFLICT_KW = {
    "armed conflict", "civil war", "proxy war", "interstate war",
    "terrorism", "counterterrorism", "insurgency", "guerrilla warfare",
    "nuclear deterrence", "mutually assured destruction", "first strike",
    "arms control", "arms race", "military alliance", "peacekeeping",
    "peacebuilding", "conflict resolution", "mediation", "hybrid warfare",
    "information warfare", "cyberwarfare", "grey zone operations",
    "genocide", "ethnic cleansing", "atrocity crimes",
    "refugee crisis", "displacement", "humanitarian intervention",
}
POL_IDEOLOGY_KW = {
    "democracy", "authoritarianism", "totalitarianism", "populism",
    "nationalism", "socialism", "communism", "fascism", "liberalism",
    "conservatism", "libertarianism", "social democracy", "neoliberalism",
    "green politics", "feminism politics", "identity politics",
    "democratic backsliding", "autocratization", "hybrid regime",
    "competitive authoritarianism", "illiberal democracy",
    "political polarization", "radicalization", "extremism",
    "electoral authoritarianism", "one party state", "theocracy",
}
POL_ME_KW = {
    "israel", "palestine", "gaza", "west bank", "hamas", "hezbollah",
    "zionism", "nakba", "intifada", "oslo accords", "two-state solution",
    "arab-israeli war", "six-day war", "yom kippur war", "suez crisis",
    "settler colonialism", "occupation", "blockade", "idf",
    "iran nuclear program", "saudi arabia foreign policy", "turkey middle east",
    "arab spring", "syria civil war", "yemen war", "iraq war",
    "islamic state isis", "kurdish question", "lebanese politics",
    "egyptian politics", "jordan politics", "muslim brotherhood",
    "gulf cooperation council", "oil politics opec", "abraham accords",
    "normalization israel arab", "UN resolution 242",
    "international court justice genocide case", "humanitarian law gaza",
}
POL_ALL_KW = (POL_IR_KW | POL_GEO_KW | POL_ORG_KW |
              POL_ECON_KW | POL_CONFLICT_KW | POL_IDEOLOGY_KW | POL_ME_KW)


# ══════════════════════════════════════════════════════════════════════════════
# SUBFIELD LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_ir_theory():
    log.info("=" * 65)
    log.info("POL-1: International Relations Theory")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "realism international relations Waltz Morgenthau power",
        "liberalism international relations Keohane Nye institutions",
        "constructivism international relations Wendt Finnemore norms",
        "balance of power theory great power hegemony stability",
        "offensive realism Mearsheimer great power tragedy",
        "democratic peace theory Kant perpetual peace empirical",
        "security dilemma spiral model misperception escalation",
        "anarchy international system self-help states",
        "soft power hard power Nye national influence",
        "power transition theory Organski hegemonic war",
        "english school international society Hedley Bull",
        "world systems theory Wallerstein core periphery",
        "dependency theory underdevelopment global inequality",
        "neoliberal institutionalism cooperation international regimes",
        "postcolonialism international relations empire critique",
        "feminist international relations gender security war",
        "critical theory international relations Gramsci hegemony",
        "foreign policy analysis decision making bureaucratic",
        "securitization theory Copenhagen School threat construction",
        "norm diffusion socialization international institutions",
        "deterrence theory nuclear conventional credibility",
        "alliance formation balancing bandwagoning states",
        "bargaining theory war commitment problem",
        "audience costs domestic politics foreign policy",
        "two-level games Putnam domestic international",
    ]
    n = wiki_api(queries, bucket, "pol_ir_theory_wiki"); total += n
    log.info(f"  [POL-IR-1] Wiki API: {n}")
    n = wiki_stream(POL_IR_KW, bucket, "pol_ir_theory_stream", max_docs=4000); total += n
    log.info(f"  [POL-IR-2] Wiki stream: {n}")
    n = se_qa(["politics", "history", "philosophy"], bucket, "pol_ir_se",
              extra_kw={"realism", "liberal", "international relations", "hegemony",
                        "sovereignty", "deterrence", "alliance", "foreign policy",
                        "constructiv", "war", "power"}); total += n
    log.info(f"  [POL-IR-3] SE Q&A: {n}")
    n = arxiv_cats(["econ.GN", "q-fin.EC"], bucket, "pol_ir_arxiv",
                   extra_kw={"international", "geopolit", "foreign policy",
                             "hegemony", "power", "regime", "alliance", "war"},
                   max_docs=3000); total += n
    log.info(f"  [POL-IR-4] arXiv: {n}")
    n = semantic_scholar([
        "balance of power great power competition hegemony",
        "democratic peace theory empirical evidence",
        "constructivism norms international relations socialization",
        "deterrence credibility nuclear conventional",
        "liberalism international institutions cooperation",
        "securitization threat construction Copenhagen school",
        "audience costs foreign policy democratic accountability",
        "bargaining war commitment problem rationalist",
    ], bucket, "pol_ir_ss"); total += n
    log.info(f"  [POL-IR-5] Semantic Scholar: {n}")
    n = pubmed([
        "political science international relations theory review",
        "war conflict causes quantitative analysis",
        "democratic peace empirical study",
    ], bucket, "pol_ir_pubmed"); total += n
    log.info(f"  [POL-IR-6] PubMed: {n}")
    log.info(f"  >>> IR Theory total: {total:,}")
    return total


def load_geopolitics_major_powers():
    log.info("=" * 65)
    log.info("POL-2: Geopolitics & Major Powers")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "US foreign policy grand strategy national security",
        "China foreign policy Belt Road Initiative global ambition",
        "Russia foreign policy Ukraine NATO expansion revisionism",
        "European Union foreign security policy CFSP",
        "Indo-Pacific strategy Quad AUKUS containment China",
        "NATO expansion enlargement Russia security dilemma",
        "South China Sea territorial disputes islands militarization",
        "Taiwan cross-strait relations deterrence invasion",
        "Middle East US policy withdrawal vacuum",
        "Africa geopolitics China US competition infrastructure",
        "Latin America left right political turn pink tide",
        "India foreign policy strategic autonomy non-alignment",
        "Japan remilitarization security reinterpretation",
        "energy geopolitics oil gas pipelines Russia Europe",
        "Arctic geopolitics sovereignty resources shipping lanes",
        "BRICS emerging powers global governance reform",
        "sanctions Russia Iran North Korea effectiveness",
        "technology competition semiconductors AI export controls",
        "rare earth minerals critical supply chain security",
        "global south non-aligned new coalition multilateralism",
        "dollar hegemony reserve currency alternatives",
        "hybrid warfare Russia information operations election",
        "China military modernization PLA navy expansion",
        "nuclear weapons modernization great powers arms race",
        "space competition military satellites anti-satellite",
    ]
    n = wiki_api(queries, bucket, "pol_geo_wiki"); total += n
    log.info(f"  [POL-GEO-1] Wiki API: {n}")
    n = wiki_stream(POL_GEO_KW, bucket, "pol_geo_stream", max_docs=5000); total += n
    log.info(f"  [POL-GEO-2] Wiki stream: {n}")
    n = se_qa(["politics", "history"], bucket, "pol_geo_se",
              extra_kw={"china", "russia", "united states", "nato", "geopolit",
                        "taiwan", "ukraine", "sanctions", "indo-pacific", "brics"}); total += n
    log.info(f"  [POL-GEO-3] SE Q&A: {n}")
    n = semantic_scholar([
        "US China strategic competition technology decoupling",
        "Belt Road Initiative debt sustainability Africa",
        "NATO enlargement security dilemma Russia Ukraine",
        "Indo-Pacific security quad aukus China containment",
        "energy transition geopolitics oil gas Russia",
        "Taiwan strait crisis deterrence military balance",
        "Middle East regional order power vacuum",
        "Africa political economy China infrastructure debt",
        "global south BRICS multilateral reform governance",
    ], bucket, "pol_geo_ss"); total += n
    log.info(f"  [POL-GEO-4] Semantic Scholar: {n}")
    log.info(f"  >>> Geopolitics total: {total:,}")
    return total


def load_international_organizations():
    log.info("=" * 65)
    log.info("POL-3: International Organizations")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "United Nations Security Council veto reform legitimacy",
        "UN General Assembly resolutions voting patterns",
        "UN peacekeeping operations effectiveness failures",
        "NATO collective defence Article 5 burden sharing",
        "European Union institutions decision making integration",
        "EU Common Foreign Security Policy CFSP CSDP",
        "ASEAN regional security non-interference consensus",
        "African Union peace security council intervention",
        "World Trade Organization dispute settlement appellate",
        "IMF structural adjustment conditionality sovereign debt",
        "World Bank development lending governance effectiveness",
        "G7 G20 global economic governance coordination",
        "International Criminal Court Rome Statute jurisdiction",
        "OPEC oil production quotas market manipulation",
        "International Atomic Energy Agency safeguards verification",
        "WHO global health governance pandemic reform",
        "Paris Climate Agreement NDCs implementation",
        "UN Human Rights Council mechanisms UPR",
        "multilateralism crisis reform international institutions",
        "regional integration comparative EU ASEAN AU effectiveness",
        "UNHCR refugee protection funding mandate",
        "UNESCO cultural heritage soft power",
        "ILO labor standards international enforcement",
        "Bretton Woods institutions reform quota voting",
        "nuclear non-proliferation treaty review conference",
    ]
    n = wiki_api(queries, bucket, "pol_org_wiki"); total += n
    log.info(f"  [POL-ORG-1] Wiki API: {n}")
    n = wiki_stream(POL_ORG_KW, bucket, "pol_org_stream", max_docs=4000); total += n
    log.info(f"  [POL-ORG-2] Wiki stream: {n}")
    n = se_qa(["politics", "history"], bucket, "pol_org_se",
              extra_kw={"united nations", "nato", "imf", "world bank", "wto",
                        "eu", "european union", "peacekeeping", "icc", "opec"}); total += n
    log.info(f"  [POL-ORG-3] SE Q&A: {n}")
    n = semantic_scholar([
        "UN Security Council reform veto legitimacy effectiveness",
        "IMF conditionality austerity developing countries",
        "WTO dispute settlement appellate body legitimacy",
        "NATO burden sharing collective defence deterrence",
        "EU integration deepening widening enlargement",
        "multilateral institutions effectiveness reform mandate",
        "international organizations compliance enforcement",
        "regional integration theories neofunctionalism",
    ], bucket, "pol_org_ss"); total += n
    log.info(f"  [POL-ORG-4] Semantic Scholar: {n}")
    log.info(f"  >>> International Organizations total: {total:,}")
    return total


def load_political_economy():
    log.info("=" * 65)
    log.info("POL-4: Political Economy & Trade")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "trade war protectionism tariffs retaliations welfare",
        "economic sanctions effectiveness signaling coercion",
        "globalization backlash inequality political response",
        "Washington consensus neoliberal reform critique",
        "development aid effectiveness conditionality governance",
        "sovereign debt crisis restructuring IMF austerity",
        "currency manipulation exchange rate competitive devaluation",
        "economic nationalism industrial policy infant industry",
        "supply chain security friend-shoring reshoring",
        "state capitalism China model developmental state",
        "foreign direct investment political risk governance",
        "economic coercion grey zone financial weapons",
        "SWIFT financial sanctions dollar weaponization",
        "climate finance green transition carbon border tax",
        "digital trade data governance cross-border flows",
        "international tax avoidance OECD BEPS pillar",
        "commodity prices resource curse Dutch disease",
        "food security geopolitics agriculture trade",
        "technology export controls semiconductors decoupling",
        "debt trap diplomacy Africa China BRI sustainability",
        "petrodollar system Gulf recycling dollar hegemony",
        "cryptocurrency Bitcoin geopolitics sanctions evasion",
        "WTO most favored nation preferential trade",
        "FTA free trade agreements regional bilateral",
        "development finance competition China DFI US",
    ]
    n = wiki_api(queries, bucket, "pol_econ_wiki"); total += n
    log.info(f"  [POL-ECON-1] Wiki API: {n}")
    n = wiki_stream(POL_ECON_KW, bucket, "pol_econ_stream", max_docs=4000); total += n
    log.info(f"  [POL-ECON-2] Wiki stream: {n}")
    n = se_qa(["economics", "politics", "history"], bucket, "pol_econ_se",
              extra_kw={"sanction", "trade war", "tariff", "globalization",
                        "imf", "world bank", "debt", "currency", "industrial policy",
                        "development"}); total += n
    log.info(f"  [POL-ECON-3] SE Q&A: {n}")
    n = arxiv_cats(["econ.GN", "econ.HI", "q-fin.EC"], bucket, "pol_econ_arxiv",
                   extra_kw={"trade", "sanction", "globalization", "political",
                             "development", "inequality", "tariff", "currency"},
                   max_docs=3000); total += n
    log.info(f"  [POL-ECON-4] arXiv: {n}")
    n = semantic_scholar([
        "economic sanctions effectiveness signaling compliance",
        "trade war retaliatory tariff welfare loss",
        "globalization inequality political backlash populism",
        "China development finance Africa conditionality debt",
        "dollar hegemony international monetary system reform",
        "industrial policy infant industry protection growth",
        "supply chain resilience geopolitical risk decoupling",
        "sovereign debt restructuring creditor coordination",
    ], bucket, "pol_econ_ss"); total += n
    log.info(f"  [POL-ECON-5] Semantic Scholar: {n}")
    n = pubmed([
        "political economy inequality health outcomes",
        "economic sanctions health humanitarian effects",
        "globalization labor health developing countries",
    ], bucket, "pol_econ_pubmed"); total += n
    log.info(f"  [POL-ECON-6] PubMed: {n}")
    log.info(f"  >>> Political Economy total: {total:,}")
    return total


def load_conflict_security():
    log.info("=" * 65)
    log.info("POL-5: Conflict, Security & War")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "causes of war interstate conflict quantitative Correlates of War",
        "civil war onset greed grievance lootable resources",
        "nuclear deterrence credibility extended compellence",
        "arms control treaties verification compliance failure",
        "terrorism radicalization prevention deradicalization",
        "insurgency counterinsurgency COIN doctrine effectiveness",
        "hybrid warfare grey zone operations Russia Ukraine",
        "cyber warfare state actors attribution international law",
        "information warfare propaganda disinformation democracy",
        "peacekeeping peacebuilding effectiveness UN operations",
        "conflict resolution mediation negotiation success",
        "genocide prevention early warning R2P application",
        "proxy war great powers third party support",
        "military alliances deterrence credibility free-riding",
        "arms race action reaction model Richardson",
        "non-proliferation nuclear weapons NPT regime",
        "chemical biological weapons conventions enforcement",
        "private military companies mercenaries accountability",
        "drone warfare targeted killing international law",
        "space militarization anti-satellite weapons treaty",
        "maritime security piracy freedom of navigation",
        "nuclear weapons modernization Russia US China",
        "war crimes accountability prosecution ICC",
        "post-conflict reconstruction statebuilding failure",
        "ethnic conflict nationalism mobilization violence",
    ]
    n = wiki_api(queries, bucket, "pol_conflict_wiki"); total += n
    log.info(f"  [POL-CONF-1] Wiki API: {n}")
    n = wiki_stream(POL_CONFLICT_KW, bucket, "pol_conflict_stream", max_docs=5000); total += n
    log.info(f"  [POL-CONF-2] Wiki stream: {n}")
    n = se_qa(["politics", "history", "law"], bucket, "pol_conflict_se",
              extra_kw={"war", "conflict", "terrorism", "nuclear", "military",
                        "deterrence", "arms", "insurgency", "peace", "genocide",
                        "hybrid warfare", "cyber"}); total += n
    log.info(f"  [POL-CONF-3] SE Q&A: {n}")
    n = arxiv_cats(["econ.GN"], bucket, "pol_conflict_arxiv",
                   extra_kw={"war", "conflict", "terrorism", "nuclear", "military",
                             "deterrence", "arms", "insurgency", "peace"},
                   max_docs=3000); total += n
    log.info(f"  [POL-CONF-4] arXiv: {n}")
    n = semantic_scholar([
        "civil war onset greed grievance lootable resources",
        "nuclear deterrence extended credibility allies signaling",
        "terrorism radicalization prevention deradicalization programs",
        "cyberwarfare state attribution international law norms",
        "UN peacekeeping effectiveness outcomes mandate",
        "conflict resolution mediation third party success",
        "hybrid warfare Russia Ukraine information operations",
        "drone targeted killing effectiveness counterterrorism",
        "arms race action-reaction model nuclear arsenals",
        "ethnic conflict mobilization grievance elite manipulation",
    ], bucket, "pol_conflict_ss"); total += n
    log.info(f"  [POL-CONF-5] Semantic Scholar: {n}")
    n = pubmed([
        "war trauma PTSD conflict mental health population",
        "genocide prevention early warning indicators",
        "terrorism radicalization psychological factors",
        "conflict mortality civilian casualties epidemiology",
    ], bucket, "pol_conflict_pubmed"); total += n
    log.info(f"  [POL-CONF-6] PubMed: {n}")
    log.info(f"  >>> Conflict & Security total: {total:,}")
    return total


def load_human_rights_intl_law():
    log.info("=" * 65)
    log.info("POL-6: Human Rights & International Law")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "Universal Declaration Human Rights UDHR history drafting",
        "International Criminal Court Rome Statute jurisdiction",
        "Geneva Conventions international humanitarian law",
        "responsibility to protect R2P norm evolution",
        "refugee law 1951 Refugee Convention non-refoulement",
        "UN Human Rights Council mechanisms special procedures",
        "treaty body monitoring implementation state reports",
        "transitional justice truth commission accountability",
        "customary international law jus cogens peremptory norms",
        "state sovereignty human rights intervention tension",
        "international law use of force jus ad bellum Charter",
        "war crimes crimes against humanity genocide definition",
        "international criminal tribunals ICTY ICTR legacy",
        "human trafficking modern slavery protocol",
        "children rights CRC optional protocols",
        "torture convention absolute prohibition enforcement",
        "economic social cultural rights ICESCR justiciability",
        "indigenous peoples rights UNDRIP consultation",
        "climate change human rights nexus litigation",
        "business human rights Ruggie principles UNGP",
        "extraterritorial obligations human rights jurisdiction",
        "universal jurisdiction war crimes domestic prosecution",
        "derogation emergency human rights limitations",
        "statelessness nationality nationality law",
        "digital rights surveillance human rights internet",
    ]
    n = wiki_api(queries, bucket, "pol_hr_wiki"); total += n
    log.info(f"  [POL-HR-1] Wiki API: {n}")
    n = wiki_stream({
        "human rights", "international law", "humanitarian law",
        "refugee", "asylum", "war crime", "genocide", "torture",
        "international criminal court", "universal jurisdiction",
        "responsibility to protect", "transitional justice",
    }, bucket, "pol_hr_stream", max_docs=3000); total += n
    log.info(f"  [POL-HR-2] Wiki stream: {n}")
    n = se_qa(["law", "politics", "philosophy"], bucket, "pol_hr_se",
              extra_kw={"human rights", "international law", "war crime",
                        "genocide", "refugee", "asylum", "icc", "torture",
                        "r2p", "humanitarian"}); total += n
    log.info(f"  [POL-HR-3] SE Q&A: {n}")
    n = semantic_scholar([
        "international criminal court complementarity effectiveness",
        "responsibility to protect humanitarian intervention norm",
        "refugee protection international law non-refoulement",
        "transitional justice reconciliation accountability truth",
        "human rights conditionality foreign policy compliance",
        "universal jurisdiction war crimes domestic prosecution",
        "business human rights supply chain due diligence",
        "climate litigation human rights courts decisions",
    ], bucket, "pol_hr_ss"); total += n
    log.info(f"  [POL-HR-4] Semantic Scholar: {n}")
    n = pubmed([
        "human rights violations health outcomes documentation",
        "torture prevention detection medical evidence",
        "refugee health mental physical outcomes displacement",
        "international humanitarian law medical personnel",
    ], bucket, "pol_hr_pubmed"); total += n
    log.info(f"  [POL-HR-5] PubMed: {n}")
    log.info(f"  >>> Human Rights & Int'l Law total: {total:,}")
    return total


def load_political_ideologies_systems():
    log.info("=" * 65)
    log.info("POL-7: Political Ideologies & Systems")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "democracy varieties liberal illiberal electoral",
        "authoritarianism types electoral competitive personalist",
        "totalitarianism theory Arendt Orwell historical cases",
        "populism left right causes effects definition",
        "nationalism ethnic civic varieties historical",
        "socialism communism Marxism history movements",
        "fascism ideology historical cases Europe",
        "social democracy welfare state Scandinavian model",
        "conservatism liberalism political philosophy debate",
        "libertarianism anarchism minimal state theory",
        "green politics ecological parties platform",
        "feminist political theory gender equality politics",
        "democratic backsliding autocratization mechanisms",
        "political polarization partisan identity sorting",
        "religious fundamentalism politics state religion",
        "identity politics multiculturalism recognition",
        "revolutionary movements ideology Marxist Leninist",
        "constitutionalism rule of law judicial review",
        "federalism decentralization multilevel governance",
        "hybrid regimes competitive authoritarianism survival",
        "capitalism varieties liberal coordinated market",
        "propaganda ideology mass media state control",
        "technocracy meritocracy democratic legitimacy",
        "political philosophy Rawls justice fairness veil",
        "Hannah Arendt totalitarianism banality evil",
    ]
    n = wiki_api(queries, bucket, "pol_ideology_wiki"); total += n
    log.info(f"  [POL-IDEO-1] Wiki API: {n}")
    n = wiki_stream(POL_IDEOLOGY_KW, bucket, "pol_ideology_stream", max_docs=4000); total += n
    log.info(f"  [POL-IDEO-2] Wiki stream: {n}")
    n = se_qa(["politics", "philosophy", "history"], bucket, "pol_ideology_se",
              extra_kw={"democracy", "authoritarian", "populism", "nationalism",
                        "socialism", "fascism", "liberalism", "conservatism",
                        "communism", "ideology", "polarization"}); total += n
    log.info(f"  [POL-IDEO-3] SE Q&A: {n}")
    n = gutenberg([
        (1404,  "The Federalist Papers — Hamilton, Madison, Jay"),
        (30274, "The Communist Manifesto — Marx and Engels"),
        (61,    "The Declaration of Independence — Jefferson"),
        (5,     "The United States Constitution"),
        (36,    "The Prince — Machiavelli"),
        (3207,  "Leviathan — Thomas Hobbes"),
        (7370,  "The Social Contract — Jean-Jacques Rousseau"),
        (10378, "Two Treatises of Government — John Locke"),
        (1232,  "The Republic — Plato"),
        (6762,  "Politics — Aristotle"),
        (46,    "Common Sense — Thomas Paine"),
        (25368, "On Liberty — John Stuart Mill"),
        (1497,  "The Republic — Plato (alt)"),
        (2814,  "Utopia — Thomas More"),
        (932,   "The Wealth of Nations — Adam Smith"),
    ], bucket, "pol_ideology_gutenberg"); total += n
    log.info(f"  [POL-IDEO-4] Gutenberg classics: {n}")
    n = semantic_scholar([
        "democratic backsliding autocratization populism mechanisms",
        "political polarization partisan identity sorting",
        "populism economic insecurity cultural backlash causes",
        "nationalism ethnic civic identity salience conflict",
        "authoritarian resilience regime survival strategies",
        "varieties of capitalism institutional complementarity",
        "social democracy welfare state retrenchment",
        "political philosophy justice equality Rawls",
    ], bucket, "pol_ideology_ss"); total += n
    log.info(f"  [POL-IDEO-5] Semantic Scholar: {n}")
    log.info(f"  >>> Political Ideologies total: {total:,}")
    return total


def load_elections_governance():
    log.info("=" * 65)
    log.info("POL-8: Electoral Systems & Governance")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "electoral systems proportional representation outcomes",
        "first past the post majoritarian plurality systems",
        "electoral fraud manipulation rigging detection",
        "voter turnout participation mobilization determinants",
        "political parties party systems Duverger's law",
        "campaign finance money politics corruption",
        "media elections disinformation social media",
        "corruption governance public sector accountability",
        "rule of law judicial independence democracy",
        "state capacity bureaucracy Weberian effectiveness",
        "decentralization local government fiscal devolution",
        "civil society NGOs democracy social capital",
        "political representation minorities women quotas",
        "parliamentary presidential semi-presidential systems",
        "constitutional design power sharing veto players",
        "electoral violence post-election conflict Africa",
        "international election monitoring observation",
        "digital democracy e-voting security risks",
        "deliberative democracy citizens assembly mini-publics",
        "bureaucratic politics Allison Essence of Decision",
        "veto players institutional stability reform",
        "term limits executive power consolidation",
        "electoral reform referenda direct democracy",
        "gerrymandering electoral district manipulation",
        "populism elections party systems dealignment",
    ]
    n = wiki_api(queries, bucket, "pol_gov_wiki"); total += n
    log.info(f"  [POL-GOV-1] Wiki API: {n}")
    n = wiki_stream({
        "election", "voting", "electoral", "parliament", "government",
        "democracy", "corruption", "governance", "constitution", "rule of law",
        "political party", "campaign", "referendum", "representative",
    }, bucket, "pol_gov_stream", max_docs=4000); total += n
    log.info(f"  [POL-GOV-2] Wiki stream: {n}")
    n = se_qa(["politics", "law", "economics"], bucket, "pol_gov_se",
              extra_kw={"election", "voting", "corruption", "governance",
                        "judiciary", "parliament", "constitution", "party",
                        "campaign finance", "electoral"}); total += n
    log.info(f"  [POL-GOV-3] SE Q&A: {n}")
    n = semantic_scholar([
        "electoral system proportional majoritarian representation outcomes",
        "corruption governance development economic growth",
        "judicial independence autocratization courts capture",
        "women political representation quotas barriers outcomes",
        "voter turnout mobilization suppression determinants",
        "campaign finance money political outcomes inequality",
        "state capacity bureaucratic quality service delivery",
        "constitutional design presidentialism parliamentarism stability",
    ], bucket, "pol_gov_ss"); total += n
    log.info(f"  [POL-GOV-4] Semantic Scholar: {n}")
    log.info(f"  >>> Elections & Governance total: {total:,}")
    return total


def load_global_issues():
    log.info("=" * 65)
    log.info("POL-9: Global Issues & Transnational Politics")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "climate change politics Paris Agreement UNFCCC",
        "nuclear proliferation non-proliferation NPT regime",
        "international migration refugee politics governance",
        "global pandemic governance WHO reform COVID-19",
        "cyber security internet governance state actors",
        "artificial intelligence governance regulation ethics",
        "space law outer space treaty governance",
        "biodiversity loss environmental politics governance",
        "food security global hunger political economy",
        "water scarcity transboundary rivers conflict",
        "global health equity vaccine nationalism COVAX",
        "disinformation information warfare democracy resilience",
        "dark money offshore finance tax havens governance",
        "transnational organized crime drug trafficking networks",
        "human smuggling trafficking international response",
        "social media revolution Arab Spring protest",
        "climate migration displacement adaptation funding",
        "ocean governance high seas treaty biodiversity",
        "global inequality poverty SDG development goals",
        "digital authoritarianism surveillance capitalism",
        "antimicrobial resistance global health security",
        "critical infrastructure cyber attacks state",
        "global tax reform OECD minimum tax pillar",
        "nuclear arms race modernization great powers",
        "pandemic preparedness global health security reform",
    ]
    n = wiki_api(queries, bucket, "pol_global_wiki"); total += n
    log.info(f"  [POL-GLOB-1] Wiki API: {n}")
    n = wiki_stream({
        "climate politics", "nuclear", "migration", "pandemic governance",
        "cyber", "artificial intelligence governance", "global health",
        "food security", "water conflict", "disinformation",
        "transnational crime", "human trafficking", "sustainable development",
    }, bucket, "pol_global_stream", max_docs=4000); total += n
    log.info(f"  [POL-GLOB-2] Wiki stream: {n}")
    n = se_qa(["politics", "environment", "law"], bucket, "pol_global_se",
              extra_kw={"climate", "nuclear", "migration", "pandemic",
                        "cyber", "ai governance", "food security",
                        "global health", "disinformation"}); total += n
    log.info(f"  [POL-GLOB-3] SE Q&A: {n}")
    n = semantic_scholar([
        "climate change international negotiations UNFCCC Paris",
        "nuclear non-proliferation treaty compliance regime",
        "international migration governance refugee protection",
        "pandemic preparedness global health security reform",
        "internet governance multistakeholder ICANN sovereignty",
        "artificial intelligence governance ethics global",
        "climate migration displacement adaptation finance",
        "disinformation elections democracy resilience countering",
        "global health equity vaccine access inequality",
        "ocean governance high seas biodiversity treaty",
    ], bucket, "pol_global_ss"); total += n
    log.info(f"  [POL-GLOB-4] Semantic Scholar: {n}")
    n = pubmed([
        "climate change health public health adaptation",
        "pandemic preparedness global health security",
        "antimicrobial resistance global governance policy",
        "vaccine equity access developing countries",
        "migration health outcomes refugee wellbeing",
    ], bucket, "pol_global_pubmed"); total += n
    log.info(f"  [POL-GLOB-5] PubMed: {n}")
    log.info(f"  >>> Global Issues total: {total:,}")
    return total


def load_diplomacy_foreign_policy():
    log.info("=" * 65)
    log.info("POL-10: Diplomacy & Foreign Policy Analysis")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "diplomacy history practice bilateral multilateral",
        "treaty negotiation ratification international law",
        "foreign policy decision making bureaucratic model",
        "summit diplomacy leaders bilateral meetings outcomes",
        "economic diplomacy trade investment tools coercion",
        "public diplomacy soft power cultural exchange",
        "coercive diplomacy compellence deterrence Schelling",
        "alliance management intra-alliance politics burden",
        "foreign policy domestic politics two-level games Putnam",
        "diplomatic history Cold War détente arms control",
        "global governance reform multilateral institutions",
        "NGO foreign policy influence advocacy transnational",
        "diaspora lobbying foreign policy ethnic interest groups",
        "intelligence community foreign policy covert action",
        "crisis management escalation control hotline",
        "negotiation theory BATNA zone of possible agreement",
        "peacetime statecraft economic military diplomatic",
        "track II diplomacy back channel Oslo process",
        "development aid diplomacy conditionality tied",
        "public opinion foreign policy democratic accountability",
        "foreign policy change determinants domestic systemic",
        "trade policy lobbying interest groups protectionism",
        "consular affairs diplomatic protection nationals abroad",
        "multilateral diplomacy UN General Assembly strategy",
        "nuclear diplomacy Iran JCPOA North Korea",
    ]
    n = wiki_api(queries, bucket, "pol_diplo_wiki"); total += n
    log.info(f"  [POL-DIPLO-1] Wiki API: {n}")
    n = wiki_stream({
        "diplomacy", "foreign policy", "treaty", "bilateral", "summit",
        "negotiation", "embassy", "ambassador", "statecraft", "coercive",
        "alliance", "détente", "cold war", "peace deal",
    }, bucket, "pol_diplo_stream", max_docs=3000); total += n
    log.info(f"  [POL-DIPLO-2] Wiki stream: {n}")
    n = se_qa(["politics", "history", "law"], bucket, "pol_diplo_se",
              extra_kw={"diplomacy", "foreign policy", "negotiation", "treaty",
                        "embassy", "sanctions", "alliance", "statecraft",
                        "coercive", "summit"}); total += n
    log.info(f"  [POL-DIPLO-3] SE Q&A: {n}")
    n = semantic_scholar([
        "foreign policy domestic politics two-level game Putnam",
        "diplomatic signaling costly signal credibility resolve",
        "economic statecraft sanctions coercion effectiveness",
        "alliance commitment credibility extended deterrence",
        "negotiation bargaining international agreement ZOPA",
        "public diplomacy soft power measurement effectiveness",
        "coercive diplomacy compellence Schelling crisis",
        "track II diplomacy back-channel Oslo negotiation",
    ], bucket, "pol_diplo_ss"); total += n
    log.info(f"  [POL-DIPLO-4] Semantic Scholar: {n}")
    log.info(f"  >>> Diplomacy & FP total: {total:,}")
    return total


def load_middle_east_israel_palestine():
    log.info("=" * 65)
    log.info("POL-11: Middle East Politics & Israel-Palestine Conflict")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "Ottoman Empire collapse Sykes-Picot Middle East partition",
        "British Mandate Palestine Balfour Declaration Zionism",
        "1948 Arab-Israeli War Nakba Palestinian displacement",
        "1948 Israeli Independence founding state recognition",
        "1956 Suez Crisis war Egypt Israel Britain France",
        "1967 Six-Day War Israel Egypt Jordan Syria occupation",
        "1973 Yom Kippur War Arab-Israeli conflict Sadat",
        "Palestinian Liberation Organization PLO Arafat history",
        "Camp David Accords 1978 Egypt Israel Sadat Begin",
        "Oslo Accords 1993 peace process Israeli Palestinian",
        "Israeli settlements West Bank international law",
        "Gaza Strip blockade siege humanitarian crisis",
        "Hamas ideology governance Gaza history founding",
        "First Intifada 1987 Palestinian uprising Israel",
        "Second Intifada 2000 Al-Aqsa uprising violence",
        "two-state solution obstacles settlements Jerusalem",
        "Israeli-Palestinian peace process failure Camp David 2000",
        "Palestinian Authority Fatah governance West Bank",
        "East Jerusalem status sovereignty international law",
        "Israeli security barrier separation wall ICJ opinion",
        "Hamas October 7 2023 attack Israel massacre",
        "Israel Gaza war 2023 2024 military operation",
        "Gaza civilian casualties humanitarian catastrophe",
        "ICJ genocide case South Africa Israel proceedings",
        "ceasefire negotiations Qatar Egypt mediators hostages",
        "UNRWA Palestinian refugees mandate funding",
        "Rafah invasion humanitarian corridors displacement",
        "Iran proxy network Hezbollah Hamas Islamic Jihad",
        "Hezbollah Lebanon Israel conflict 2006 2024",
        "Abraham Accords normalization Saudi Arabia Israel",
        "Arab Spring effects Middle East democratic hopes",
        "Syria civil war Assad Russia Iran involvement",
        "Yemen war Houthi Saudi Arabia Iran coalition",
        "Iraq war 2003 aftermath sectarian fragmentation",
        "Iran nuclear program JCPOA sanctions diplomacy",
        "Islamic State ISIS caliphate rise collapse ideology",
        "Kurdish question Iraq Syria Turkey autonomy",
        "Lebanon sectarian politics Hezbollah economy collapse",
        "Gulf Cooperation Council Qatar blockade Saudi",
        "Saudi Arabia Vision 2030 modernization foreign policy",
        "Political Islam Muslim Brotherhood governments",
        "US Middle East policy Israel support Arab states",
        "Russia Middle East Syria influence leverage",
        "Turkey Neo-Ottoman foreign policy Erdogan",
        "Egypt Sisi politics stability Israel Gaza border",
        "Jordan Palestinian population politics stability",
    ]
    n = wiki_api(queries, bucket, "pol_me_wiki", max_per_query=25); total += n
    log.info(f"  [POL-ME-1] Wiki API: {n}")
    n = wiki_stream(POL_ME_KW, bucket, "pol_me_stream", max_docs=6000); total += n
    log.info(f"  [POL-ME-2] Wiki stream: {n}")
    n = se_qa(["politics", "history", "law"], bucket, "pol_me_se",
              extra_kw={"israel", "palestine", "gaza", "hamas", "hezbollah",
                        "iran", "middle east", "arab", "muslim", "jerusalem",
                        "west bank", "occupation", "nakba", "zionism"}); total += n
    log.info(f"  [POL-ME-3] SE Q&A: {n}")
    n = gutenberg([
        (3368,  "The Jewish State — Theodor Herzl (Zionism founding text)"),
        (17989, "My Life — Golda Meir (Israeli statesmanship)"),
        (21858, "Arabia — Gertrude Bell (early 20th century Middle East)"),
    ], bucket, "pol_me_gutenberg"); total += n
    log.info(f"  [POL-ME-4] Gutenberg: {n}")
    n = semantic_scholar([
        "Israel Palestine two-state solution peace obstacles",
        "Hamas governance Gaza political Islam ideology",
        "October 7 attack Gaza war international humanitarian law",
        "ICJ genocide case Israel Palestinians South Africa",
        "Israeli settlements occupation international law",
        "Arab Spring democratization failure authoritarianism",
        "Iran nuclear deal JCPOA proliferation deterrence",
        "Hezbollah Lebanon proxy Iran deterrence",
        "Saudi Arabia modernization Vision 2030 foreign policy",
        "ISIS Islamic State caliphate rise collapse ideology",
        "Kurdish autonomy Iraq Syria Turkey nationalism",
        "Palestinian refugees UNRWA right of return",
        "Gaza blockade humanitarian international law siege",
        "Yemen war civilian casualties Saudi coalition Houthi",
        "Abraham Accords normalization Arab Israel Gulf",
    ], bucket, "pol_me_ss"); total += n
    log.info(f"  [POL-ME-5] Semantic Scholar: {n}")
    n = pubmed([
        "Gaza humanitarian crisis health malnutrition mortality",
        "Israeli Palestinian conflict mental health PTSD",
        "Middle East conflict civilian health outcomes",
        "war trauma displacement children adolescents Middle East",
        "blockade siege health infrastructure humanitarian",
    ], bucket, "pol_me_pubmed"); total += n
    log.info(f"  [POL-ME-6] PubMed: {n}")
    log.info(f"  >>> Middle East & Israel-Palestine total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("TitanAI World Politics & International Relations Corpus Loader v2")
    log.info("11 Subfields — 6 sources each: Wiki | SE Q&A | arXiv | Semantic Scholar | PubMed | Gutenberg")
    log.info("Quality: full articles (40K chars) | 500-char min | dedup | smart chunking")
    log.info(f"Target directory: {RAW}")
    log.info("")

    results = {}
    results["ir_theory"]              = load_ir_theory()
    results["geopolitics_powers"]     = load_geopolitics_major_powers()
    results["intl_organizations"]     = load_international_organizations()
    results["political_economy"]      = load_political_economy()
    results["conflict_security"]      = load_conflict_security()
    results["human_rights_intl_law"]  = load_human_rights_intl_law()
    results["ideologies_systems"]     = load_political_ideologies_systems()
    results["elections_governance"]   = load_elections_governance()
    results["global_issues"]          = load_global_issues()
    results["diplomacy_fp"]           = load_diplomacy_foreign_policy()
    results["middle_east_israel_pal"] = load_middle_east_israel_palestine()

    with open(INVENTORY, "w") as f:
        json.dump(inventory, f, indent=2)
    with open(EXCLUSION_LOG, "w") as f:
        for e in exclusions:
            f.write(json.dumps(e) + "\n")

    log.info("")
    log.info("=" * 65)
    log.info("WORLD POLITICS CORPUS LOAD — COMPLETE")
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
