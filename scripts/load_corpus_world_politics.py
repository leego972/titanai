#!/usr/bin/env python3
"""
TitanAI World Politics & International Relations Corpus Loader
==============================================================
Adds comprehensive world politics training data to corpus_C_technical.
Target: 50,000+ documents across all key subfields.

Subfields covered:
  1.  International Relations Theory   — realism, liberalism, constructivism,
                                          Waltz, Mearsheimer, Keohane, Wendt
  2.  Geopolitics & Major Powers       — US, China, Russia, EU foreign policy,
                                          Indo-Pacific, Africa, Latin America
  3.  International Organizations      — UN, NATO, EU, ASEAN, AU, WTO, IMF,
                                          World Bank, ICC, OPEC, G7, G20
  4.  Diplomacy & Foreign Policy       — negotiation, treaties, summits,
                                          bilateral/multilateral relations
  5.  Political Economy                — trade wars, sanctions, globalization,
                                          development finance, debt diplomacy
  6.  Conflict, Security & War         — armed conflict, civil war, terrorism,
                                          nuclear deterrence, peacekeeping
  7.  Human Rights & International Law — UDHR, ICC, Geneva Conventions,
                                          refugee law, R2P doctrine
  8.  Political Ideologies & Systems   — democracy, authoritarianism, populism,
                                          nationalism, socialism, fascism
  9.  Electoral Systems & Governance   — voting systems, corruption, rule of law,
                                          state capacity, democratic backsliding
 10.  Global Issues                    — climate politics, nuclear proliferation,
                                          migration, pandemic governance, cyber

Sources per subfield:
  • Wikipedia REST API   — targeted queries per subfield
  • wikimedia/wikipedia  — streaming keyword filter
  • arXiv preprints      — econ.GN, econ.HI, q-bio.PE, cs.GT, q-fin.EC
  • Project Gutenberg    — classic political texts
  • OpenAlex API         — open-access political science research

Run:
  pip install datasets requests tqdm
  python scripts/load_corpus_world_politics.py
"""

import os, sys, json, time, logging, requests
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


# ── Reusable loaders ───────────────────────────────────────────────────────────
def wiki_api(queries, bucket_dir, tag, max_per_query=20):
    API = "https://en.wikipedia.org/w/api.php"
    seen, docs = set(), []
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
                    if len(text) > 300:
                        docs.append(f"# {t}\n\n{text[:10000]}")
                time.sleep(0.12)
        except Exception as e:
            log.warning(f"    wiki_api '{q[:40]}': {e}")
        time.sleep(0.05)
    return write_docs(bucket_dir, docs, tag)

def wiki_stream(keywords, bucket_dir, tag, max_docs=5000):
    try:
        from datasets import load_dataset
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en",
                            split="train", streaming=True)
        kw = {k.lower() for k in keywords}
        docs, scanned = [], 0
        for item in wiki:
            scanned += 1
            if scanned > 900_000:
                break
            title = item.get("title", "").lower()
            text  = item.get("text", "").strip()
            if len(text) < 300:
                continue
            if any(k in title or k in text[:600].lower() for k in kw):
                docs.append(f"# {item['title']}\n\n{text[:9000]}")
            if len(docs) >= max_docs:
                break
        return write_docs(bucket_dir, docs, tag)
    except Exception as e:
        record_exclusion(f"wiki_stream:{tag}", str(e))
        return 0

def arxiv_cats(cats, bucket_dir, tag, extra_kw=None, max_docs=3000):
    try:
        from datasets import load_dataset
        arxiv = load_dataset("Cornell-University/arxiv",
                             split="train", streaming=True)
        cat_set = set(cats)
        docs = []
        for i, item in enumerate(arxiv):
            if i > 5_000_000:
                break
            c = set((item.get("categories", "") or "").split())
            if not c.intersection(cat_set):
                continue
            title = (item.get("title", "") or "").replace("\n", " ").strip()
            abst  = (item.get("abstract", "") or "").replace("\n", " ").strip()
            if len(abst) < 80:
                continue
            if extra_kw:
                combo = (title + " " + abst).lower()
                if not any(k in combo for k in extra_kw):
                    continue
            docs.append(f"Title: {title}\nCategories: {' '.join(sorted(c))}\n\n"
                        f"Abstract:\n{abst}")
            if len(docs) >= max_docs:
                break
        return write_docs(bucket_dir, docs, tag)
    except Exception as e:
        record_exclusion(f"arxiv:{tag}", str(e))
        return 0

def gutenberg(ids, bucket_dir, tag):
    docs = []
    for gid, desc in ids:
        for url in [
            f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
            f"https://gutenberg.org/files/{gid}/{gid}-0.txt",
            f"https://gutenberg.org/files/{gid}/{gid}.txt",
        ]:
            try:
                r = requests.get(url, timeout=25)
                if r.status_code == 200 and len(r.text) > 500:
                    text  = gutenberg_strip(r.text)
                    words = text.split()
                    for s in range(0, len(words), 1800):
                        chunk = " ".join(words[s:s + 1800])
                        if len(chunk) > 300:
                            docs.append(f"[{desc} — Gutenberg #{gid}]\n\n{chunk}")
                    log.info(f"    Gutenberg #{gid}: {len(words)} words")
                    break
            except Exception as e:
                log.warning(f"    Gutenberg #{gid}: {e}")
        time.sleep(0.6)
    return write_docs(bucket_dir, docs, tag)

def openalex(queries, bucket_dir, tag, label="Political Science Research"):
    docs = []
    for q in queries:
        try:
            r = requests.get(
                f"https://api.openalex.org/works?search={requests.utils.quote(q)}"
                "&per-page=100&filter=open_access.is_oa:true"
                "&select=title,abstract_inverted_index",
                timeout=20, headers={"User-Agent": "TitanAI/1.0"}
            )
            if r.status_code != 200:
                continue
            for w in r.json().get("results", []):
                title = (w.get("title", "") or "").strip()
                inv   = w.get("abstract_inverted_index") or {}
                if not inv:
                    continue
                mx = max(p for ps in inv.values() for p in ps)
                wl = [""] * (mx + 1)
                for word, ps in inv.items():
                    for p in ps:
                        wl[p] = word
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

POL_ALL_KW = (POL_IR_KW | POL_GEO_KW | POL_ORG_KW |
              POL_ECON_KW | POL_CONFLICT_KW | POL_IDEOLOGY_KW)


# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_ir_theory():
    log.info("=" * 65)
    log.info("POL-1: International Relations Theory")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "realism international relations Waltz Morgenthau",
        "liberalism international relations Keohane Nye",
        "constructivism international relations Wendt Finnemore",
        "balance of power theory hegemony",
        "offensive realism Mearsheimer great power",
        "democratic peace theory Kant perpetual peace",
        "security dilemma spiral model",
        "anarchy international system states",
        "soft power hard power Nye",
        "power transition theory Organski",
        "english school international society Bull",
        "world systems theory Wallerstein core periphery",
        "dependency theory underdevelopment",
        "neoliberal institutionalism international cooperation",
        "postcolonialism international relations",
        "feminist international relations gender security",
        "critical theory international relations Gramsci",
        "foreign policy analysis decision making",
        "securitization theory Copenhagen School",
        "norm diffusion socialization international",
    ]
    n = wiki_api(queries, bucket, "pol_ir_theory_wiki"); total += n
    record_source("corpus_C_technical", "Wikipedia IR Theory API",
                  "https://en.wikipedia.org", approx_mb([""] * n),
                  "IR theory: realism, liberalism, constructivism, power theory", n)
    log.info(f"  [POL-IR-1] Wiki API: {n}")

    n = wiki_stream(POL_IR_KW, bucket, "pol_ir_theory_stream", max_docs=3000)
    total += n
    log.info(f"  [POL-IR-2] Wiki stream: {n}")

    n = arxiv_cats(["econ.GN", "q-fin.EC"], bucket, "pol_ir_arxiv",
                   extra_kw={"international", "geopolit", "foreign policy",
                             "hegemony", "power", "regime", "alliance", "war"},
                   max_docs=2000)
    total += n
    log.info(f"  [POL-IR-3] arXiv: {n}")

    n = openalex([
        "balance of power great power competition",
        "democratic peace theory empirical test",
        "constructivism norms international relations",
        "soft power public diplomacy effectiveness",
        "securitization theory threat construction",
    ], bucket, "pol_ir_openalex", "IR Theory Research")
    total += n
    log.info(f"  [POL-IR-4] OpenAlex: {n}")

    log.info(f"  >>> IR Theory total: {total:,}")
    return total


def load_geopolitics_major_powers():
    log.info("=" * 65)
    log.info("POL-2: Geopolitics & Major Powers")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "US foreign policy grand strategy",
        "China foreign policy Belt Road Initiative",
        "Russia foreign policy Ukraine NATO",
        "European Union foreign security policy",
        "Indo-Pacific strategy Quad AUKUS",
        "NATO expansion enlargement Russia",
        "South China Sea territorial dispute",
        "Taiwan cross-strait relations",
        "Middle East geopolitics Israel Palestine",
        "Africa geopolitics China US competition",
        "Latin America politics left right turn",
        "India foreign policy strategic autonomy",
        "Japan remilitarization foreign policy",
        "energy geopolitics oil gas pipelines",
        "Arctic geopolitics sovereignty resources",
        "BRICS emerging powers global governance",
        "sanctions Russia Iran North Korea",
        "technology competition semiconductors AI",
        "rare earth minerals critical supply chains",
        "global south non-aligned movement new",
    ]
    n = wiki_api(queries, bucket, "pol_geo_wiki"); total += n
    record_source("corpus_C_technical", "Wikipedia Geopolitics API",
                  "https://en.wikipedia.org", approx_mb([""] * n),
                  "Major powers, regional politics, geopolitical competition", n)
    log.info(f"  [POL-GEO-1] Wiki API: {n}")

    n = wiki_stream(POL_GEO_KW, bucket, "pol_geo_stream", max_docs=4000)
    total += n
    log.info(f"  [POL-GEO-2] Wiki stream: {n}")

    n = openalex([
        "US China strategic competition technology decoupling",
        "Belt Road Initiative debt sustainability Africa",
        "NATO enlargement security dilemma Russia",
        "Indo-Pacific security quad aukus China",
        "energy transition geopolitics oil gas",
        "Taiwan strait crisis deterrence",
        "Middle East regional order Saudi Iran",
        "Africa political economy governance development",
    ], bucket, "pol_geo_openalex", "Geopolitics Research")
    total += n
    log.info(f"  [POL-GEO-3] OpenAlex: {n}")

    log.info(f"  >>> Geopolitics total: {total:,}")
    return total


def load_international_organizations():
    log.info("=" * 65)
    log.info("POL-3: International Organizations")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "United Nations Security Council veto power",
        "UN General Assembly resolutions voting",
        "UN peacekeeping operations effectiveness",
        "NATO collective defence Article 5",
        "European Union institutions decision making",
        "EU Common Foreign Security Policy",
        "ASEAN regional security non-interference",
        "African Union peace security council",
        "World Trade Organization dispute settlement",
        "IMF structural adjustment conditionality",
        "World Bank development lending governance",
        "G7 G20 global economic governance",
        "International Criminal Court jurisdiction",
        "OPEC oil production quotas",
        "International Atomic Energy Agency safeguards",
        "WHO global health governance pandemic",
        "Paris Climate Agreement implementation",
        "UN Human Rights Council mechanisms",
        "multilateralism crisis reform international",
        "regional integration comparative EU ASEAN AU",
    ]
    n = wiki_api(queries, bucket, "pol_org_wiki"); total += n
    record_source("corpus_C_technical", "Wikipedia Int'l Orgs API",
                  "https://en.wikipedia.org", approx_mb([""] * n),
                  "UN, NATO, EU, WTO, IMF, regional organizations", n)
    log.info(f"  [POL-ORG-1] Wiki API: {n}")

    n = wiki_stream(POL_ORG_KW, bucket, "pol_org_stream", max_docs=3000)
    total += n
    log.info(f"  [POL-ORG-2] Wiki stream: {n}")

    n = openalex([
        "UN Security Council reform veto legitimacy",
        "IMF conditionality austerity developing countries",
        "WTO dispute settlement appellate body crisis",
        "NATO burden sharing collective defence",
        "EU integration deepening widening",
        "multilateral institutions effectiveness reform",
    ], bucket, "pol_org_openalex", "International Organizations Research")
    total += n
    log.info(f"  [POL-ORG-3] OpenAlex: {n}")

    log.info(f"  >>> International Organizations total: {total:,}")
    return total


def load_political_economy():
    log.info("=" * 65)
    log.info("POL-4: Political Economy & Trade")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "trade war protectionism tariffs retaliations",
        "economic sanctions effectiveness foreign policy",
        "globalization backlash inequality",
        "Washington consensus neoliberal reform",
        "development aid effectiveness conditionality",
        "sovereign debt crisis restructuring IMF",
        "currency manipulation exchange rate politics",
        "economic nationalism industrial policy",
        "supply chain security friend-shoring",
        "state capitalism China model",
        "foreign direct investment political risk",
        "economic coercion grey zone tools",
        "financial sanctions SWIFT dollar weaponization",
        "climate finance green transition politics",
        "digital trade data governance",
        "international tax avoidance OECD BEPS",
        "commodity prices politics resource curse",
        "food security geopolitics agriculture",
        "technology export controls semiconductors",
        "debt trap diplomacy Africa China",
    ]
    n = wiki_api(queries, bucket, "pol_econ_wiki"); total += n
    record_source("corpus_C_technical", "Wikipedia Political Economy API",
                  "https://en.wikipedia.org", approx_mb([""] * n),
                  "Trade, sanctions, globalization, development finance", n)
    log.info(f"  [POL-ECON-1] Wiki API: {n}")

    n = wiki_stream(POL_ECON_KW, bucket, "pol_econ_stream", max_docs=3000)
    total += n
    log.info(f"  [POL-ECON-2] Wiki stream: {n}")

    n = arxiv_cats(["econ.GN", "econ.HI", "q-fin.EC"], bucket, "pol_econ_arxiv",
                   extra_kw={"trade", "sanction", "globalization", "political",
                             "development", "inequality", "tariff", "currency"},
                   max_docs=2000)
    total += n
    log.info(f"  [POL-ECON-3] arXiv: {n}")

    n = openalex([
        "economic sanctions effectiveness signaling",
        "trade war retaliatory tariff welfare",
        "globalization inequality political backlash",
        "China development finance Africa conditionality",
        "dollar hegemony international monetary system",
        "industrial policy infant industry protection",
        "supply chain resilience geopolitical risk",
    ], bucket, "pol_econ_openalex", "Political Economy Research")
    total += n
    log.info(f"  [POL-ECON-4] OpenAlex: {n}")

    log.info(f"  >>> Political Economy total: {total:,}")
    return total


def load_conflict_security():
    log.info("=" * 65)
    log.info("POL-5: Conflict, Security & War")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "causes of war interstate conflict quantitative",
        "civil war onset greed grievance",
        "nuclear deterrence credibility extended",
        "arms control treaties verification",
        "terrorism counterterrorism effectiveness",
        "insurgency counterinsurgency COIN doctrine",
        "hybrid warfare grey zone operations Russia",
        "cyber warfare state actors attribution",
        "information warfare propaganda disinformation",
        "peacekeeping peacebuilding effectiveness UN",
        "conflict resolution mediation negotiation",
        "genocide prevention responsibility to protect",
        "proxy war great powers third party",
        "military alliances deterrence credibility",
        "arms race action reaction model",
        "non-proliferation nuclear weapons NPT",
        "chemical biological weapons conventions",
        "private military companies mercenaries",
        "drone warfare targeted killing",
        "space militarization anti-satellite weapons",
    ]
    n = wiki_api(queries, bucket, "pol_conflict_wiki"); total += n
    record_source("corpus_C_technical", "Wikipedia Conflict/Security API",
                  "https://en.wikipedia.org", approx_mb([""] * n),
                  "War, deterrence, terrorism, peacekeeping, arms control", n)
    log.info(f"  [POL-CONF-1] Wiki API: {n}")

    n = wiki_stream(POL_CONFLICT_KW, bucket, "pol_conflict_stream", max_docs=4000)
    total += n
    log.info(f"  [POL-CONF-2] Wiki stream: {n}")

    n = arxiv_cats(["econ.GN"], bucket, "pol_conflict_arxiv",
                   extra_kw={"war", "conflict", "terrorism", "nuclear", "military",
                             "deterrence", "arms", "insurgency", "peace"},
                   max_docs=2000)
    total += n
    log.info(f"  [POL-CONF-3] arXiv: {n}")

    n = openalex([
        "civil war onset greed grievance lootable resources",
        "nuclear deterrence extended credibility allies",
        "terrorism radicalization prevention deradicalization",
        "cyberwarfare state attribution international law",
        "UN peacekeeping effectiveness outcomes",
        "conflict resolution mediation third party",
        "hybrid warfare Russia Ukraine information",
    ], bucket, "pol_conflict_openalex", "Conflict & Security Research")
    total += n
    log.info(f"  [POL-CONF-4] OpenAlex: {n}")

    log.info(f"  >>> Conflict & Security total: {total:,}")
    return total


def load_human_rights_intl_law():
    log.info("=" * 65)
    log.info("POL-6: Human Rights & International Law")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "Universal Declaration Human Rights UDHR",
        "International Criminal Court Rome Statute",
        "Geneva Conventions international humanitarian law",
        "responsibility to protect R2P norm",
        "refugee law 1951 convention asylum",
        "UN Human Rights Council mechanisms",
        "treaty body monitoring implementation",
        "transitional justice truth commission",
        "customary international law jus cogens",
        "state sovereignty human rights tension",
        "international law use of force jus ad bellum",
        "war crimes crimes against humanity genocide",
        "international criminal tribunals ICTY ICTR",
        "human trafficking modern slavery",
        "children rights convention CRC",
        "torture convention absolute prohibition",
        "economic social cultural rights ICESCR",
        "indigenous peoples rights UNDRIP",
        "climate change human rights nexus",
        "business human rights Ruggie principles",
    ]
    n = wiki_api(queries, bucket, "pol_hr_wiki"); total += n
    record_source("corpus_C_technical", "Wikipedia Human Rights/Int'l Law API",
                  "https://en.wikipedia.org", approx_mb([""] * n),
                  "UDHR, ICC, Geneva Conventions, R2P, refugee law", n)
    log.info(f"  [POL-HR-1] Wiki API: {n}")

    n = wiki_stream({
        "human rights", "international law", "humanitarian law",
        "refugee", "asylum", "war crime", "genocide", "torture",
        "international criminal court", "universal jurisdiction",
        "responsibility to protect", "transitional justice",
    }, bucket, "pol_hr_stream", max_docs=2000)
    total += n
    log.info(f"  [POL-HR-2] Wiki stream: {n}")

    n = openalex([
        "international criminal court complementarity effectiveness",
        "responsibility to protect humanitarian intervention",
        "refugee protection international law non-refoulement",
        "transitional justice reconciliation accountability",
        "human rights conditionality foreign policy",
        "universal jurisdiction war crimes domestic prosecution",
    ], bucket, "pol_hr_openalex", "Human Rights & Int'l Law Research")
    total += n
    log.info(f"  [POL-HR-3] OpenAlex: {n}")

    log.info(f"  >>> Human Rights & Int'l Law total: {total:,}")
    return total


def load_political_ideologies_systems():
    log.info("=" * 65)
    log.info("POL-7: Political Ideologies & Systems")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "democracy varieties liberal illiberal",
        "authoritarianism types electoral competitive",
        "totalitarianism theory Arendt Orwell",
        "populism left right causes effects",
        "nationalism ethnic civic varieties",
        "socialism communism Marxism history",
        "fascism ideology historical cases",
        "social democracy welfare state Scandinavian",
        "conservatism liberalism political philosophy",
        "libertarianism anarchism minimal state",
        "green politics ecological parties",
        "feminist political theory gender equality",
        "democratic backsliding autocratization",
        "political polarization partisan divide",
        "religious fundamentalism politics",
        "identity politics multiculturalism",
        "revolutionary movements ideology",
        "constitutionalism rule of law separation powers",
        "federalism decentralization multilevel governance",
        "hybrid regimes competitive authoritarianism",
    ]
    n = wiki_api(queries, bucket, "pol_ideology_wiki"); total += n
    record_source("corpus_C_technical", "Wikipedia Ideologies/Systems API",
                  "https://en.wikipedia.org", approx_mb([""] * n),
                  "Democracy, authoritarianism, populism, nationalism, ideologies", n)
    log.info(f"  [POL-IDEO-1] Wiki API: {n}")

    n = wiki_stream(POL_IDEOLOGY_KW, bucket, "pol_ideology_stream", max_docs=3000)
    total += n
    log.info(f"  [POL-IDEO-2] Wiki stream: {n}")

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
    ], bucket, "pol_ideology_gutenberg")
    total += n
    record_source("corpus_C_technical", "Gutenberg Classic Political Texts",
                  "https://gutenberg.org", approx_mb([""] * n),
                  "Federalist Papers, Marx, Hobbes, Locke, Rousseau, Plato, Aristotle", n)
    log.info(f"  [POL-IDEO-3] Gutenberg classics: {n}")

    n = openalex([
        "democratic backsliding autocratization populism",
        "political polarization partisan identity",
        "populism economic insecurity cultural backlash",
        "nationalism ethnic civic identity salience",
        "authoritarian resilience regime survival strategies",
        "comparative democracy electoral systems institutions",
    ], bucket, "pol_ideology_openalex", "Political Ideology Research")
    total += n
    log.info(f"  [POL-IDEO-4] OpenAlex: {n}")

    log.info(f"  >>> Political Ideologies total: {total:,}")
    return total


def load_elections_governance():
    log.info("=" * 65)
    log.info("POL-8: Electoral Systems & Governance")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "electoral systems proportional representation",
        "first past the post majoritarian elections",
        "electoral fraud rigging manipulation",
        "voter turnout participation determinants",
        "political parties party systems Duverger",
        "campaign finance money politics",
        "media and elections disinformation",
        "corruption governance public sector",
        "rule of law judicial independence",
        "state capacity bureaucracy effectiveness",
        "decentralization local government",
        "civil society NGOs democracy",
        "political representation minorities women",
        "parliamentary presidential semi-presidential",
        "constitutional design power sharing",
        "electoral violence post-election conflict",
        "international election monitoring",
        "digital democracy e-voting blockchain",
        "deliberative democracy citizens assembly",
        "bureaucratic politics Allison models",
    ]
    n = wiki_api(queries, bucket, "pol_gov_wiki"); total += n
    record_source("corpus_C_technical", "Wikipedia Elections/Governance API",
                  "https://en.wikipedia.org", approx_mb([""] * n),
                  "Electoral systems, governance, corruption, rule of law", n)
    log.info(f"  [POL-GOV-1] Wiki API: {n}")

    n = wiki_stream({
        "election", "voting", "electoral", "parliament", "government",
        "democracy", "corruption", "governance", "constitution", "rule of law",
        "political party", "campaign", "referendum", "representative",
    }, bucket, "pol_gov_stream", max_docs=3000)
    total += n
    log.info(f"  [POL-GOV-2] Wiki stream: {n}")

    n = openalex([
        "electoral system proportional majoritarian outcomes",
        "corruption governance development economic growth",
        "judicial independence autocratization courts",
        "women political representation quotas barriers",
        "voter turnout mobilization suppression",
        "campaign finance money political outcomes",
    ], bucket, "pol_gov_openalex", "Elections & Governance Research")
    total += n
    log.info(f"  [POL-GOV-3] OpenAlex: {n}")

    log.info(f"  >>> Elections & Governance total: {total:,}")
    return total


def load_global_issues():
    log.info("=" * 65)
    log.info("POL-9: Global Issues & Transnational Politics")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "climate change politics Paris Agreement",
        "nuclear proliferation non-proliferation NPT",
        "international migration refugee politics",
        "global pandemic governance WHO reform",
        "cyber security internet governance",
        "artificial intelligence governance regulation",
        "space law governance outer space treaty",
        "biodiversity loss environmental politics",
        "food security global hunger politics",
        "water scarcity transboundary rivers",
        "global health equity vaccine nationalism",
        "disinformation information warfare democracy",
        "dark money offshore finance tax havens",
        "transnational organized crime drug trafficking",
        "human smuggling trafficking networks",
        "social media revolution Arab Spring",
        "climate migration displacement",
        "ocean governance high seas treaty",
        "global inequality poverty Sustainable Development",
        "digital authoritarianism surveillance capitalism",
    ]
    n = wiki_api(queries, bucket, "pol_global_wiki"); total += n
    record_source("corpus_C_technical", "Wikipedia Global Issues API",
                  "https://en.wikipedia.org", approx_mb([""] * n),
                  "Climate, nuclear, migration, pandemic, cyber, global governance", n)
    log.info(f"  [POL-GLOB-1] Wiki API: {n}")

    n = wiki_stream({
        "climate politics", "nuclear", "migration", "pandemic governance",
        "cyber", "artificial intelligence governance", "global health",
        "food security", "water conflict", "disinformation",
        "transnational crime", "human trafficking", "sustainable development",
    }, bucket, "pol_global_stream", max_docs=3000)
    total += n
    log.info(f"  [POL-GLOB-2] Wiki stream: {n}")

    n = openalex([
        "climate change international negotiations UNFCCC",
        "nuclear non-proliferation treaty compliance",
        "international migration governance refugee protection",
        "pandemic preparedness global health security",
        "internet governance multistakeholder model",
        "artificial intelligence governance ethics policy",
        "climate migration displacement adaptation",
        "disinformation elections democracy resilience",
    ], bucket, "pol_global_openalex", "Global Issues Research")
    total += n
    log.info(f"  [POL-GLOB-3] OpenAlex: {n}")

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
        "treaty negotiation ratification international",
        "foreign policy decision making process",
        "summit diplomacy leaders bilateral meetings",
        "economic diplomacy trade investment tools",
        "public diplomacy soft power cultural",
        "coercive diplomacy compellence deterrence",
        "alliance management intra-alliance politics",
        "foreign policy domestic politics two-level games",
        "diplomatic history cold war détente",
        "global governance reform multilateral",
        "non-governmental organizations foreign policy",
        "diaspora lobbying foreign policy influence",
        "intelligence community foreign policy",
        "crisis management escalation control",
        "negotiation theory strategies BATNA",
        "peacetime statecraft instruments power",
        "track II diplomacy back channel",
        "development aid diplomacy conditionality",
        "public opinion foreign policy democratic",
    ]
    n = wiki_api(queries, bucket, "pol_diplo_wiki"); total += n
    record_source("corpus_C_technical", "Wikipedia Diplomacy/FP API",
                  "https://en.wikipedia.org", approx_mb([""] * n),
                  "Diplomacy, foreign policy analysis, negotiation, statecraft", n)
    log.info(f"  [POL-DIPLO-1] Wiki API: {n}")

    n = wiki_stream({
        "diplomacy", "foreign policy", "treaty", "bilateral", "summit",
        "negotiation", "embassy", "ambassador", "statecraft", "coercive",
        "alliance", "détente", "cold war", "peace deal",
    }, bucket, "pol_diplo_stream", max_docs=2000)
    total += n
    log.info(f"  [POL-DIPLO-2] Wiki stream: {n}")

    n = openalex([
        "foreign policy domestic politics two-level game",
        "diplomatic signaling costly signal credibility",
        "economic statecraft sanctions coercion",
        "alliance commitment credibility deterrence",
        "negotiation bargaining international agreement",
        "public diplomacy soft power measurement",
    ], bucket, "pol_diplo_openalex", "Diplomacy & FP Research")
    total += n
    log.info(f"  [POL-DIPLO-3] OpenAlex: {n}")

    log.info(f"  >>> Diplomacy & FP total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("TitanAI World Politics & International Relations Corpus Loader")
    log.info("10 Subfields: IR Theory | Geopolitics | Int'l Orgs | Political Economy")
    log.info("              Conflict | Human Rights | Ideologies | Governance")
    log.info("              Global Issues | Diplomacy")
    log.info(f"Target directory: {RAW}")
    log.info("")

    results = {}
    results["ir_theory"]             = load_ir_theory()
    results["geopolitics_powers"]    = load_geopolitics_major_powers()
    results["intl_organizations"]    = load_international_organizations()
    results["political_economy"]     = load_political_economy()
    results["conflict_security"]     = load_conflict_security()
    results["human_rights_intl_law"] = load_human_rights_intl_law()
    results["ideologies_systems"]    = load_political_ideologies_systems()
    results["elections_governance"]  = load_elections_governance()
    results["global_issues"]         = load_global_issues()
    results["diplomacy_fp"]          = load_diplomacy_foreign_policy()

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
