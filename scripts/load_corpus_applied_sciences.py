#!/usr/bin/env python3
"""
TitanAI Applied Sciences Corpus Loader
=======================================
Adds premium training data across six applied science and engineering domains.
Target: 60,000+ documents.

Subfields covered:
  1.  Apothecary & Herbal Medicine     — ethnobotany, pharmacognosy, traditional
                                          medicine systems (Ayurveda, TCM, Unani),
                                          medicinal plants, plant compounds,
                                          phytotherapy, historical apothecary
  2.  Crystals, Metals & Materials     — crystallography, crystal systems, gemology,
                                          metallurgy, alloys, nanomaterials, ceramics,
                                          semiconductors, superconductors, biomaterials,
                                          properties: optical, electrical, thermal
  3.  Magnets & Applied Magnetism      — electromagnetism, permanent magnets, MRI,
                                          magnetic storage, electric motors, maglev,
                                          magnetic sensors, spintronics, fusion
  4.  Complex Circuit Systems          — analog/digital circuits, IC design, VLSI,
                                          FPGA, PCB layout, power electronics,
                                          RF circuits, signal processing, EDA tools
  5.  Lasers & Photonics               — laser physics, types (CO2, fiber, diode,
                                          excimer), laser surgery, LiDAR, optical
                                          comms, holography, photonic chips
  6.  Propulsion Mechanics             — rocket propulsion (chemical, nuclear, ion),
                                          jet engines (turbojet, turbofan, ramjet),
                                          propeller theory, electric propulsion,
                                          spacecraft maneuvering, hypersonics

Sources per subfield:
  • Wikipedia REST API   — targeted queries
  • wikimedia/wikipedia  — streaming keyword filter
  • arXiv preprints      — physics, engineering, materials categories
  • Project Gutenberg    — classic texts (alchemy, natural philosophy)
  • OpenAlex API         — open-access research papers

Run:
  pip install datasets requests tqdm
  python scripts/load_corpus_applied_sciences.py
"""

import os, sys, json, time, logging, requests
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

def wiki_stream(keywords, bucket_dir, tag, max_docs=4000):
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

def openalex(queries, bucket_dir, tag, label="Applied Science Research"):
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
HERB_KW = {
    "herbal medicine", "medicinal plants", "ethnobotany", "pharmacognosy",
    "phytotherapy", "botanical medicine", "traditional medicine",
    "ayurveda", "traditional chinese medicine", "unani medicine",
    "apothecary", "materia medica", "plant alkaloids", "phytochemicals",
    "essential oils", "tincture", "decoction", "adaptogen",
    "echinacea", "valerian", "ginseng", "turmeric curcumin",
    "chamomile", "lavender", "peppermint", "elderberry", "ashwagandha",
    "plant secondary metabolites", "flavonoids", "terpenes", "saponins",
    "medicinal herb cultivation", "herbal pharmacopoeia",
}

CRYSTAL_KW = {
    "crystallography", "crystal structure", "crystal system", "unit cell",
    "x-ray diffraction", "bragg law", "crystal lattice", "miller indices",
    "gemology", "mineral properties", "quartz crystal", "diamond structure",
    "semiconductor crystal", "silicon crystal growth", "doping",
    "metallurgy", "alloy", "steel microstructure", "heat treatment",
    "titanium alloy", "nickel superalloy", "copper properties",
    "nanomaterials", "nanocrystal", "quantum dot", "graphene",
    "piezoelectric crystal", "ferroelectric", "pyroelectric",
    "superconductor", "ceramic material", "biomaterial implant",
    "optical crystal", "nonlinear optics crystal", "liquid crystal",
}

MAGNET_KW = {
    "electromagnetism", "magnetic field", "permanent magnet",
    "neodymium magnet", "ferromagnetism", "paramagnetism", "diamagnetism",
    "superconducting magnet", "mri magnetic resonance imaging",
    "electric motor", "generator electromagnetic induction",
    "transformer magnetic core", "maglev magnetic levitation",
    "magnetic storage hard drive", "spintronics",
    "hall effect sensor", "magnetic resonance",
    "electromagnet solenoid coil", "faraday law",
    "magnetic confinement fusion tokamak",
    "magnetohydrodynamics", "geomagnetic field",
    "magnetic nanoparticle drug delivery",
}

CIRCUIT_KW = {
    "electronic circuit", "circuit design", "analog circuit",
    "digital circuit", "integrated circuit", "vlsi design",
    "fpga field programmable", "pcb layout", "signal integrity",
    "operational amplifier", "filter design", "oscillator circuit",
    "power electronics", "switching power supply", "inverter",
    "rf circuit", "microwave circuit", "antenna design",
    "transistor bjt mosfet", "logic gate", "flip flop",
    "adc dac converter", "microcontroller", "embedded systems",
    "eda tools cadence", "spice simulation", "timing analysis",
    "ic fabrication cmos", "photolithography", "thermal management",
}

LASER_KW = {
    "laser physics", "stimulated emission", "laser cavity",
    "co2 laser", "nd yag laser", "fiber laser", "diode laser",
    "excimer laser", "femtosecond laser", "ultrafast laser",
    "laser surgery lasik", "photodynamic therapy laser",
    "lidar laser ranging", "laser cutting welding",
    "optical fiber communications", "wavelength division multiplexing",
    "holography laser", "laser interferometry",
    "photonic integrated circuit", "optical amplifier edfa",
    "nonlinear optics", "second harmonic generation",
    "laser spectroscopy", "raman spectroscopy",
    "laser cooling atom trap", "optical tweezers",
    "directed energy weapon laser",
}

PROPULSION_KW = {
    "rocket propulsion", "chemical rocket", "solid propellant",
    "liquid propellant rocket", "specific impulse", "thrust",
    "ion thruster electric propulsion", "hall effect thruster",
    "nuclear thermal rocket", "jet engine", "turbojet turbofan",
    "turboprop", "ramjet scramjet", "pulse detonation engine",
    "propeller blade aerodynamics", "cavitation propeller",
    "spacecraft orbital mechanics", "delta-v maneuver",
    "reentry vehicle thermal protection", "hypersonic vehicle",
    "aerospike nozzle", "combustion chamber", "nozzle design",
    "rocket staging", "launch vehicle", "space shuttle engine",
    "marine propulsion", "underwater propulsion",
    "electric aircraft propulsion",
}

ALL_KW = (HERB_KW | CRYSTAL_KW | MAGNET_KW |
          CIRCUIT_KW | LASER_KW | PROPULSION_KW)


# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_apothecary_herbal_medicine():
    log.info("=" * 65)
    log.info("APP-1: Apothecary & Herbal Medicine")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "pharmacognosy medicinal plant active compounds",
        "ethnobotany traditional plant medicine use",
        "Ayurveda herbal formulations doshas tridosha",
        "Traditional Chinese Medicine herbs acupuncture",
        "Unani Tibb Islamic medicine materia medica",
        "plant alkaloids morphine quinine berberine",
        "flavonoids polyphenols antioxidant plant",
        "essential oils terpenes therapeutic properties",
        "adaptogen ashwagandha ginseng rhodiola stress",
        "turmeric curcumin anti-inflammatory mechanisms",
        "echinacea immune modulation clinical trials",
        "valerian passionflower sleep sedation herbs",
        "St John's wort depression serotonin herb",
        "milk thistle silymarin hepatoprotective liver",
        "garlic allicin cardiovascular antimicrobial",
        "ginkgo biloba memory cognition circulation",
        "elderberry Sambucus antiviral immune",
        "chamomile apigenin anti-inflammatory calming",
        "peppermint menthol IBS pain analgesia",
        "lavender aromatherapy anxiety sleep",
        "cannabis cannabinoids therapeutic endocannabinoid",
        "opium history morphine laudanum apothecary",
        "phytochemical extraction isolation methods",
        "herbal drug interactions cytochrome P450",
        "WHO traditional medicine guidelines safety",
        "plant cell culture secondary metabolite production",
        "natural product drug discovery screening",
        "saponins glycosides tannins plant defense",
        "medicinal mushrooms reishi chaga lion's mane",
        "historical apothecary pharmacy Renaissance",
    ]
    n = wiki_api(queries, bucket, "app_herb_wiki"); total += n
    record_source("corpus_C_technical", "Wikipedia Herbal Medicine API",
                  "https://en.wikipedia.org", approx_mb([""] * n),
                  "Pharmacognosy, ethnobotany, Ayurveda, TCM, plant compounds", n)
    log.info(f"  [APP-HERB-1] Wiki API: {n}")

    n = wiki_stream(HERB_KW, bucket, "app_herb_stream", max_docs=4000)
    total += n
    log.info(f"  [APP-HERB-2] Wiki stream: {n}")

    n = gutenberg([
        (4698,  "Culpeper's Complete Herbal — Nicholas Culpeper"),
        (21214, "The Herbalist — Meyer (19th century materia medica)"),
        (27948, "A Modern Herbal — Mrs. M. Grieve"),
        (36988, "Herbals: Their Origin and Evolution — Arber"),
        (3666,  "The Doctrine and Literature of the Kabalah — Waite"),
    ], bucket, "app_herb_gutenberg")
    total += n
    record_source("corpus_C_technical", "Gutenberg Herbal & Apothecary Classics",
                  "https://gutenberg.org", approx_mb([""] * n),
                  "Culpeper's Herbal, Mrs. Grieve, historical materia medica", n)
    log.info(f"  [APP-HERB-3] Gutenberg: {n}")

    n = arxiv_cats(["q-bio.QM", "q-bio.MN"], bucket, "app_herb_arxiv",
                   extra_kw={"herbal", "plant", "phytochem", "alkaloid", "medicinal",
                             "ethnobotany", "natural product", "botanical"},
                   max_docs=2000)
    total += n
    log.info(f"  [APP-HERB-4] arXiv: {n}")

    n = openalex([
        "pharmacognosy medicinal plant secondary metabolites",
        "ethnobotany traditional knowledge medicinal plants",
        "curcumin anti-inflammatory mechanism clinical trial",
        "plant alkaloid biosynthesis pathway",
        "herbal medicine drug interaction safety",
        "natural product drug discovery antibiotic",
        "adaptogen ashwagandha cortisol stress clinical",
        "traditional Chinese medicine phytochemistry",
        "cannabis cannabidiol therapeutic applications",
        "essential oil antimicrobial mechanism terpene",
        "flavonoid polyphenol bioavailability cardiovascular",
        "Ayurvedic formulation clinical pharmacology",
    ], bucket, "app_herb_openalex", "Herbal Medicine Research")
    total += n
    log.info(f"  [APP-HERB-5] OpenAlex: {n}")

    log.info(f"  >>> Apothecary & Herbal Medicine total: {total:,}")
    return total


def load_crystals_metals_materials():
    log.info("=" * 65)
    log.info("APP-2: Crystals, Metals & Materials Science")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "crystal structure symmetry space group Bravais lattice",
        "X-ray diffraction crystallography Bragg law",
        "semiconductor silicon germanium band gap",
        "diamond crystal structure cubic hardness",
        "quartz piezoelectric crystal oscillator",
        "gemstone mineralogy optical properties",
        "metallurgy iron steel microstructure phase diagram",
        "titanium alloy aerospace biomedical properties",
        "nickel superalloy turbine blade high temperature",
        "copper electrical conductivity applications",
        "aluminum alloy lightweight aerospace automotive",
        "shape memory alloy nitinol applications",
        "steel heat treatment quenching tempering",
        "nanomaterials quantum confinement nanocrystal",
        "graphene properties electronic applications",
        "carbon nanotube mechanical electrical properties",
        "superconductor BCS theory high temperature",
        "ceramic material alumina zirconia properties",
        "piezoelectric material sensor actuator",
        "ferroelectric perovskite barium titanate",
        "liquid crystal display LCD structure",
        "optical glass refractive index dispersion",
        "nonlinear optical crystal KTP lithium niobate",
        "biomaterial implant osseointegration titanium",
        "corrosion protection coating metal",
        "powder metallurgy sintering processing",
        "thin film deposition PVD CVD sputtering",
        "transmission electron microscopy materials",
        "scanning electron microscopy EDS microanalysis",
        "density functional theory materials properties",
    ]
    n = wiki_api(queries, bucket, "app_cryst_wiki"); total += n
    record_source("corpus_C_technical", "Wikipedia Crystals/Materials API",
                  "https://en.wikipedia.org", approx_mb([""] * n),
                  "Crystallography, metallurgy, nanomaterials, semiconductors", n)
    log.info(f"  [APP-CRYST-1] Wiki API: {n}")

    n = wiki_stream(CRYSTAL_KW, bucket, "app_cryst_stream", max_docs=4000)
    total += n
    log.info(f"  [APP-CRYST-2] Wiki stream: {n}")

    n = arxiv_cats(["cond-mat.mtrl-sci", "cond-mat.supr-con",
                    "cond-mat.mes-hall", "physics.app-ph"],
                   bucket, "app_cryst_arxiv", max_docs=4000)
    total += n
    record_source("corpus_C_technical", "arXiv Materials Science Papers",
                  "https://arxiv.org", approx_mb([""] * n),
                  "Condensed matter: materials, superconductors, nanoscale", n)
    log.info(f"  [APP-CRYST-3] arXiv: {n}")

    n = openalex([
        "crystal structure prediction machine learning",
        "high entropy alloy mechanical properties",
        "2D materials graphene electronic properties",
        "perovskite solar cell efficiency stability",
        "superconductor high temperature mechanism",
        "biomaterial scaffold tissue engineering",
        "corrosion inhibitor metal protection",
        "nanomaterial synthesis characterization application",
        "semiconductor doping carrier concentration",
        "piezoelectric energy harvesting MEMS",
    ], bucket, "app_cryst_openalex", "Materials Science Research")
    total += n
    log.info(f"  [APP-CRYST-4] OpenAlex: {n}")

    log.info(f"  >>> Crystals, Metals & Materials total: {total:,}")
    return total


def load_magnets_applied_magnetism():
    log.info("=" * 65)
    log.info("APP-3: Magnets & Applied Magnetism")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "permanent magnet neodymium NdFeB properties",
        "ferromagnetism domains Weiss molecular field",
        "magnetic hysteresis B-H curve coercivity",
        "electromagnetic induction Faraday law Lenz",
        "electric motor torque speed winding design",
        "AC induction motor three-phase rotating field",
        "synchronous motor permanent magnet BLDC",
        "transformer magnetic core losses lamination",
        "MRI magnetic resonance imaging physics gradients",
        "superconducting magnet cryogenics field strength",
        "magnetic levitation maglev train bearings",
        "hard disk drive magnetic recording write head",
        "spintronics giant magnetoresistance GMR",
        "magnetic tunnel junction spin transfer torque",
        "Hall effect sensor current measurement",
        "magnetohydrodynamics MHD propulsion",
        "magnetic confinement fusion tokamak coils",
        "geomagnetic field Earth core dynamo",
        "magnetic nanoparticle hyperthermia cancer",
        "electromagnetic compatibility shielding EMC",
        "eddy current testing nondestructive",
        "magnetic compass navigation declination",
        "electromagnet solenoid design force calculation",
        "Helmholtz coil uniform magnetic field",
        "Maxwell equations electromagnetic wave",
    ]
    n = wiki_api(queries, bucket, "app_mag_wiki"); total += n
    record_source("corpus_C_technical", "Wikipedia Magnets/Electromagnetism API",
                  "https://en.wikipedia.org", approx_mb([""] * n),
                  "Electromagnetism, motors, MRI, spintronics, maglev, fusion", n)
    log.info(f"  [APP-MAG-1] Wiki API: {n}")

    n = wiki_stream(MAGNET_KW, bucket, "app_mag_stream", max_docs=3000)
    total += n
    log.info(f"  [APP-MAG-2] Wiki stream: {n}")

    n = arxiv_cats(["cond-mat.str-el", "cond-mat.supr-con",
                    "physics.app-ph", "eess.SY"],
                   bucket, "app_mag_arxiv",
                   extra_kw={"magnet", "spin", "ferromagnet", "electromagnet",
                             "mri", "motor", "maglev", "magnetic", "hall"},
                   max_docs=3000)
    total += n
    log.info(f"  [APP-MAG-3] arXiv: {n}")

    n = openalex([
        "permanent magnet motor efficiency design",
        "spintronics magnetic memory MRAM device",
        "MRI physics gradient coil image reconstruction",
        "magnetic nanoparticle biomedical hyperthermia",
        "maglev levitation control stability",
        "tokamak magnetic confinement plasma",
        "giant magnetoresistance sensor applications",
        "wireless power transfer magnetic coupling",
        "magnetic refrigeration magnetocaloric effect",
        "eddy current loss reduction laminated core",
    ], bucket, "app_mag_openalex", "Applied Magnetism Research")
    total += n
    log.info(f"  [APP-MAG-4] OpenAlex: {n}")

    log.info(f"  >>> Magnets & Applied Magnetism total: {total:,}")
    return total


def load_complex_circuit_systems():
    log.info("=" * 65)
    log.info("APP-4: Complex Circuit Systems — Design & Build")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "analog circuit design operational amplifier feedback",
        "filter design Butterworth Chebyshev Bessel",
        "oscillator circuit crystal LC RC design",
        "amplifier class A B AB D efficiency",
        "CMOS inverter logic gate propagation delay",
        "VLSI digital design standard cell place route",
        "FPGA architecture LUT programming synthesis",
        "PCB layout signal integrity ground plane",
        "power electronics switching MOSFET gate drive",
        "buck boost converter duty cycle regulation",
        "LLC resonant converter soft switching",
        "three-phase inverter PWM motor drive",
        "RF circuit impedance matching S-parameters",
        "microwave amplifier noise figure gain",
        "antenna design patch array beamforming",
        "ADC DAC design sigma-delta successive approximation",
        "phase-locked loop PLL frequency synthesis",
        "SPICE simulation circuit modeling",
        "timing analysis setup hold violation",
        "IC fabrication CMOS process photolithography",
        "power integrity decoupling capacitor PDN",
        "EMI suppression filter ferrite choke",
        "mixed-signal circuit design layout matching",
        "high-speed serial link SerDes equalization",
        "thermal management heat sink junction temperature",
        "semiconductor packaging flip chip BGA",
        "EDA tools Cadence Synopsys design flow",
        "functional safety IEC 61508 automotive ISO 26262",
        "memory design SRAM DRAM flash architecture",
        "SoC system on chip integration design",
    ]
    n = wiki_api(queries, bucket, "app_circ_wiki"); total += n
    record_source("corpus_C_technical", "Wikipedia Circuit Systems API",
                  "https://en.wikipedia.org", approx_mb([""] * n),
                  "Analog, digital, VLSI, FPGA, PCB, power electronics, RF", n)
    log.info(f"  [APP-CIRC-1] Wiki API: {n}")

    n = wiki_stream(CIRCUIT_KW, bucket, "app_circ_stream", max_docs=4000)
    total += n
    log.info(f"  [APP-CIRC-2] Wiki stream: {n}")

    n = arxiv_cats(["eess.SP", "eess.SY", "cs.ET", "physics.app-ph"],
                   bucket, "app_circ_arxiv",
                   extra_kw={"circuit", "vlsi", "fpga", "analog", "digital",
                             "amplifier", "power", "rf", "filter", "adc", "dac",
                             "cmos", "chip", "semiconductor", "pcb"},
                   max_docs=3000)
    total += n
    log.info(f"  [APP-CIRC-3] arXiv: {n}")

    n = openalex([
        "CMOS analog circuit noise power efficiency",
        "FPGA architecture reconfigurable computing",
        "power converter switching loss efficiency",
        "RF transceiver integrated circuit design",
        "PCB signal integrity differential pair",
        "neuromorphic circuit spiking neural network",
        "in-memory computing resistive memory",
        "millimeter wave 5G circuit beamforming",
        "ADC high speed low power sigma delta",
        "mixed signal layout mismatch matching",
    ], bucket, "app_circ_openalex", "Circuit Systems Research")
    total += n
    log.info(f"  [APP-CIRC-4] OpenAlex: {n}")

    log.info(f"  >>> Complex Circuit Systems total: {total:,}")
    return total


def load_lasers_photonics():
    log.info("=" * 65)
    log.info("APP-5: Lasers & Their Applications in Modern Technology")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "laser physics stimulated emission population inversion",
        "laser cavity resonator mode structure",
        "CO2 laser industrial cutting welding engraving",
        "Nd:YAG laser pulsed CW applications",
        "fiber laser high power telecommunications",
        "semiconductor diode laser applications",
        "excimer laser UV lithography eye surgery",
        "femtosecond ultrafast laser pulse compression",
        "Ti:Sapphire laser tunable ultrafast",
        "laser surgery LASIK eye refractive correction",
        "photodynamic therapy laser cancer treatment",
        "LiDAR laser ranging autonomous vehicle mapping",
        "laser cutting manufacturing precision material",
        "laser welding automotive aerospace metals",
        "laser marking engraving industrial",
        "laser spectroscopy absorption emission analysis",
        "Raman spectroscopy laser molecular fingerprint",
        "optical fiber communications wavelength DWDM",
        "optical amplifier EDFA gain bandwidth",
        "holography laser wavefront reconstruction",
        "laser interferometry gravitational wave LIGO",
        "optical coherence tomography OCT medical imaging",
        "photonic integrated circuit silicon photonics",
        "nonlinear optics second harmonic generation",
        "optical tweezers laser trapping manipulation",
        "laser cooling magneto-optical trap BEC",
        "laser printing photocopier xerography",
        "directed energy weapon high power laser",
        "laser isotope separation uranium enrichment",
        "free electron laser synchrotron radiation",
    ]
    n = wiki_api(queries, bucket, "app_laser_wiki"); total += n
    record_source("corpus_C_technical", "Wikipedia Lasers & Photonics API",
                  "https://en.wikipedia.org", approx_mb([""] * n),
                  "Laser physics, types, surgery, LiDAR, fiber optics, photonics", n)
    log.info(f"  [APP-LASER-1] Wiki API: {n}")

    n = wiki_stream(LASER_KW, bucket, "app_laser_stream", max_docs=4000)
    total += n
    log.info(f"  [APP-LASER-2] Wiki stream: {n}")

    n = arxiv_cats(["physics.optics", "quant-ph", "eess.SP"],
                   bucket, "app_laser_arxiv",
                   extra_kw={"laser", "photon", "optical", "fiber", "lidar",
                             "nonlinear optics", "coherent", "ultrafast", "spectroscopy"},
                   max_docs=4000)
    total += n
    record_source("corpus_C_technical", "arXiv Optics & Photonics Papers",
                  "https://arxiv.org", approx_mb([""] * n),
                  "Laser physics, nonlinear optics, photonics, quantum optics", n)
    log.info(f"  [APP-LASER-3] arXiv: {n}")

    n = openalex([
        "ultrafast laser femtosecond pulse material processing",
        "LiDAR autonomous vehicle point cloud",
        "silicon photonics integrated circuit chip",
        "optical coherence tomography retinal imaging",
        "fiber laser high power beam quality",
        "laser cooling ultracold atoms Bose Einstein",
        "nonlinear photonics frequency comb generation",
        "directed energy laser weapon atmospheric",
        "Raman spectroscopy in vivo tissue diagnosis",
        "photonic quantum computing optical qubit",
    ], bucket, "app_laser_openalex", "Laser & Photonics Research")
    total += n
    log.info(f"  [APP-LASER-4] OpenAlex: {n}")

    log.info(f"  >>> Lasers & Photonics total: {total:,}")
    return total


def load_propulsion_mechanics():
    log.info("=" * 65)
    log.info("APP-6: Propulsion Mechanics")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "rocket propulsion specific impulse thrust equation",
        "chemical rocket solid liquid propellant combustion",
        "rocket nozzle de Laval expansion exhaust velocity",
        "ion thruster electric propulsion xenon",
        "Hall effect thruster plasma propulsion",
        "nuclear thermal rocket NTR hydrogen propellant",
        "solar sail photon pressure propulsion",
        "jet engine turbojet thermodynamic Brayton cycle",
        "turbofan bypass ratio fuel efficiency",
        "turboprop propeller shaft power aircraft",
        "ramjet supersonic combustion scramjet",
        "pulse detonation engine detonation wave",
        "aerospike nozzle altitude compensating",
        "rocket staging multistage delta-v Tsiolkovsky",
        "orbital mechanics Hohmann transfer",
        "spacecraft attitude control reaction wheel",
        "reentry vehicle ablative heat shield",
        "hypersonic vehicle thermal protection",
        "propeller blade aerodynamics pitch advance ratio",
        "marine propulsion diesel screw efficiency",
        "underwater propulsion submarine pump jet",
        "helicopter rotor aerodynamics lift hover",
        "aircraft turbine combustion chamber design",
        "solid rocket motor grain geometry burning rate",
        "bipropellant engine liquid oxygen kerosene",
        "methane rocket engine Raptor Merlin",
        "electric aircraft propulsion distributed",
        "VASIMR variable specific impulse magnetoplasma",
        "launch vehicle payload performance trade",
        "combustion instability rocket chamber",
    ]
    n = wiki_api(queries, bucket, "app_prop_wiki"); total += n
    record_source("corpus_C_technical", "Wikipedia Propulsion Mechanics API",
                  "https://en.wikipedia.org", approx_mb([""] * n),
                  "Rocket, jet, ion, nuclear, ramjet, orbital propulsion systems", n)
    log.info(f"  [APP-PROP-1] Wiki API: {n}")

    n = wiki_stream(PROPULSION_KW, bucket, "app_prop_stream", max_docs=4000)
    total += n
    log.info(f"  [APP-PROP-2] Wiki stream: {n}")

    n = arxiv_cats(["physics.flu-dyn", "physics.app-ph", "eess.SY"],
                   bucket, "app_prop_arxiv",
                   extra_kw={"rocket", "propulsion", "thrust", "nozzle", "turbine",
                             "jet", "combustion", "hypersonic", "ion", "plasma",
                             "propeller", "spacecraft", "launch"},
                   max_docs=3000)
    total += n
    log.info(f"  [APP-PROP-3] arXiv: {n}")

    n = gutenberg([
        (19510, "Rocket Propulsion Elements — Sutton (public domain excerpts)"),
        (4352,  "The Aeroplane Speaks — H. Barber (early aviation)"),
    ], bucket, "app_prop_gutenberg")
    total += n
    log.info(f"  [APP-PROP-4] Gutenberg: {n}")

    n = openalex([
        "ion thruster xenon Hall effect specific impulse",
        "hypersonic vehicle scramjet combustion instability",
        "solid rocket propellant burn rate additives",
        "turbofan engine performance efficiency blade cooling",
        "electric aircraft propulsion motor efficiency",
        "rocket nozzle thrust optimization CFD",
        "orbital mechanics low thrust trajectory optimization",
        "nuclear thermal propulsion reactor core design",
        "rotating detonation engine propulsion",
        "propeller cavitation noise underwater",
    ], bucket, "app_prop_openalex", "Propulsion Research")
    total += n
    log.info(f"  [APP-PROP-5] OpenAlex: {n}")

    log.info(f"  >>> Propulsion Mechanics total: {total:,}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("TitanAI Applied Sciences Corpus Loader — Starting")
    log.info("6 Subfields: Herbal Medicine | Crystals/Metals | Magnets")
    log.info("             Circuit Systems | Lasers | Propulsion")
    log.info(f"Target directory: {RAW}")
    log.info("")

    results = {}
    results["herbal_medicine"]       = load_apothecary_herbal_medicine()
    results["crystals_metals"]       = load_crystals_metals_materials()
    results["magnets_applied"]       = load_magnets_applied_magnetism()
    results["circuit_systems"]       = load_complex_circuit_systems()
    results["lasers_photonics"]      = load_lasers_photonics()
    results["propulsion_mechanics"]  = load_propulsion_mechanics()

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
