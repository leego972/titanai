"""
TitanAI Engineering Corpus Loader
===================================
Adds three new knowledge domains to corpus_C_technical:

  1. Electrical Engineering  — circuits, electronics, power systems, RF,
       semiconductors, signal processing, control systems, PCB design,
       microelectronics, electromagnetic theory  (~15,000 docs)

  2. Mechanics  — classical mechanics, fluid dynamics, thermodynamics,
       materials science, structural engineering, vibrations, statics,
       dynamics, continuum mechanics, tribology  (~15,000 docs)

  3. Robotics & Nano-Mechanics  — robot kinematics/dynamics, control,
       computer vision for robotics, MEMS, nanomechanics, molecular
       machines, soft robotics, autonomous systems  (~10,000 docs)

All docs are appended to corpus_C_technical so that the existing
sharding pipeline (generate_shards.py) picks them up without any
ratio configuration changes.

Sources used:
  - Wikipedia (wikimedia/wikipedia) — keyword-filtered by domain
  - StackExchange Q&A (HuggingFaceH4/stack-exchange-preferences)
    filtered by community: electronics, physics, robotics
  - arXiv abstracts (ArXiv dataset, eess / cs.RO / cond-mat.mes-hall)
  - Project Gutenberg open engineering textbooks
  - NIST / IEC open technical documentation (scraped summaries)

Run:
  pip install datasets requests tqdm
  python scripts/load_corpus_engineering.py
"""

import os
import sys
import json
import time
import logging
import hashlib
import requests
from pathlib import Path
from datetime import datetime

# ── Setup ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
RAW  = BASE / "data" / "raw"
LOG  = BASE / "data" / "corpus_engineering.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(str(LOG))]
)
log = logging.getLogger("engineering_loader")

INVENTORY     = BASE / "data" / "source_inventory.json"
EXCLUSION_LOG = BASE / "data" / "exclusions_engineering.jsonl"

try:
    with open(INVENTORY) as f:
        inventory = json.load(f)
except Exception:
    inventory = {}
exclusions = []

# ── Helpers ────────────────────────────────────────────────────────────────────
def record_source(bucket, source_name, source_url, size_mb, reason, n_docs):
    inventory.setdefault(bucket, []).append({
        "source": source_name,
        "url": source_url,
        "size_mb": round(size_mb, 2),
        "n_documents": n_docs,
        "reason_for_inclusion": reason,
        "loaded_at": datetime.utcnow().isoformat(),
    })

def record_exclusion(source_name, reason):
    exclusions.append({"source": source_name, "reason": reason,
                        "timestamp": datetime.utcnow().isoformat()})
    log.warning(f"EXCLUDED: {source_name} — {reason}")

def write_docs(bucket_dir: Path, docs: list, source_tag: str) -> int:
    bucket_dir.mkdir(parents=True, exist_ok=True)
    existing = list(bucket_dir.glob(f"{source_tag}_*.txt"))
    start_idx = len(existing)
    written = 0
    for i, text in enumerate(docs):
        if not text or len(text.strip()) < 150:
            continue
        fname = bucket_dir / f"{source_tag}_{start_idx + i:06d}.txt"
        fname.write_text(text.strip(), encoding="utf-8")
        written += 1
    log.info(f"  Wrote {written} docs → {bucket_dir.name}/{source_tag}_*.txt")
    return written

def approx_mb(docs):
    return sum(len(d.encode("utf-8")) for d in docs) / 1_048_576

def safe_get(item, *keys, default=""):
    for k in keys:
        if isinstance(item, dict):
            item = item.get(k, default)
        else:
            return default
    return item or default

# ── Keyword sets ───────────────────────────────────────────────────────────────
EE_KEYWORDS = {
    "electrical engineering", "electronic", "circuit", "transistor", "capacitor",
    "resistor", "inductor", "diode", "semiconductor", "integrated circuit",
    "microprocessor", "microcontroller", "amplifier", "oscillator", "filter",
    "signal processing", "digital signal", "analog signal", "power electronics",
    "voltage", "current", "impedance", "frequency", "bandwidth", "antenna",
    "radio frequency", "RF engineering", "electromagnetic", "maxwell", "faraday",
    "ohm's law", "kirchhoff", "fourier transform", "laplace transform",
    "control system", "feedback", "pid controller", "transfer function",
    "operational amplifier", "op-amp", "MOSFET", "BJT", "CMOS", "VLSI",
    "PCB", "printed circuit board", "soldering", "multimeter", "oscilloscope",
    "power supply", "transformer", "motor drive", "inverter", "rectifier",
    "photovoltaic", "battery management", "electric vehicle", "smart grid",
    "FPGA", "ASIC", "digital logic", "boolean algebra", "flip flop",
    "microelectronics", "nanotechnology chip", "sensor", "actuator",
    "embedded system", "arduino", "raspberry pi", "IoT",
}

MECHANICS_KEYWORDS = {
    "classical mechanics", "newtonian mechanics", "statics", "dynamics",
    "kinematics", "kinetics", "force", "torque", "momentum", "angular momentum",
    "energy conservation", "work-energy theorem", "friction", "tension",
    "compression", "shear", "bending moment", "structural analysis",
    "finite element", "stress", "strain", "elasticity", "plasticity",
    "material science", "materials science", "fatigue", "fracture mechanics",
    "fluid mechanics", "fluid dynamics", "bernoulli", "navier-stokes",
    "turbulence", "laminar flow", "viscosity", "hydraulics", "pneumatics",
    "thermodynamics", "heat transfer", "conduction", "convection", "radiation",
    "carnot cycle", "entropy", "enthalpy", "thermodynamic cycle",
    "vibrations", "oscillation", "resonance", "damping", "modal analysis",
    "tribology", "wear", "lubrication", "contact mechanics",
    "continuum mechanics", "solid mechanics", "deformation",
    "machine design", "gear", "bearing", "shaft", "coupling",
    "manufacturing process", "CNC", "machining", "casting", "welding",
    "mechanical engineering", "engineering mechanics",
}

ROBOTICS_NANO_KEYWORDS = {
    "robotics", "robot", "robotic arm", "manipulator", "end effector",
    "degrees of freedom", "forward kinematics", "inverse kinematics",
    "jacobian", "trajectory planning", "path planning", "motion planning",
    "robot dynamics", "robot control", "PID robot", "servo motor",
    "autonomous robot", "mobile robot", "humanoid robot", "collaborative robot",
    "cobot", "SLAM", "simultaneous localization", "ROS", "robot operating system",
    "computer vision robot", "lidar", "point cloud", "sensor fusion",
    "soft robotics", "bio-inspired robot", "swarm robotics",
    "drone", "UAV", "quadrotor", "autonomous vehicle",
    "MEMS", "micro-electro-mechanical", "microactuator", "microsensor",
    "nanomechanics", "nanomachine", "molecular motor", "nanoscale",
    "atomic force microscopy", "AFM", "scanning tunneling microscope",
    "nanotribology", "nanoindentation", "quantum mechanics engineering",
    "micro robot", "nano robot", "DNA origami", "molecular machine",
    "piezoelectric", "shape memory alloy", "smart material",
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ELECTRICAL ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def load_electrical_engineering():
    log.info("=" * 60)
    log.info("SECTION 1: Electrical Engineering")
    log.info("=" * 60)
    bucket = RAW / "corpus_C_technical"
    total = 0

    # ── EE-1: Wikipedia — Electrical Engineering articles ─────────────────────
    try:
        from datasets import load_dataset
        log.info("[EE-1] Wikipedia EE articles (15,000 candidates)...")
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en",
                            split="train", streaming=True)
        docs = []
        scanned = 0
        for item in wiki:
            scanned += 1
            if scanned > 600_000:
                break
            title = item.get("title", "").lower()
            text  = item.get("text", "").strip()
            if len(text) < 300:
                continue
            if any(kw in title or kw in text[:500].lower() for kw in EE_KEYWORDS):
                # Format with clear title header and full article
                docs.append(f"# {item['title']}\n\n{text[:8000]}")
            if len(docs) >= 7000:
                break
        n = write_docs(bucket, docs, "ee_wikipedia")
        record_source("corpus_C_technical", "Wikipedia EE Articles",
                      "https://huggingface.co/datasets/wikimedia/wikipedia",
                      approx_mb(docs),
                      "Electrical engineering Wikipedia articles: circuits, semiconductors, "
                      "power electronics, RF, signal processing, control systems", n)
        total += n
        log.info(f"  [EE-1] Done: {n} docs from {scanned} scanned")
    except Exception as e:
        log.error(f"  [EE-1] Failed: {e}")
        record_exclusion("Wikipedia EE Articles", str(e))

    # ── EE-2: Electronics StackExchange Q&A ───────────────────────────────────
    try:
        from datasets import load_dataset
        log.info("[EE-2] Electronics StackExchange Q&A...")
        se = load_dataset("HuggingFaceH4/stack-exchange-preferences",
                          split="train", streaming=True)
        docs = []
        for i, item in enumerate(se):
            if i > 2_000_000:
                break
            domain = safe_get(item, "domain", default="")
            if "electronics" not in domain.lower():
                continue
            question = safe_get(item, "question", default="")
            answers  = item.get("answers", []) or []
            if not question:
                continue
            best_ans = ""
            best_score = -999
            for ans in answers:
                score = ans.get("pm_score", 0) or 0
                if score > best_score:
                    best_score = score
                    best_ans = ans.get("text", "")
            if best_ans and len(question) > 50:
                docs.append(
                    f"Electronics Q&A\n\nQuestion: {question.strip()}\n\n"
                    f"Answer (score {best_score}): {best_ans.strip()[:3000]}"
                )
            if len(docs) >= 4000:
                break
        n = write_docs(bucket, docs, "ee_stackexchange")
        record_source("corpus_C_technical", "Electronics StackExchange",
                      "https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences",
                      approx_mb(docs),
                      "Real-world electronics Q&A: circuit design, component selection, "
                      "debugging, PCB layout, power supply design", n)
        total += n
        log.info(f"  [EE-2] Done: {n} electronics Q&A docs")
    except Exception as e:
        log.error(f"  [EE-2] Failed: {e}")
        record_exclusion("Electronics StackExchange", str(e))

    # ── EE-3: arXiv EESS (Electrical Engineering & Signal Science) ─────────────
    try:
        from datasets import load_dataset
        log.info("[EE-3] arXiv EESS abstracts...")
        # Use ArXiv metadata dataset
        arxiv = load_dataset("Cornell-University/arxiv", split="train", streaming=True)
        eess_cats = {"eess.SP", "eess.PE", "eess.SY", "eess.IV",
                     "cs.SY", "cs.ET", "physics.app-ph"}
        docs = []
        for i, item in enumerate(arxiv):
            if i > 2_000_000:
                break
            cats = set((item.get("categories", "") or "").split())
            if not cats.intersection(eess_cats):
                continue
            title    = (item.get("title", "") or "").strip().replace("\n", " ")
            abstract = (item.get("abstract", "") or "").strip().replace("\n", " ")
            authors  = (item.get("authors", "") or "").strip()
            journal  = (item.get("journal-ref", "") or "").strip()
            if len(abstract) < 100:
                continue
            doc = f"Title: {title}\nAuthors: {authors}\n"
            if journal:
                doc += f"Published in: {journal}\n"
            doc += f"Categories: {' '.join(cats)}\n\nAbstract:\n{abstract}"
            docs.append(doc)
            if len(docs) >= 4000:
                break
        n = write_docs(bucket, docs, "ee_arxiv")
        record_source("corpus_C_technical", "arXiv EESS Papers",
                      "https://huggingface.co/datasets/Cornell-University/arxiv",
                      approx_mb(docs),
                      "Electrical engineering research: signal processing, power electronics, "
                      "control systems, image/video processing", n)
        total += n
        log.info(f"  [EE-3] Done: {n} arXiv EESS docs")
    except Exception as e:
        log.error(f"  [EE-3] Failed: {e}")
        record_exclusion("arXiv EESS", str(e))

    # ── EE-4: Textbooks — Project Gutenberg Engineering Books ─────────────────
    try:
        log.info("[EE-4] Gutenberg engineering textbooks...")
        # Gutenberg IDs for open electrical engineering / physics textbooks
        gutenberg_ee_ids = [
            # Electricity & magnetism, applied physics, engineering texts
            9116,   # Hawkins Electrical Guide
            14062,  # Principles of Electrical Engineering
            30429,  # The Story of Electricity
            22380,  # Electricity for Boys
            29765,  # Electric Lighting
            24264,  # A History of Electric Telegraphy
            17799,  # The Wireless Operator
            28321,  # Radio Instruments and Measurements (NBS)
            10745,  # Physics (Household Science & Art)
            55148,  # Experimental Researches in Electricity (Faraday)
            9181,   # On the Various Forces of Nature (Faraday)
            33642,  # Matter and Motion (Maxwell)
            38769,  # A Treatise on Electricity and Magnetism Vol 1 (Maxwell)
        ]
        docs = []
        for gid in gutenberg_ee_ids:
            try:
                for mirror in [
                    f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
                    f"https://gutenberg.org/files/{gid}/{gid}-0.txt",
                    f"https://gutenberg.org/files/{gid}/{gid}.txt",
                ]:
                    r = requests.get(mirror, timeout=20)
                    if r.status_code == 200 and len(r.text) > 1000:
                        text = r.text
                        # Remove Gutenberg header/footer
                        for marker in ["*** START OF", "***START OF"]:
                            if marker in text:
                                text = text[text.index(marker):]
                                text = text[text.index("\n")+1:]
                                break
                        for marker in ["*** END OF", "***END OF", "End of Project Gutenberg"]:
                            if marker in text:
                                text = text[:text.index(marker)]
                        # Split into ~2000-word chunks
                        words = text.split()
                        for chunk_start in range(0, len(words), 1500):
                            chunk = " ".join(words[chunk_start:chunk_start+1500])
                            if len(chunk) > 300:
                                docs.append(f"[Electrical Engineering Textbook — Gutenberg #{gid}]\n\n{chunk}")
                        log.info(f"    Gutenberg #{gid}: OK ({len(words)} words)")
                        break
                time.sleep(0.5)
            except Exception as ge:
                log.warning(f"    Gutenberg #{gid}: {ge}")
        n = write_docs(bucket, docs, "ee_gutenberg")
        record_source("corpus_C_technical", "Gutenberg EE Textbooks",
                      "https://www.gutenberg.org",
                      approx_mb(docs),
                      "Classic electrical engineering textbooks: Faraday, Maxwell, Hawkins "
                      "Electrical Guide, NBS Radio Instruments, wireless telegraphy", n)
        total += n
        log.info(f"  [EE-4] Done: {n} textbook chunks")
    except Exception as e:
        log.error(f"  [EE-4] Failed: {e}")
        record_exclusion("Gutenberg EE Textbooks", str(e))

    # ── EE-5: NIST Electrical Standards (public API) ──────────────────────────
    try:
        log.info("[EE-5] NIST SP 800-series & technical notes (EE-relevant)...")
        # NIST publications API - filter for EE-related content
        nist_resp = requests.get(
            "https://csrc.nist.gov/CSRC/media/Publications/sp/800/53/r5/final/documents/sp800-53r5-control-catalog.json",
            timeout=30
        )
        docs = []
        if nist_resp.status_code == 200:
            # Extract EE-relevant control descriptions
            data = nist_resp.json()
            for control in (data.get("controls", []) or []):
                title = control.get("title", "")
                desc  = control.get("description", "")
                if desc and len(desc) > 100:
                    docs.append(f"NIST Standard: {title}\n\n{desc}")
        # Also fetch some IEEE-style open content via DOE OSTI
        osti_resp = requests.get(
            "https://www.osti.gov/api/v1/records?q=electrical+engineering&page=0&size=100",
            timeout=30
        )
        if osti_resp.status_code == 200:
            osti = osti_resp.json()
            for rec in (osti.get("records", []) or []):
                title    = rec.get("title", "").strip()
                abstract = rec.get("description", "").strip()
                journal  = rec.get("journal_name", "")
                if abstract and len(abstract) > 100:
                    docs.append(
                        f"DOE/OSTI Technical Report\n\nTitle: {title}\n"
                        f"Source: {journal}\n\n{abstract}"
                    )
        n = write_docs(bucket, docs, "ee_nist_osti")
        if n > 0:
            record_source("corpus_C_technical", "NIST & OSTI EE Documents",
                          "https://www.nist.gov / https://www.osti.gov",
                          approx_mb(docs),
                          "US government EE technical documents: NIST standards, DOE "
                          "research reports on electrical engineering topics", n)
        total += n
        log.info(f"  [EE-5] Done: {n} docs")
    except Exception as e:
        log.error(f"  [EE-5] Failed (non-critical): {e}")

    log.info(f"\n  >>> Electrical Engineering total: {total} docs")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MECHANICS
# ══════════════════════════════════════════════════════════════════════════════

def load_mechanics():
    log.info("=" * 60)
    log.info("SECTION 2: Mechanics")
    log.info("=" * 60)
    bucket = RAW / "corpus_C_technical"
    total = 0

    # ── MECH-1: Wikipedia — Mechanics articles ─────────────────────────────────
    try:
        from datasets import load_dataset
        log.info("[MECH-1] Wikipedia mechanics articles...")
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en",
                            split="train", streaming=True)
        docs = []
        scanned = 0
        for item in wiki:
            scanned += 1
            if scanned > 700_000:
                break
            title = item.get("title", "").lower()
            text  = item.get("text", "").strip()
            if len(text) < 300:
                continue
            if any(kw in title or kw in text[:600].lower()
                   for kw in MECHANICS_KEYWORDS):
                docs.append(f"# {item['title']}\n\n{text[:8000]}")
            if len(docs) >= 7000:
                break
        n = write_docs(bucket, docs, "mech_wikipedia")
        record_source("corpus_C_technical", "Wikipedia Mechanics Articles",
                      "https://huggingface.co/datasets/wikimedia/wikipedia",
                      approx_mb(docs),
                      "Mechanics Wikipedia: classical mechanics, fluid dynamics, "
                      "thermodynamics, materials science, structural engineering", n)
        total += n
        log.info(f"  [MECH-1] Done: {n} docs from {scanned} scanned")
    except Exception as e:
        log.error(f"  [MECH-1] Failed: {e}")
        record_exclusion("Wikipedia Mechanics", str(e))

    # ── MECH-2: Physics StackExchange — mechanics subset ──────────────────────
    try:
        from datasets import load_dataset
        log.info("[MECH-2] Physics StackExchange Q&A (mechanics)...")
        se = load_dataset("HuggingFaceH4/stack-exchange-preferences",
                          split="train", streaming=True)
        docs = []
        for i, item in enumerate(se):
            if i > 2_000_000:
                break
            domain = safe_get(item, "domain", default="")
            if "physics" not in domain.lower():
                continue
            question = safe_get(item, "question", default="")
            q_lower  = question.lower()
            if not any(kw in q_lower for kw in MECHANICS_KEYWORDS):
                continue
            answers = item.get("answers", []) or []
            best_ans, best_score = "", -999
            for ans in answers:
                score = ans.get("pm_score", 0) or 0
                if score > best_score:
                    best_score = score
                    best_ans = ans.get("text", "")
            if best_ans and len(question) > 50:
                docs.append(
                    f"Mechanics Q&A (Physics StackExchange)\n\n"
                    f"Question: {question.strip()}\n\n"
                    f"Answer (score {best_score}): {best_ans.strip()[:3000]}"
                )
            if len(docs) >= 4000:
                break
        n = write_docs(bucket, docs, "mech_stackexchange")
        record_source("corpus_C_technical", "Physics StackExchange Mechanics",
                      "https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences",
                      approx_mb(docs),
                      "Real mechanics Q&A: statics, dynamics, thermodynamics, fluid "
                      "mechanics, material properties, vibrations", n)
        total += n
        log.info(f"  [MECH-2] Done: {n} docs")
    except Exception as e:
        log.error(f"  [MECH-2] Failed: {e}")
        record_exclusion("Physics SE Mechanics", str(e))

    # ── MECH-3: arXiv Mechanical Engineering ──────────────────────────────────
    try:
        from datasets import load_dataset
        log.info("[MECH-3] arXiv mechanics / thermal / fluid abstracts...")
        arxiv = load_dataset("Cornell-University/arxiv", split="train", streaming=True)
        mech_cats = {
            "physics.flu-dyn",   # Fluid dynamics
            "physics.class-ph",  # Classical physics
            "cond-mat.mtrl-sci", # Materials science
            "cond-mat.soft",     # Soft matter
            "physics.app-ph",    # Applied physics
            "cs.CE",             # Computational engineering
        }
        docs = []
        for i, item in enumerate(arxiv):
            if i > 2_000_000:
                break
            cats = set((item.get("categories", "") or "").split())
            if not cats.intersection(mech_cats):
                continue
            title    = (item.get("title", "") or "").replace("\n", " ").strip()
            abstract = (item.get("abstract", "") or "").replace("\n", " ").strip()
            if len(abstract) < 100:
                continue
            docs.append(
                f"Title: {title}\nCategories: {' '.join(cats)}\n\n"
                f"Abstract:\n{abstract}"
            )
            if len(docs) >= 4000:
                break
        n = write_docs(bucket, docs, "mech_arxiv")
        record_source("corpus_C_technical", "arXiv Mechanics Papers",
                      "https://huggingface.co/datasets/Cornell-University/arxiv",
                      approx_mb(docs),
                      "Mechanics research papers: fluid dynamics, classical physics, "
                      "materials science, soft matter, computational engineering", n)
        total += n
        log.info(f"  [MECH-3] Done: {n} docs")
    except Exception as e:
        log.error(f"  [MECH-3] Failed: {e}")
        record_exclusion("arXiv Mechanics", str(e))

    # ── MECH-4: Gutenberg Mechanics Textbooks ─────────────────────────────────
    try:
        log.info("[MECH-4] Gutenberg mechanics & engineering textbooks...")
        gutenberg_mech_ids = [
            # Classical mechanics, thermodynamics, engineering textbooks
            5001,   # Principia Mathematica (Newton)
            28233,  # Mechanics (Lagrange / Mach)
            14725,  # The Mechanical Properties of Fluids
            33272,  # The Steam Engine
            25864,  # Hydraulics and Fluid Mechanics
            20417,  # Practical Mechanics for Boys
            41568,  # Heat Engines
            38876,  # Thermodynamics (Bryan)
            40780,  # Materials of Construction
            26839,  # The Theory of Heat (Maxwell)
            1202,   # The Law of Thermodynamics
            4367,   # Engineering Descriptive Geometry
            20218,  # Machine Design: Kinematics of Machinery
        ]
        docs = []
        for gid in gutenberg_mech_ids:
            try:
                for mirror in [
                    f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
                    f"https://gutenberg.org/files/{gid}/{gid}-0.txt",
                    f"https://gutenberg.org/files/{gid}/{gid}.txt",
                ]:
                    r = requests.get(mirror, timeout=20)
                    if r.status_code == 200 and len(r.text) > 1000:
                        text = r.text
                        for marker in ["*** START OF", "***START OF"]:
                            if marker in text:
                                text = text[text.index(marker):]
                                text = text[text.index("\n")+1:]
                                break
                        for marker in ["*** END OF", "***END OF", "End of Project Gutenberg"]:
                            if marker in text:
                                text = text[:text.index(marker)]
                        words = text.split()
                        for chunk_start in range(0, len(words), 1500):
                            chunk = " ".join(words[chunk_start:chunk_start+1500])
                            if len(chunk) > 300:
                                docs.append(
                                    f"[Mechanics Textbook — Gutenberg #{gid}]\n\n{chunk}"
                                )
                        log.info(f"    Gutenberg #{gid}: OK ({len(words)} words)")
                        break
                time.sleep(0.5)
            except Exception as ge:
                log.warning(f"    Gutenberg #{gid}: {ge}")
        n = write_docs(bucket, docs, "mech_gutenberg")
        record_source("corpus_C_technical", "Gutenberg Mechanics Textbooks",
                      "https://www.gutenberg.org",
                      approx_mb(docs),
                      "Classic mechanics texts: Newton Principia, Lagrange mechanics, "
                      "Maxwell heat theory, thermodynamics, steam engines, hydraulics", n)
        total += n
        log.info(f"  [MECH-4] Done: {n} textbook chunks")
    except Exception as e:
        log.error(f"  [MECH-4] Failed: {e}")
        record_exclusion("Gutenberg Mechanics", str(e))

    log.info(f"\n  >>> Mechanics total: {total} docs")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ROBOTICS & NANO-MECHANICS
# ══════════════════════════════════════════════════════════════════════════════

def load_robotics_nanomechanics():
    log.info("=" * 60)
    log.info("SECTION 3: Robotics & Nano-Mechanics")
    log.info("=" * 60)
    bucket = RAW / "corpus_C_technical"
    total = 0

    # ── ROBO-1: Wikipedia — Robotics & Nanotechnology articles ────────────────
    try:
        from datasets import load_dataset
        log.info("[ROBO-1] Wikipedia robotics & nano articles...")
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en",
                            split="train", streaming=True)
        docs = []
        scanned = 0
        for item in wiki:
            scanned += 1
            if scanned > 500_000:
                break
            title = item.get("title", "").lower()
            text  = item.get("text", "").strip()
            if len(text) < 300:
                continue
            if any(kw in title or kw in text[:600].lower()
                   for kw in ROBOTICS_NANO_KEYWORDS):
                docs.append(f"# {item['title']}\n\n{text[:8000]}")
            if len(docs) >= 5000:
                break
        n = write_docs(bucket, docs, "robo_wikipedia")
        record_source("corpus_C_technical", "Wikipedia Robotics & Nano Articles",
                      "https://huggingface.co/datasets/wikimedia/wikipedia",
                      approx_mb(docs),
                      "Robotics & nanotechnology Wikipedia: robot kinematics, SLAM, "
                      "ROS, MEMS, nanomechanics, molecular machines, soft robotics", n)
        total += n
        log.info(f"  [ROBO-1] Done: {n} docs from {scanned} scanned")
    except Exception as e:
        log.error(f"  [ROBO-1] Failed: {e}")
        record_exclusion("Wikipedia Robotics/Nano", str(e))

    # ── ROBO-2: Robotics StackExchange Q&A ────────────────────────────────────
    try:
        from datasets import load_dataset
        log.info("[ROBO-2] Robotics StackExchange Q&A...")
        se = load_dataset("HuggingFaceH4/stack-exchange-preferences",
                          split="train", streaming=True)
        docs = []
        for i, item in enumerate(se):
            if i > 2_000_000:
                break
            domain = safe_get(item, "domain", default="")
            if "robotics" not in domain.lower():
                continue
            question = safe_get(item, "question", default="")
            answers  = item.get("answers", []) or []
            best_ans, best_score = "", -999
            for ans in answers:
                score = ans.get("pm_score", 0) or 0
                if score > best_score:
                    best_score = score
                    best_ans = ans.get("text", "")
            if best_ans and len(question) > 50:
                docs.append(
                    f"Robotics Q&A (StackExchange)\n\n"
                    f"Question: {question.strip()}\n\n"
                    f"Answer (score {best_score}): {best_ans.strip()[:3000]}"
                )
            if len(docs) >= 3000:
                break
        n = write_docs(bucket, docs, "robo_stackexchange")
        record_source("corpus_C_technical", "Robotics StackExchange",
                      "https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences",
                      approx_mb(docs),
                      "Real robotics Q&A: kinematics, ROS, motion planning, sensor "
                      "fusion, SLAM, actuator selection, robot control", n)
        total += n
        log.info(f"  [ROBO-2] Done: {n} docs")
    except Exception as e:
        log.error(f"  [ROBO-2] Failed: {e}")
        record_exclusion("Robotics StackExchange", str(e))

    # ── ROBO-3: arXiv Robotics (cs.RO) ────────────────────────────────────────
    try:
        from datasets import load_dataset
        log.info("[ROBO-3] arXiv robotics & nanoscience abstracts...")
        arxiv = load_dataset("Cornell-University/arxiv", split="train", streaming=True)
        robo_cats = {
            "cs.RO",             # Robotics
            "cs.AI",             # AI (robotics subset)
            "cond-mat.mes-hall", # Nanomechanics / mesoscopic systems
            "cond-mat.soft",     # Soft matter / bio-inspired
            "physics.bio-ph",    # Biophysics (molecular motors)
            "cs.CV",             # Computer vision (robot perception)
        }
        docs = []
        for i, item in enumerate(arxiv):
            if i > 3_000_000:
                break
            cats = set((item.get("categories", "") or "").split())
            if not cats.intersection(robo_cats):
                continue
            title    = (item.get("title", "") or "").replace("\n", " ").strip()
            abstract = (item.get("abstract", "") or "").replace("\n", " ").strip()
            if len(abstract) < 100:
                continue
            # Only keep robotics-relevant ones from broad AI/CV categories
            if "cs.AI" in cats or "cs.CV" in cats:
                if not any(kw in title.lower() or kw in abstract.lower()
                           for kw in ROBOTICS_NANO_KEYWORDS):
                    continue
            docs.append(
                f"Title: {title}\nCategories: {' '.join(cats)}\n\n"
                f"Abstract:\n{abstract}"
            )
            if len(docs) >= 4000:
                break
        n = write_docs(bucket, docs, "robo_arxiv")
        record_source("corpus_C_technical", "arXiv Robotics & Nano Papers",
                      "https://huggingface.co/datasets/Cornell-University/arxiv",
                      approx_mb(docs),
                      "Robotics & nanomechanics research: cs.RO, mesoscopic physics, "
                      "soft matter, biophysics, molecular motors, computer vision", n)
        total += n
        log.info(f"  [ROBO-3] Done: {n} docs")
    except Exception as e:
        log.error(f"  [ROBO-3] Failed: {e}")
        record_exclusion("arXiv Robotics/Nano", str(e))

    # ── ROBO-4: Gutenberg — Automation, Machine Intelligence, Cybernetics ──────
    try:
        log.info("[ROBO-4] Gutenberg automation & cybernetics texts...")
        gutenberg_robo_ids = [
            21352,  # The Principles of Scientific Management (Taylor — automation)
            25453,  # Automatic and Manual Control (systems theory)
            61696,  # Cybernetics & Society (Wiener)
            28054,  # The Human Use of Human Beings (Wiener)
            20417,  # Practical Mechanics for Boys (mechanism & machines)
            6268,   # Wheels and Wheeling (mechanical systems)
        ]
        docs = []
        for gid in gutenberg_robo_ids:
            try:
                for mirror in [
                    f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
                    f"https://gutenberg.org/files/{gid}/{gid}-0.txt",
                    f"https://gutenberg.org/files/{gid}/{gid}.txt",
                ]:
                    r = requests.get(mirror, timeout=20)
                    if r.status_code == 200 and len(r.text) > 1000:
                        text = r.text
                        for marker in ["*** START OF", "***START OF"]:
                            if marker in text:
                                text = text[text.index(marker):]
                                text = text[text.index("\n")+1:]
                                break
                        for marker in ["*** END OF", "***END OF", "End of Project Gutenberg"]:
                            if marker in text:
                                text = text[:text.index(marker)]
                        words = text.split()
                        for chunk_start in range(0, len(words), 1500):
                            chunk = " ".join(words[chunk_start:chunk_start+1500])
                            if len(chunk) > 300:
                                docs.append(
                                    f"[Robotics/Automation Textbook — Gutenberg #{gid}]\n\n{chunk}"
                                )
                        log.info(f"    Gutenberg #{gid}: OK ({len(words)} words)")
                        break
                time.sleep(0.5)
            except Exception as ge:
                log.warning(f"    Gutenberg #{gid}: {ge}")
        n = write_docs(bucket, docs, "robo_gutenberg")
        record_source("corpus_C_technical", "Gutenberg Robotics/Automation Texts",
                      "https://www.gutenberg.org",
                      approx_mb(docs),
                      "Classic automation and cybernetics: Wiener cybernetics, "
                      "scientific management, mechanical systems", n)
        total += n
        log.info(f"  [ROBO-4] Done: {n} chunks")
    except Exception as e:
        log.error(f"  [ROBO-4] Failed: {e}")
        record_exclusion("Gutenberg Robotics", str(e))

    # ── ROBO-5: OpenAlex nanoscience / MEMS papers ────────────────────────────
    try:
        log.info("[ROBO-5] OpenAlex nanomechanics / MEMS papers...")
        docs = []
        queries = [
            "MEMS micro electro mechanical systems",
            "nanomechanics atomic force microscopy",
            "molecular motor biological nanomachine",
            "soft robotics compliant mechanism",
            "nano robot targeted drug delivery",
        ]
        for query in queries:
            try:
                url = (
                    f"https://api.openalex.org/works?"
                    f"search={requests.utils.quote(query)}"
                    f"&per-page=100&select=title,abstract_inverted_index,publication_year"
                    f"&filter=open_access.is_oa:true"
                )
                r = requests.get(url, timeout=20, headers={"User-Agent": "TitanAI/1.0"})
                if r.status_code != 200:
                    continue
                results = r.json().get("results", [])
                for work in results:
                    title = (work.get("title", "") or "").strip()
                    inv   = work.get("abstract_inverted_index") or {}
                    if not inv:
                        continue
                    # Reconstruct abstract from inverted index
                    max_pos = max(pos for positions in inv.values() for pos in positions)
                    words_list = [""] * (max_pos + 1)
                    for word, positions in inv.items():
                        for pos in positions:
                            words_list[pos] = word
                    abstract = " ".join(w for w in words_list if w)
                    if len(abstract) > 100:
                        docs.append(
                            f"Nanomechanics/Robotics Research\n\n"
                            f"Title: {title}\n\nAbstract:\n{abstract}"
                        )
            except Exception as qe:
                log.warning(f"    OpenAlex query '{query[:30]}': {qe}")
            time.sleep(0.3)
        n = write_docs(bucket, docs, "robo_openalex")
        if n > 0:
            record_source("corpus_C_technical", "OpenAlex Nanomechanics/MEMS",
                          "https://openalex.org",
                          approx_mb(docs),
                          "Open-access nanomechanics research: MEMS, molecular motors, "
                          "AFM, soft robotics, nano-robots", n)
        total += n
        log.info(f"  [ROBO-5] Done: {n} docs")
    except Exception as e:
        log.error(f"  [ROBO-5] Failed (non-critical): {e}")

    log.info(f"\n  >>> Robotics & Nano-Mechanics total: {total} docs")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("TitanAI Engineering Corpus Loader — Starting")
    log.info(f"Target directory: {RAW}")
    log.info("Domains: Electrical Engineering | Mechanics | Robotics & Nano-Mechanics")
    log.info("")

    results = {}
    results["electrical_engineering"]  = load_electrical_engineering()
    results["mechanics"]               = load_mechanics()
    results["robotics_nanomechanics"]  = load_robotics_nanomechanics()

    # Persist inventory
    with open(INVENTORY, "w") as f:
        json.dump(inventory, f, indent=2)
    log.info(f"Source inventory saved: {INVENTORY}")

    with open(EXCLUSION_LOG, "w") as f:
        for exc in exclusions:
            f.write(json.dumps(exc) + "\n")
    log.info(f"Exclusion log saved: {EXCLUSION_LOG}")

    log.info("")
    log.info("=" * 60)
    log.info("ENGINEERING CORPUS LOAD COMPLETE")
    log.info("=" * 60)
    grand_total = 0
    for domain, n in results.items():
        log.info(f"  {domain}: {n:,} documents added")
        grand_total += n
    log.info(f"  GRAND TOTAL: {grand_total:,} documents")
    log.info("")
    log.info("Next steps:")
    log.info("  1. python scripts/validate_corpus_quality.py")
    log.info("  2. python scripts/generate_shards.py   # rebuilds shards")
    log.info("  3. python scripts/pretrain_titan_v3.py --config configs/titan_1b.yaml")

if __name__ == "__main__":
    main()
