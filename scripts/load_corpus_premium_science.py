"""
  TitanAI Premium Science Corpus Loader — 9 Domains
  ===================================================
  Adds premium-depth training data across nine scientific domains
  to corpus_C_technical. Target: 120,000+ high-quality documents.

  Domains:
    1. Advanced Chemistry       — organic, inorganic, physical, analytical,
                                   biochemistry, polymer, computational, materials
    2. Biology                  — cell, molecular, genetics, evolution, ecology,
                                   microbiology, physiology, neuroscience, biotech
    3. Advanced Mathematics     — analysis, algebra, topology, number theory,
                                   probability, statistics, discrete, numerical
    4. Advanced Physics         — classical, quantum, condensed matter, relativity,
                                   astrophysics, particle physics, plasma, optics
    5. Medicine                 — Western clinical + Eastern (TCM, Ayurveda, Kampo,
                                   Unani, Siddha) + integrative & herbal medicine
    6. Advanced Electrical Eng. — power systems, RF/microwave, photonics, analog IC,
                                   EMC, instrumentation, high-voltage, electric machines
    7. Advanced Mechanical Eng. — fracture mechanics, CFD, combustion, tribology,
                                   advanced materials, precision engineering, aero
    8. Systems Design (Online)  — digital, FPGA/ASIC, distributed, control theory,
                                   DSP, communications, power electronics, software arch
    9. Systems Design (Mech)    — mechanism design, hydraulics, thermal systems,
                                   manufacturing, mechatronics, composites

  Sources per domain (layered for depth):
    • Wikipedia REST API  — targeted subtopic queries (50+ queries per domain)
    • wikimedia/wikipedia — HuggingFace streaming, keyword-filtered
    • StackExchange Q&A   — HuggingFaceH4/stack-exchange-preferences
    • arXiv preprints     — Cornell-University/arxiv, per-domain categories
    • Project Gutenberg   — classic open textbooks
    • OpenAlex API        — open-access research papers
    • PubChem API         — compound chemistry data
    • NIST WebBook        — physics & chemistry reference data
    • WHO/NIH OpenAlex    — medical research & guidelines

  Run on instance:
    pip install datasets requests tqdm
    python scripts/load_corpus_premium_science.py
  """

  import os, sys, json, time, logging, textwrap, requests
  from pathlib import Path
  from datetime import datetime

  # ── Paths ──────────────────────────────────────────────────────────────────────
  BASE = Path(__file__).parent.parent
  RAW  = BASE / "data" / "raw"
  LOG  = BASE / "data" / "corpus_premium_science.log"

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(message)s",
      handlers=[logging.StreamHandler(), logging.FileHandler(str(LOG))]
  )
  log = logging.getLogger("premium_loader")

  INVENTORY     = BASE / "data" / "source_inventory.json"
  EXCLUSION_LOG = BASE / "data" / "exclusions_premium.jsonl"

  try:
      with open(INVENTORY) as f:
          inventory = json.load(f)
  except Exception:
      inventory = {}
  exclusions = []

  # ── Core helpers ───────────────────────────────────────────────────────────────
  def record_source(bucket, source_name, url, size_mb, reason, n_docs):
      inventory.setdefault(bucket, []).append({
          "source": source_name, "url": url,
          "size_mb": round(size_mb, 2), "n_documents": n_docs,
          "reason_for_inclusion": reason,
          "loaded_at": datetime.utcnow().isoformat(),
      })

  def record_exclusion(source_name, reason):
      exclusions.append({"source": source_name, "reason": reason,
                         "timestamp": datetime.utcnow().isoformat()})
      log.warning(f"EXCLUDED: {source_name} — {reason}")

  def write_docs(bucket_dir, docs, source_tag):
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

  def safe_get(d, *keys, default=""):
      for k in keys:
          d = d.get(k, default) if isinstance(d, dict) else default
      return d or default

  def gutenberg_strip(text):
      for marker in ["*** START OF", "***START OF", "*END*THE SMALL PRINT"]:
          if marker in text:
              text = text[text.index(marker):]
              text = text[text.index("\n")+1:]
              break
      for marker in ["*** END OF", "***END OF", "End of Project Gutenberg",
                     "End of the Project Gutenberg"]:
          if marker in text:
              text = text[:text.index(marker)]
      return text

  # ── Reusable source loaders ────────────────────────────────────────────────────

  def wiki_api_targeted(queries, bucket_dir, source_tag, max_per_query=25):
      """
      Fetch full Wikipedia articles via REST API for a list of search queries.
      Returns number of docs written.
      """
      SEARCH = "https://en.wikipedia.org/w/api.php"
      seen_titles = set()
      docs = []
      for query in queries:
          try:
              r = requests.get(SEARCH, params={
                  "action": "query", "list": "search", "srsearch": query,
                  "srlimit": max_per_query, "format": "json",
                  "srnamespace": 0
              }, timeout=15)
              hits = r.json().get("query", {}).get("search", [])
              for hit in hits:
                  title = hit["title"]
                  if title in seen_titles:
                      continue
                  seen_titles.add(title)
                  # Fetch full article text
                  ar = requests.get(SEARCH, params={
                      "action": "query", "titles": title,
                      "prop": "extracts", "explaintext": True,
                      "exsectionformat": "plain", "format": "json"
                  }, timeout=15)
                  pages = ar.json().get("query", {}).get("pages", {})
                  for pid, page in pages.items():
                      if pid == "-1":
                          continue
                      text = page.get("extract", "").strip()
                      if len(text) > 300:
                          docs.append(f"# {title}\n\n{text[:10000]}")
                  time.sleep(0.12)
          except Exception as e:
              log.warning(f"    wiki_api '{query[:40]}': {e}")
          time.sleep(0.05)
      return write_docs(bucket_dir, docs, source_tag)

  def wiki_stream_filtered(keywords, bucket_dir, source_tag, max_docs=5000):
      """Filter wikimedia/wikipedia by keyword match, write docs."""
      try:
          from datasets import load_dataset
          wiki = load_dataset("wikimedia/wikipedia", "20231101.en",
                              split="train", streaming=True)
          docs = []
          scanned = 0
          kw_lower = {k.lower() for k in keywords}
          for item in wiki:
              scanned += 1
              if scanned > 800_000:
                  break
              title = item.get("title", "").lower()
              text  = item.get("text", "").strip()
              if len(text) < 300:
                  continue
              snippet = (title + " " + text[:600]).lower()
              if any(kw in snippet for kw in kw_lower):
                  docs.append(f"# {item['title']}\n\n{text[:9000]}")
              if len(docs) >= max_docs:
                  break
          return write_docs(bucket_dir, docs, source_tag)
      except Exception as e:
          record_exclusion(f"wiki_stream:{source_tag}", str(e))
          return 0

  def stackexchange_qa(domain_keywords, bucket_dir, source_tag,
                       extra_kw_filter=None, max_docs=4000):
      """Extract Q&A from HuggingFaceH4/stack-exchange-preferences."""
      try:
          from datasets import load_dataset
          se = load_dataset("HuggingFaceH4/stack-exchange-preferences",
                            split="train", streaming=True)
          docs = []
          for i, item in enumerate(se):
              if i > 3_000_000:
                  break
              domain = safe_get(item, "domain")
              if not any(kw in domain.lower() for kw in domain_keywords):
                  continue
              question = safe_get(item, "question")
              if extra_kw_filter:
                  if not any(kw in question.lower() for kw in extra_kw_filter):
                      continue
              answers = item.get("answers", []) or []
              best_ans, best_score = "", -999
              for ans in answers:
                  sc = ans.get("pm_score", 0) or 0
                  if sc > best_score:
                      best_score = sc
                      best_ans = ans.get("text", "")
              if best_ans and len(question) > 60:
                  docs.append(
                      f"Q&A [{domain}]\n\n"
                      f"Question: {question.strip()}\n\n"
                      f"Best Answer (score {best_score}):\n{best_ans.strip()[:4000]}"
                  )
              if len(docs) >= max_docs:
                  break
          return write_docs(bucket_dir, docs, source_tag)
      except Exception as e:
          record_exclusion(f"stackexchange:{source_tag}", str(e))
          return 0

  def arxiv_filtered(categories, bucket_dir, source_tag,
                     extra_kw=None, max_docs=4000):
      """Extract arXiv abstracts by category (+ optional keyword)."""
      try:
          from datasets import load_dataset
          arxiv = load_dataset("Cornell-University/arxiv",
                               split="train", streaming=True)
          cat_set = set(categories)
          docs = []
          for i, item in enumerate(arxiv):
              if i > 4_000_000:
                  break
              cats = set((item.get("categories", "") or "").split())
              if not cats.intersection(cat_set):
                  continue
              title    = (item.get("title", "") or "").replace("\n", " ").strip()
              abstract = (item.get("abstract", "") or "").replace("\n", " ").strip()
              authors  = (item.get("authors", "") or "").strip()
              journal  = (item.get("journal-ref", "") or "").strip()
              if len(abstract) < 80:
                  continue
              if extra_kw:
                  combo = (title + " " + abstract).lower()
                  if not any(k in combo for k in extra_kw):
                      continue
              doc = f"Title: {title}\nAuthors: {authors}\n"
              if journal:
                  doc += f"Published: {journal}\n"
              doc += f"Categories: {' '.join(sorted(cats))}\n\nAbstract:\n{abstract}"
              docs.append(doc)
              if len(docs) >= max_docs:
                  break
          return write_docs(bucket_dir, docs, source_tag)
      except Exception as e:
          record_exclusion(f"arxiv:{source_tag}", str(e))
          return 0

  def gutenberg_load(ids_with_desc, bucket_dir, source_tag):
      """Download and chunk Project Gutenberg texts."""
      docs = []
      for gid, desc in ids_with_desc:
          fetched = False
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
                          chunk = " ".join(words[s:s+1800])
                          if len(chunk) > 300:
                              docs.append(f"[{desc} — Gutenberg #{gid}]\n\n{chunk}")
                      log.info(f"    Gutenberg #{gid} ({desc}): {len(words)} words")
                      fetched = True
                      break
              except Exception as e:
                  log.warning(f"    Gutenberg #{gid}: {e}")
          if not fetched:
              log.warning(f"    Gutenberg #{gid}: all mirrors failed")
          time.sleep(0.6)
      return write_docs(bucket_dir, docs, source_tag)

  def openalex_fetch(queries, bucket_dir, source_tag, label="Research"):
      """Fetch open-access papers from OpenAlex API."""
      docs = []
      for query in queries:
          try:
              url = (
                  "https://api.openalex.org/works?"
                  f"search={requests.utils.quote(query)}"
                  "&per-page=100&filter=open_access.is_oa:true"
                  "&select=title,abstract_inverted_index,publication_year,primary_location"
              )
              r = requests.get(url, timeout=20,
                               headers={"User-Agent": "TitanAI/1.0"})
              if r.status_code != 200:
                  continue
              for work in r.json().get("results", []):
                  title = (work.get("title", "") or "").strip()
                  inv   = work.get("abstract_inverted_index") or {}
                  if not inv:
                      continue
                  max_pos = max(p for ps in inv.values() for p in ps)
                  words_list = [""] * (max_pos + 1)
                  for word, positions in inv.items():
                      for pos in positions:
                          words_list[pos] = word
                  abstract = " ".join(w for w in words_list if w)
                  if len(abstract) > 100:
                      docs.append(
                          f"{label} Paper\n\nTitle: {title}\n\n"
                          f"Abstract:\n{abstract}"
                      )
          except Exception as e:
              log.warning(f"    OpenAlex '{query[:35]}': {e}")
          time.sleep(0.35)
      return write_docs(bucket_dir, docs, source_tag)

  
  # ══════════════════════════════════════════════════════════════════════════════
  # KEYWORD SETS  (used for Wikipedia streaming filters)
  # ══════════════════════════════════════════════════════════════════════════════

  CHEM_KW = {
      "organic chemistry","reaction mechanism","nucleophilic substitution",
      "electrophilic addition","elimination reaction","diels-alder","aldol condensation",
      "grignard reagent","wittig reaction","retrosynthesis","stereochemistry","chirality",
      "enantiomer","diastereomer","functional group","carbonyl","aromatic compound",
      "benzene","phenol","carboxylic acid","amine","amide","ester","ketone","aldehyde",
      "polymer chemistry","condensation polymerization","radical polymerization",
      "inorganic chemistry","coordination compound","ligand","transition metal",
      "crystal field theory","d-orbital","oxidation state","chelate",
      "organometallic","physical chemistry","gibbs free energy","enthalpy","entropy",
      "reaction kinetics","rate law","activation energy","arrhenius equation",
      "quantum chemistry","molecular orbital","vsepr","hybridization",
      "analytical chemistry","nmr spectroscopy","infrared spectroscopy",
      "mass spectrometry","chromatography","hplc","gas chromatography",
      "electrochemistry","titration","spectrophotometry",
      "biochemistry","enzyme kinetics","metabolic pathway","glycolysis",
      "krebs cycle","atp synthesis","protein structure","amino acid",
      "materials chemistry","semiconductor material","band gap","nanomaterial",
      "surface chemistry","heterogeneous catalysis","zeolite",
      "computational chemistry","density functional theory","molecular dynamics",
      "force field","polymer science","cross-linking","crystallography",
      "x-ray diffraction","carbon nanotube","graphene","sol-gel","electroplating",
  }

  BIO_KW = {
      "cell biology","organelle","mitochondria","endoplasmic reticulum",
      "golgi apparatus","cell membrane","phospholipid bilayer","signal transduction",
      "cell signaling","receptor","second messenger","cell cycle","mitosis","meiosis",
      "chromosome","chromatin","gene expression","transcription factor","epigenetics",
      "dna methylation","histone","molecular biology","dna replication","rna polymerase",
      "ribosome","translation","codon","genetic code","mutation","gene regulation",
      "crispr","gene editing","genetics","mendelian genetics","allele","dominance",
      "genetic linkage","population genetics","hardy-weinberg","natural selection",
      "genetic drift","evolution","phylogenetics","speciation","adaptation",
      "ecology","ecosystem","food web","trophic level","biogeochemical cycle",
      "nitrogen cycle","carbon cycle","population ecology","community ecology","biome",
      "microbiology","bacteria","archaea","virus","bacteriophage","antibiotic resistance",
      "pathogenesis","biofilm","microbiome","physiology","homeostasis",
      "nervous system","cardiovascular system","respiratory system","endocrine system",
      "immune system","digestive system","neuroscience","neuron","synapse",
      "neurotransmitter","action potential","neural circuit","brain region",
      "biotechnology","pcr","gel electrophoresis","western blot","cell culture",
      "protein engineering","monoclonal antibody","stem cell","regenerative medicine",
      "proteomics","genomics","bioinformatics","metagenomics","synthetic biology",
  }

  MATH_KW = {
      "differential calculus","integral calculus","limit","derivative","chain rule",
      "integration by parts","taylor series","maclaurin series",
      "multivariable calculus","partial derivative","gradient","divergence","curl",
      "stokes theorem","green's theorem","divergence theorem",
      "ordinary differential equation","partial differential equation",
      "laplace equation","fourier series","fourier transform","laplace transform",
      "linear algebra","matrix algebra","determinant","eigenvalue","eigenvector",
      "singular value decomposition","linear transformation","vector space","basis",
      "orthogonality","inner product space","real analysis","continuity",
      "metric space","cauchy sequence","measure theory","lebesgue integral",
      "complex analysis","holomorphic function","cauchy-riemann","contour integration",
      "residue theorem","conformal mapping","topology","homeomorphism","homotopy",
      "manifold","algebraic topology","abstract algebra","group theory","ring theory",
      "field theory","galois theory","lie algebra","number theory","prime number",
      "modular arithmetic","diophantine equation","riemann hypothesis",
      "combinatorics","graph theory","generating function","ramsey theory",
      "probability theory","random variable","central limit theorem",
      "stochastic process","markov chain","bayesian inference",
      "hypothesis testing","regression analysis","maximum likelihood",
      "numerical analysis","newton's method","finite difference","runge-kutta",
      "finite element method","convex optimization","linear programming",
      "gradient descent","dynamic programming","category theory","set theory",
      "mathematical logic","boolean algebra","turing machine","complexity theory",
  }

  PHYSICS_KW = {
      "classical mechanics","lagrangian mechanics","hamiltonian mechanics",
      "action principle","conservation law","angular momentum","rigid body",
      "moment of inertia","wave mechanics","wave equation",
      "maxwell's equations","electromagnetic wave","gauss's law","faraday's law",
      "ampere's law","electromagnetic induction","electric field","magnetic field",
      "electromagnetic spectrum","geometric optics","diffraction","interference",
      "polarization","quantum mechanics","wave-particle duality",
      "schrödinger equation","quantum state","superposition","entanglement",
      "heisenberg uncertainty","quantum tunneling","quantum field theory",
      "particle physics","standard model","higgs boson","quark","lepton",
      "hadron","boson","fermion","nuclear physics","radioactive decay",
      "fission","fusion","nuclear reactor","special relativity","time dilation",
      "length contraction","lorentz transformation","mass-energy equivalence",
      "general relativity","spacetime curvature","gravitational waves","black hole",
      "cosmology","big bang","dark matter","dark energy","cosmic inflation",
      "astrophysics","stellar evolution","neutron star","pulsar","galaxy formation",
      "condensed matter physics","solid state physics","crystal lattice","band theory",
      "superconductivity","bcs theory","topological insulator","quantum hall effect",
      "statistical mechanics","boltzmann distribution","partition function",
      "bose-einstein condensate","fermi-dirac statistics","plasma physics",
      "magnetohydrodynamics","chaos theory","fluid mechanics","turbulence",
      "nonlinear dynamics","string theory","loop quantum gravity",
  }

  MED_WESTERN_KW = {
      "anatomy","histology","pathology","pharmacology","pharmacokinetics",
      "pharmacodynamics","clinical medicine","internal medicine","surgery",
      "differential diagnosis","evidence-based medicine","randomized controlled trial",
      "cardiology","myocardial infarction","arrhythmia","heart failure","hypertension",
      "atherosclerosis","coronary artery","echocardiography","cardiac catheterization",
      "neurology","stroke","epilepsy","parkinson's disease","alzheimer's disease",
      "multiple sclerosis","neurodegenerative","cerebrospinal fluid",
      "oncology","cancer","tumor","chemotherapy","radiation therapy","immunotherapy",
      "metastasis","carcinoma","lymphoma","leukemia","biopsy","staging",
      "pulmonology","pneumonia","asthma","copd","pulmonary fibrosis",
      "mechanical ventilation","bronchoscopy",
      "gastroenterology","liver disease","inflammatory bowel disease","cirrhosis",
      "endoscopy","hepatitis","pancreatitis",
      "immunology","immune response","autoimmune disease","allergy","vaccination",
      "antibody","t cell","b cell","cytokine","inflammation","innate immunity",
      "endocrinology","diabetes","thyroid disease","hormones","metabolic syndrome",
      "insulin resistance","adrenal gland","pituitary",
      "infectious disease","antibiotic","antiviral","antimicrobial resistance",
      "sepsis","covid-19","viral replication","epidemiology","outbreak",
      "orthopedics","fracture","joint replacement","arthroplasty","osteoporosis",
      "radiology","mri","ct scan","x-ray","ultrasound","nuclear medicine","pet scan",
      "psychiatry","depression","schizophrenia","antidepressant","cbt","anxiety",
      "obstetrics","gynecology","pediatrics","neonatology","prenatal",
      "precision medicine","clinical trial","pharmacogenomics","proteomics",
      "medical imaging","telemedicine","robotic surgery","laparoscopy",
  }

  MED_EASTERN_KW = {
      "traditional chinese medicine","tcm","acupuncture","meridian","qi energy",
      "yin yang","five elements","herbal medicine","chinese herbal","materia medica",
      "zang-fu organs","tongue diagnosis","pulse diagnosis","moxibustion",
      "cupping therapy","tui na massage","qigong","tai chi","chinese pharmacopoeia",
      "ayurveda","dosha","vata pitta kapha","panchakarma","ayurvedic medicine",
      "yoga therapy","pranayama","rasayana","triphala","ashwagandha","turmeric",
      "ayurvedic herb","siddha medicine","unani medicine","tibb","hippocratic",
      "kampo","japanese traditional medicine","korean medicine","hanbang",
      "native american medicine","indigenous healing","shamanic healing",
      "naturopathy","homeopathy","botanical medicine","herbal remedy",
      "phytotherapy","adaptogen","tonic herb","medicinal plant",
      "integrative medicine","complementary medicine","holistic medicine",
      "mind-body medicine","meditation healing","biofeedback","acupressure",
      "reflexology","chiropractic","osteopathy","traditional african medicine",
      "ethnobotany","ethnomedicine","traditional knowledge","folk medicine",
  }

  ADV_EE_KW = {
      "power system analysis","load flow","fault analysis","power system stability",
      "synchronous machine","induction motor drive","thyristor","igbt",
      "power converter topology","switched-mode power supply",
      "electromagnetic compatibility","emc","signal integrity","power integrity",
      "rf circuit design","microwave engineering","transmission line theory",
      "smith chart","impedance matching","antenna design","phased array",
      "radar systems","electronic warfare","photonics","optical fiber communication",
      "laser diode","photodetector","semiconductor physics","p-n junction",
      "bipolar transistor","mosfet operation","cmos logic","memory design",
      "analog ic design","differential amplifier","bandgap reference",
      "oscillator design","phase-locked loop","analog-to-digital converter",
      "noise analysis","thermal noise","active filter","switched capacitor",
      "instrumentation amplifier","precision measurement","data acquisition",
      "grounding shielding","high-voltage engineering","insulation coordination",
      "power quality","harmonic distortion","electric machine design",
      "motor efficiency","regenerative braking","power electronics",
      "grid-tied inverter","wireless power transfer","energy harvesting",
      "electric vehicle charging","battery management system",
  }

  ADV_ME_KW = {
      "fracture mechanics","fatigue crack growth","damage tolerance",
      "plasticity theory","creep mechanics","viscoelasticity",
      "computational fluid dynamics","turbulence modeling","boundary layer theory",
      "compressible flow","shock wave","hypersonic aerodynamics",
      "exergy analysis","combined cycle","rankine cycle",
      "combustion engineering","internal combustion engine","fuel cell",
      "advanced heat transfer","radiation heat transfer","boiling condensation",
      "heat pipe","titanium alloy","nickel superalloy","ceramic matrix composite",
      "shape memory alloy","additive manufacturing","3d printing metal",
      "topology optimization","reliability engineering","weibull analysis","fmea",
      "noise vibration harshness","acoustic emission","condition monitoring",
      "predictive maintenance","corrosion engineering","cathodic protection",
      "surface treatment","nano-manufacturing","mems fabrication","photolithography",
      "precision engineering","ultra-precision machining","metrology","cmm",
      "aeroelasticity","flutter","wind tunnel testing","gas turbine",
  }

  SYS_ONLINE_KW = {
      "digital systems design","boolean algebra","combinational logic",
      "sequential logic","finite state machine","flip-flop","multiplexer",
      "fpga field programmable gate array","verilog","vhdl","rtl design",
      "hardware synthesis","place and route","timing constraint","clock domain",
      "asic design","chip design","microarchitecture","pipeline processor",
      "hazard detection","branch prediction","cache coherence","memory hierarchy",
      "out-of-order execution","embedded system","real-time operating system",
      "rtos","interrupt handling","device driver","microcontroller",
      "tcp/ip protocol","routing algorithm","bgp ospf","software-defined networking",
      "distributed system","consensus algorithm","raft paxos","cap theorem",
      "eventual consistency","microservices","message queue","event sourcing",
      "state space representation","transfer function design","bode plot",
      "nyquist criterion","root locus","optimal control","lqr controller",
      "model predictive control","digital signal processing","fir iir filter design",
      "fast fourier transform","z-transform","dsp architecture",
      "ofdm","mimo antenna","channel coding","error correction","5g nr","lte",
      "buck boost converter","inverter design","pulse width modulation",
      "design pattern","microservices architecture","database design","scalability",
  }

  SYS_MECH_KW = {
      "mechanism design","kinematic synthesis","four-bar linkage","cam mechanism",
      "gear design","gear train","epicyclic gear","involute tooth profile",
      "machine element design","rolling element bearing","shaft coupling",
      "spring design","fatigue life","bolted joint","weld design",
      "truss analysis","frame analysis","finite element method",
      "stress concentration","buckling analysis","structural dynamics",
      "hydraulic system design","hydraulic pump","directional control valve",
      "servo hydraulic","proportional valve","hydraulic actuator",
      "pneumatic system design","pneumatic logic circuit",
      "heat exchanger sizing","shell and tube","refrigeration cycle",
      "heat pump design","thermal management","hvac psychrometrics",
      "lean manufacturing","cellular manufacturing","six sigma",
      "process capability","cnc machining","computer aided design",
      "parametric cad","tolerance analysis","gd&t",
      "servo motor control","motion control","linear actuator",
      "precision mechanism","vibration isolation","composite structure design",
      "fiber reinforced polymer","laminate analysis","failure criterion",
      "tribology","surface engineering","lubrication regime","wear mechanism",
  }

  
  # ══════════════════════════════════════════════════════════════════════════════
  # DOMAIN 1 — ADVANCED CHEMISTRY
  # ══════════════════════════════════════════════════════════════════════════════
  def load_advanced_chemistry():
      log.info("=" * 65)
      log.info("DOMAIN 1: Advanced Chemistry")
      log.info("=" * 65)
      bucket = RAW / "corpus_C_technical"
      total  = 0

      wiki_queries_chem = [
          "organic chemistry reaction mechanisms",
          "nucleophilic substitution SN1 SN2",
          "electrophilic aromatic substitution",
          "Diels-Alder cycloaddition reaction",
          "aldol condensation reaction mechanism",
          "Grignard reagent reactions",
          "carboxylic acid derivatives reactions",
          "stereochemistry chirality enantiomers",
          "retrosynthetic analysis organic chemistry",
          "polymer chemistry polymerization mechanisms",
          "coordination chemistry ligand field theory",
          "organometallic chemistry catalysis",
          "transition metal complexes crystal field",
          "physical chemistry thermodynamics",
          "chemical kinetics rate laws",
          "quantum chemistry molecular orbital theory",
          "NMR spectroscopy structure determination",
          "mass spectrometry fragmentation patterns",
          "infrared spectroscopy functional groups",
          "HPLC chromatography separation techniques",
          "electrochemistry redox reactions",
          "enzyme kinetics Michaelis Menten",
          "metabolic pathways glycolysis TCA cycle",
          "protein structure folding biochemistry",
          "DNA RNA biochemistry nucleic acids",
          "lipid biochemistry membrane structure",
          "computational chemistry DFT calculations",
          "molecular dynamics simulation force fields",
          "materials chemistry semiconductor doping",
          "nanomaterials carbon nanotubes graphene",
          "surface chemistry catalysis heterogeneous",
          "polymer materials glass transition",
          "analytical chemistry titration methods",
          "photochemistry excited states reactions",
          "green chemistry sustainable synthesis",
          "medicinal chemistry drug design",
          "supramolecular chemistry host-guest",
          "solid state chemistry crystal structures",
          "atmospheric chemistry ozone reactions",
          "electroanalytical chemistry voltammetry",
      ]

      n = wiki_api_targeted(wiki_queries_chem, bucket, "advchem_wiki_api", max_per_query=20)
      record_source("corpus_C_technical", "Wikipedia Chemistry API (targeted)",
                    "https://en.wikipedia.org", approx_mb([""]*n),
                    "Targeted Wikipedia articles: all major chemistry subfields", n)
      total += n; log.info(f"  [CHEM-1] Wikipedia API: {n} docs")

      n = wiki_stream_filtered(CHEM_KW, bucket, "advchem_wiki_stream", max_docs=5000)
      record_source("corpus_C_technical", "Wikipedia Chemistry Stream",
                    "https://huggingface.co/datasets/wikimedia/wikipedia",
                    approx_mb([""]*n), "Broad Wikipedia chemistry keyword scan", n)
      total += n; log.info(f"  [CHEM-2] Wikipedia stream: {n} docs")

      n = stackexchange_qa(["chemistry"], bucket, "advchem_se", max_docs=4000)
      record_source("corpus_C_technical", "Chemistry StackExchange Q&A",
                    "https://chemistry.stackexchange.com", approx_mb([""]*n),
                    "Expert chemistry Q&A: mechanisms, lab techniques, theory", n)
      total += n; log.info(f"  [CHEM-3] StackExchange: {n} docs")

      n = arxiv_filtered(["physics.chem-ph","cond-mat.mtrl-sci","q-bio.BM"],
                         bucket, "advchem_arxiv", max_docs=4000)
      record_source("corpus_C_technical", "arXiv Chemistry Papers",
                    "https://arxiv.org", approx_mb([""]*n),
                    "Chemistry research: chem-ph, materials science, biomolecules", n)
      total += n; log.info(f"  [CHEM-4] arXiv: {n} docs")

      n = gutenberg_load([
          (45503, "A System of Chemistry — Thomson"),
          (16097, "Chemistry — The Central Science"),
          (22238, "The Principles of Chemistry — Mendeleev"),
          (14612, "A History of Chemistry — Thorpe"),
          (38709, "Organic Chemistry for Advanced Students"),
          (58585, "Physical Chemistry — Lewis"),
          (27552, "Chemistry of Familiar Things"),
          (10015, "The Electron Theory of Matter"),
          (9181,  "On the Various Forces of Nature — Faraday"),
          (55148, "Experimental Researches in Electricity — Faraday"),
      ], bucket, "advchem_gutenberg")
      record_source("corpus_C_technical", "Gutenberg Chemistry Textbooks",
                    "https://www.gutenberg.org", approx_mb([""]*n),
                    "Classic texts: Mendeleev, Faraday, Thomson, Lewis, Thorpe", n)
      total += n; log.info(f"  [CHEM-5] Gutenberg: {n} docs")

      # PubChem compound descriptions
      try:
          log.info("  [CHEM-6] PubChem compound data...")
          docs = []
          compound_cids = list(range(1, 500)) + list(range(2000, 2300)) +                           list(range(5000, 5200)) + list(range(10000, 10150))
          for cid in compound_cids:
              try:
                  r = requests.get(
                      f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}"
                      f"/description/JSON", timeout=10
                  )
                  if r.status_code != 200: continue
                  for info in r.json().get("InformationList", {}).get("Information", []):
                      title = info.get("Title", "")
                      desc  = info.get("Description", "")
                      src   = info.get("DescriptionSourceName", "")
                      if desc and len(desc) > 80:
                          docs.append(
                              f"PubChem Compound: {title}\nCID: {cid}\n"
                              f"Source: {src}\n\nDescription:\n{desc}"
                          )
              except: pass
              time.sleep(0.06)
          n = write_docs(bucket, docs, "advchem_pubchem")
          record_source("corpus_C_technical", "PubChem Compound Descriptions",
                        "https://pubchem.ncbi.nlm.nih.gov", approx_mb(docs),
                        "Official compound descriptions for common chemicals", n)
          total += n; log.info(f"  [CHEM-6] PubChem: {n} docs")
      except Exception as e:
          log.error(f"  [CHEM-6] PubChem failed: {e}")

      n = openalex_fetch([
          "organic synthesis total synthesis natural products",
          "asymmetric catalysis enantioselective reactions",
          "computational quantum chemistry DFT",
          "polymer synthesis controlled radical",
          "nanomaterials synthesis characterization",
          "green chemistry solvent-free reactions",
          "medicinal chemistry drug discovery",
          "electrochemistry energy storage",
          "protein folding molecular simulation",
          "materials chemistry perovskite",
      ], bucket, "advchem_openalex", label="Chemistry Research")
      record_source("corpus_C_technical", "OpenAlex Chemistry Papers",
                    "https://openalex.org", approx_mb([""]*n),
                    "Open-access chemistry research papers", n)
      total += n; log.info(f"  [CHEM-7] OpenAlex: {n} docs")

      log.info(f"  >>> Advanced Chemistry total: {total:,} docs")
      return total


  # ══════════════════════════════════════════════════════════════════════════════
  # DOMAIN 2 — BIOLOGY
  # ══════════════════════════════════════════════════════════════════════════════
  def load_biology():
      log.info("=" * 65)
      log.info("DOMAIN 2: Biology")
      log.info("=" * 65)
      bucket = RAW / "corpus_C_technical"
      total  = 0

      wiki_queries_bio = [
          "cell biology organelles membrane transport",
          "cell signaling signal transduction pathways",
          "cell cycle mitosis meiosis regulation",
          "DNA replication repair mechanisms",
          "RNA transcription gene expression regulation",
          "protein synthesis translation ribosome",
          "epigenetics DNA methylation histone modification",
          "CRISPR gene editing genome engineering",
          "Mendelian genetics inheritance laws",
          "population genetics Hardy-Weinberg equilibrium",
          "molecular evolution phylogenetics",
          "natural selection evolutionary biology Darwin",
          "ecology ecosystem food webs energy flow",
          "biogeochemical cycles nitrogen carbon",
          "microbiology bacteria pathogenesis",
          "virology virus replication immune evasion",
          "immunology adaptive immune response",
          "T cell B cell antibody production",
          "neuroscience neuron synapse neurotransmitter",
          "action potential neural circuits brain",
          "human physiology cardiovascular system",
          "endocrine system hormones metabolism",
          "respiratory system gas exchange",
          "digestive system biochemistry enzymes",
          "biotechnology PCR cloning techniques",
          "genomics sequencing bioinformatics",
          "proteomics mass spectrometry protein identification",
          "synthetic biology genetic circuits",
          "stem cell biology differentiation",
          "cancer biology tumor suppressor oncogene",
          "plant biology photosynthesis chloroplast",
          "developmental biology embryogenesis",
          "marine biology deep sea ecosystems",
          "microbial ecology microbiome",
          "evolutionary genetics adaptation speciation",
      ]

      n = wiki_api_targeted(wiki_queries_bio, bucket, "bio_wiki_api", max_per_query=20)
      record_source("corpus_C_technical", "Wikipedia Biology API (targeted)",
                    "https://en.wikipedia.org", approx_mb([""]*n),
                    "Targeted Wikipedia: cell bio, genetics, evolution, neuroscience", n)
      total += n; log.info(f"  [BIO-1] Wikipedia API: {n} docs")

      n = wiki_stream_filtered(BIO_KW, bucket, "bio_wiki_stream", max_docs=5000)
      record_source("corpus_C_technical", "Wikipedia Biology Stream",
                    "https://huggingface.co/datasets/wikimedia/wikipedia",
                    approx_mb([""]*n), "Broad Wikipedia biology keyword scan", n)
      total += n; log.info(f"  [BIO-2] Wikipedia stream: {n} docs")

      n = stackexchange_qa(["biology","bioinformatics"], bucket, "bio_se", max_docs=4000)
      record_source("corpus_C_technical", "Biology StackExchange Q&A",
                    "https://biology.stackexchange.com", approx_mb([""]*n),
                    "Expert biology Q&A: molecular bio, genetics, physiology", n)
      total += n; log.info(f"  [BIO-3] StackExchange: {n} docs")

      n = arxiv_filtered(
          ["q-bio.BM","q-bio.CB","q-bio.GN","q-bio.NC","q-bio.PE","q-bio.TO"],
          bucket, "bio_arxiv", max_docs=4000)
      record_source("corpus_C_technical", "arXiv Biology Papers",
                    "https://arxiv.org", approx_mb([""]*n),
                    "Biology research: biomolecules, cell behavior, genomics, neuroscience", n)
      total += n; log.info(f"  [BIO-4] arXiv: {n} docs")

      n = gutenberg_load([
          (1228,  "The Voyage of the Beagle — Darwin"),
          (2009,  "The Origin of Species — Darwin"),
          (28846, "The Descent of Man — Darwin"),
          (14363, "The Expression of Emotions in Man and Animals — Darwin"),
          (4461,  "The Formation of Vegetable Mould — Darwin"),
          (17147, "The Life of the Bee — Maeterlinck"),
          (3818,  "The Study of the Cell — Wilson"),
          (38262, "The Physical Basis of Life — Huxley"),
          (38775, "Evolution and Ethics — Huxley"),
          (17510, "Heredity and Environment — Morgan"),
      ], bucket, "bio_gutenberg")
      record_source("corpus_C_technical", "Gutenberg Biology Textbooks",
                    "https://www.gutenberg.org", approx_mb([""]*n),
                    "Darwin, Huxley, Morgan, Wilson — foundational biology texts", n)
      total += n; log.info(f"  [BIO-5] Gutenberg: {n} docs")

      n = openalex_fetch([
          "CRISPR genome editing therapeutic",
          "single cell RNA sequencing transcriptomics",
          "protein structure prediction AlphaFold",
          "microbiome gut bacteria health",
          "cancer immunotherapy checkpoint inhibitor",
          "synthetic biology genetic circuit design",
          "evolutionary biology population genomics",
          "neuroscience synaptic plasticity memory",
          "epigenetics chromatin regulation",
          "stem cell differentiation organoid",
      ], bucket, "bio_openalex", label="Biology Research")
      record_source("corpus_C_technical", "OpenAlex Biology Papers",
                    "https://openalex.org", approx_mb([""]*n),
                    "Open-access biology research papers", n)
      total += n; log.info(f"  [BIO-6] OpenAlex: {n} docs")

      log.info(f"  >>> Biology total: {total:,} docs")
      return total


  # ══════════════════════════════════════════════════════════════════════════════
  # DOMAIN 3 — ADVANCED MATHEMATICS
  # ══════════════════════════════════════════════════════════════════════════════
  def load_advanced_mathematics():
      log.info("=" * 65)
      log.info("DOMAIN 3: Advanced Mathematics")
      log.info("=" * 65)
      bucket = RAW / "corpus_C_technical"
      total  = 0

      wiki_queries_math = [
          "real analysis continuity uniform convergence",
          "complex analysis holomorphic functions Cauchy theorem",
          "measure theory Lebesgue integration",
          "functional analysis Hilbert Banach spaces",
          "differential geometry Riemannian manifolds",
          "algebraic topology homology cohomology",
          "abstract algebra group theory ring field",
          "Galois theory polynomial equations",
          "linear algebra eigenvalues singular value decomposition",
          "ordinary differential equations phase portrait",
          "partial differential equations heat wave Laplace",
          "Fourier analysis transform series",
          "number theory prime numbers modular arithmetic",
          "Riemann hypothesis zeta function",
          "combinatorics graph theory enumeration",
          "probability theory measure theoretic foundations",
          "stochastic processes Brownian motion martingales",
          "Bayesian statistics inference",
          "numerical methods finite element analysis",
          "convex optimization gradient methods",
          "dynamical systems chaos bifurcation",
          "topology metric spaces compactness",
          "category theory functors natural transformations",
          "set theory axioms Cantor cardinals",
          "mathematical logic Godel incompleteness",
          "cryptography number theory applications",
          "information theory entropy Shannon",
          "game theory Nash equilibrium",
          "statistics regression hypothesis testing",
          "calculus of variations Euler-Lagrange",
      ]

      n = wiki_api_targeted(wiki_queries_math, bucket, "advmath_wiki_api", max_per_query=20)
      record_source("corpus_C_technical", "Wikipedia Mathematics API (targeted)",
                    "https://en.wikipedia.org", approx_mb([""]*n),
                    "Targeted Wikipedia: analysis, algebra, topology, number theory", n)
      total += n; log.info(f"  [MATH-1] Wikipedia API: {n} docs")

      n = wiki_stream_filtered(MATH_KW, bucket, "advmath_wiki_stream", max_docs=5000)
      record_source("corpus_C_technical", "Wikipedia Mathematics Stream",
                    "https://huggingface.co/datasets/wikimedia/wikipedia",
                    approx_mb([""]*n), "Broad Wikipedia mathematics keyword scan", n)
      total += n; log.info(f"  [MATH-2] Wikipedia stream: {n} docs")

      n = stackexchange_qa(["math","mathematics"], bucket, "advmath_se", max_docs=5000)
      record_source("corpus_C_technical", "Math StackExchange Q&A",
                    "https://math.stackexchange.com", approx_mb([""]*n),
                    "Expert math Q&A: proofs, analysis, algebra, geometry, statistics", n)
      total += n; log.info(f"  [MATH-3] StackExchange: {n} docs")

      n = arxiv_filtered(
          ["math.CA","math.NT","math.GR","math.PR","math.ST",
           "math.CO","math.AP","math.NA","math.DG","math.AG"],
          bucket, "advmath_arxiv", max_docs=4000)
      record_source("corpus_C_technical", "arXiv Mathematics Papers",
                    "https://arxiv.org", approx_mb([""]*n),
                    "Math research: analysis, number theory, probability, combinatorics", n)
      total += n; log.info(f"  [MATH-4] arXiv: {n} docs")

      n = gutenberg_load([
          (33283, "A Course of Pure Mathematics — Hardy"),
          (5000,  "Euclid's Elements — Euclid"),
          (13692, "An Introduction to Mathematics — Whitehead"),
          (9296,  "A Brief History of Mathematics — Cajori"),
          (36640, "Flatland: A Romance of Many Dimensions — Abbott"),
          (17384, "A Primer of Higher Mathematics"),
          (35286, "Lectures on Fundamental Concepts of Algebra — Hensel"),
          (31246, "Introduction to Infinitesimal Analysis — Courant"),
          (38769, "A Treatise on Electricity and Magnetism Vol 1 — Maxwell"),
          (28233, "Science and Hypothesis — Poincare"),
      ], bucket, "advmath_gutenberg")
      record_source("corpus_C_technical", "Gutenberg Mathematics Textbooks",
                    "https://www.gutenberg.org", approx_mb([""]*n),
                    "Hardy, Euclid, Whitehead, Poincaré, Cajori foundational texts", n)
      total += n; log.info(f"  [MATH-5] Gutenberg: {n} docs")

      log.info(f"  >>> Advanced Mathematics total: {total:,} docs")
      return total


  # ══════════════════════════════════════════════════════════════════════════════
  # DOMAIN 4 — ADVANCED PHYSICS
  # ══════════════════════════════════════════════════════════════════════════════
  def load_advanced_physics():
      log.info("=" * 65)
      log.info("DOMAIN 4: Advanced Physics")
      log.info("=" * 65)
      bucket = RAW / "corpus_C_technical"
      total  = 0

      wiki_queries_phys = [
          "Lagrangian Hamiltonian classical mechanics",
          "special relativity Lorentz transformation",
          "general relativity Einstein field equations",
          "quantum mechanics Schrodinger equation",
          "quantum entanglement Bell inequalities",
          "quantum field theory renormalization",
          "Standard Model particle physics",
          "Higgs mechanism electroweak unification",
          "quantum chromodynamics strong force quarks",
          "condensed matter physics band theory",
          "superconductivity BCS theory Cooper pairs",
          "topological insulators quantum Hall effect",
          "statistical mechanics thermodynamic ensembles",
          "Bose-Einstein condensate quantum gases",
          "plasma physics magnetohydrodynamics",
          "nuclear physics radioactive decay fission fusion",
          "astrophysics stellar nucleosynthesis",
          "black hole thermodynamics Hawking radiation",
          "gravitational waves LIGO detection",
          "dark matter dark energy cosmology",
          "cosmic microwave background inflation",
          "optics wave diffraction interference",
          "laser physics stimulated emission",
          "nonlinear dynamics chaos Lorenz attractor",
          "string theory M-theory extra dimensions",
          "loop quantum gravity spin foam",
          "Maxwell's equations electromagnetic waves",
          "solid state physics crystal structures",
          "semiconductors p-n junction band gap",
          "neutron star pulsar magnetar physics",
      ]

      n = wiki_api_targeted(wiki_queries_phys, bucket, "advphys_wiki_api", max_per_query=20)
      record_source("corpus_C_technical", "Wikipedia Physics API (targeted)",
                    "https://en.wikipedia.org", approx_mb([""]*n),
                    "Targeted Wikipedia: quantum, relativity, condensed matter, astrophysics", n)
      total += n; log.info(f"  [PHYS-1] Wikipedia API: {n} docs")

      n = wiki_stream_filtered(PHYSICS_KW, bucket, "advphys_wiki_stream", max_docs=5000)
      record_source("corpus_C_technical", "Wikipedia Physics Stream",
                    "https://huggingface.co/datasets/wikimedia/wikipedia",
                    approx_mb([""]*n), "Broad Wikipedia physics keyword scan", n)
      total += n; log.info(f"  [PHYS-2] Wikipedia stream: {n} docs")

      n = stackexchange_qa(["physics"], bucket, "advphys_se", max_docs=5000)
      record_source("corpus_C_technical", "Physics StackExchange Q&A",
                    "https://physics.stackexchange.com", approx_mb([""]*n),
                    "Expert physics Q&A: QM, relativity, condensed matter, astrophysics", n)
      total += n; log.info(f"  [PHYS-3] StackExchange: {n} docs")

      n = arxiv_filtered(
          ["physics.class-ph","quant-ph","cond-mat.str-el","cond-mat.supr-con",
           "astro-ph.SR","astro-ph.GA","astro-ph.CO","hep-th","hep-ph",
           "physics.optics","physics.atom-ph","physics.plasm-ph"],
          bucket, "advphys_arxiv", max_docs=5000)
      record_source("corpus_C_technical", "arXiv Physics Papers",
                    "https://arxiv.org", approx_mb([""]*n),
                    "Physics research: QM, condensed matter, astrophysics, HEP", n)
      total += n; log.info(f"  [PHYS-4] arXiv: {n} docs")

      n = gutenberg_load([
          (5001,  "Principia Mathematica — Newton"),
          (42933, "Relativity: Special & General Theory — Einstein"),
          (28233, "Science and Hypothesis — Poincare"),
          (26839, "The Theory of Heat — Maxwell"),
          (33642, "Matter and Motion — Maxwell"),
          (38769, "A Treatise on Electricity and Magnetism — Maxwell"),
          (9181,  "On the Various Forces of Nature — Faraday"),
          (1202,  "The Law of Thermodynamics — Boltzmann"),
          (20417, "Modern Physics — Rutherford"),
          (55148, "Experimental Researches in Electricity — Faraday"),
      ], bucket, "advphys_gutenberg")
      record_source("corpus_C_technical", "Gutenberg Physics Textbooks",
                    "https://www.gutenberg.org", approx_mb([""]*n),
                    "Newton, Einstein, Maxwell, Faraday, Boltzmann foundational texts", n)
      total += n; log.info(f"  [PHYS-5] Gutenberg: {n} docs")

      # NIST physics data
      try:
          log.info("  [PHYS-6] NIST physics reference data...")
          docs = []
          nist_endpoints = [
              ("https://physics.nist.gov/cgi-bin/cuu/Results?search_for=all", "NIST Physical Constants"),
          ]
          r = requests.get(
              "https://www.osti.gov/api/v1/records?q=quantum+mechanics+condensed+matter&page=0&size=200",
              timeout=20
          )
          if r.status_code == 200:
              for rec in r.json().get("records", []):
                  title    = rec.get("title", "").strip()
                  abstract = rec.get("description", "").strip()
                  if abstract and len(abstract) > 100:
                      docs.append(f"Physics Technical Report\n\nTitle: {title}\n\n{abstract}")
          n = write_docs(bucket, docs, "advphys_nist_osti")
          if n > 0:
              record_source("corpus_C_technical", "NIST/OSTI Physics Reports",
                            "https://www.osti.gov", approx_mb(docs),
                            "DOE physics research reports from OSTI", n)
          total += n; log.info(f"  [PHYS-6] NIST/OSTI: {n} docs")
      except Exception as e:
          log.error(f"  [PHYS-6] NIST/OSTI failed: {e}")

      log.info(f"  >>> Advanced Physics total: {total:,} docs")
      return total


  # ══════════════════════════════════════════════════════════════════════════════
  # DOMAIN 5 — MEDICINE (WESTERN + EASTERN)
  # ══════════════════════════════════════════════════════════════════════════════
  def load_medicine():
      log.info("=" * 65)
      log.info("DOMAIN 5: Medicine — Western & Eastern")
      log.info("=" * 65)
      bucket = RAW / "corpus_C_technical"
      total  = 0

      # ── 5A: Western Medicine ──────────────────────────────────────────────────
      wiki_queries_med_west = [
          "cardiology myocardial infarction treatment",
          "heart failure pathophysiology management",
          "neurology stroke ischemic hemorrhagic",
          "Parkinson disease dopamine treatment",
          "Alzheimer disease neurodegeneration biomarkers",
          "oncology cancer treatment chemotherapy",
          "immunotherapy checkpoint inhibitors cancer",
          "diabetes mellitus insulin therapy",
          "hypertension antihypertensive drugs mechanisms",
          "pneumonia respiratory infection antibiotics",
          "asthma COPD pathophysiology treatment",
          "liver cirrhosis hepatic failure management",
          "inflammatory bowel disease Crohn colitis",
          "autoimmune disease immunology pathogenesis",
          "vaccine immunization mechanism immunity",
          "antibiotic resistance mechanisms treatment",
          "sepsis systemic inflammatory response",
          "surgery anesthesia intraoperative monitoring",
          "medical imaging MRI CT contrast agents",
          "pharmacology drug receptor interaction",
          "clinical pharmacokinetics drug metabolism",
          "evidence-based medicine clinical trials",
          "emergency medicine trauma resuscitation",
          "pediatrics childhood diseases vaccination",
          "obstetrics pregnancy complications",
          "psychiatry depression treatment SSRIs",
          "schizophrenia antipsychotic mechanisms",
          "radiology interventional procedures",
          "orthopedics fracture healing bone repair",
          "nephrology renal failure dialysis",
          "hematology blood disorders coagulation",
          "endocrinology thyroid adrenal disorders",
          "infectious disease HIV antiretroviral",
          "dermatology skin disease treatment",
          "ophthalmology eye disease treatment",
      ]

      n = wiki_api_targeted(wiki_queries_med_west, bucket, "med_west_wiki_api", max_per_query=20)
      record_source("corpus_C_technical", "Wikipedia Western Medicine API",
                    "https://en.wikipedia.org", approx_mb([""]*n),
                    "Targeted: cardiology, neurology, oncology, pharmacology, surgery", n)
      total += n; log.info(f"  [MED-1] West Wiki API: {n} docs")

      n = wiki_stream_filtered(MED_WESTERN_KW, bucket, "med_west_wiki_stream", max_docs=4000)
      record_source("corpus_C_technical", "Wikipedia Western Medicine Stream",
                    "https://huggingface.co/datasets/wikimedia/wikipedia",
                    approx_mb([""]*n), "Broad Wikipedia western medicine scan", n)
      total += n; log.info(f"  [MED-2] West Wiki stream: {n} docs")

      # ── 5B: Eastern & Integrative Medicine ───────────────────────────────────
      wiki_queries_med_east = [
          "traditional chinese medicine TCM theory",
          "acupuncture meridians clinical evidence",
          "Chinese herbal medicine pharmacology",
          "qi gong tai chi health benefits",
          "Ayurveda doshas treatment principles",
          "Ayurvedic herbs ashwagandha turmeric evidence",
          "Panchakarma detoxification therapy",
          "yoga therapy therapeutic applications",
          "Unani medicine tibb humoral theory",
          "Kampo Japanese traditional medicine",
          "Korean traditional medicine hanbang",
          "Siddha medicine Tamil Nadu tradition",
          "naturopathy botanical medicine evidence",
          "homeopathy principles clinical trials",
          "traditional African medicine ethnobotany",
          "Native American healing practices",
          "phytotherapy herbal medicine pharmacology",
          "adaptogen herbs stress response",
          "integrative medicine complementary therapy",
          "mind-body medicine meditation neuroscience",
          "acupressure reflexology evidence base",
          "traditional medicine WHO guidelines",
          "ethnopharmacology plant medicine compounds",
          "folk medicine remedies evidence review",
          "Tibetan medicine Buddhist healing",
      ]

      n = wiki_api_targeted(wiki_queries_med_east, bucket, "med_east_wiki_api", max_per_query=20)
      record_source("corpus_C_technical", "Wikipedia Eastern Medicine API",
                    "https://en.wikipedia.org", approx_mb([""]*n),
                    "TCM, Ayurveda, Unani, Kampo, Siddha, integrative medicine", n)
      total += n; log.info(f"  [MED-3] East Wiki API: {n} docs")

      n = wiki_stream_filtered(MED_EASTERN_KW, bucket, "med_east_wiki_stream", max_docs=3000)
      record_source("corpus_C_technical", "Wikipedia Eastern Medicine Stream",
                    "https://huggingface.co/datasets/wikimedia/wikipedia",
                    approx_mb([""]*n), "Broad Wikipedia eastern/integrative medicine scan", n)
      total += n; log.info(f"  [MED-4] East Wiki stream: {n} docs")

      n = stackexchange_qa(["medical","health","biology"],
                           bucket, "med_se",
                           extra_kw_filter={
                               "disease","treatment","drug","symptom","diagnosis",
                               "medicine","clinical","patient","therapy","anatomy",
                               "pharmacology","surgery","cancer","heart","brain",
                           }, max_docs=3000)
      record_source("corpus_C_technical", "Medical StackExchange Q&A",
                    "https://health.stackexchange.com", approx_mb([""]*n),
                    "Medical Q&A: clinical questions, pharmacology, disease mechanisms", n)
      total += n; log.info(f"  [MED-5] StackExchange: {n} docs")

      n = arxiv_filtered(
          ["q-bio.TO","q-bio.NC","q-bio.CB","q-bio.BM"],
          bucket, "med_arxiv",
          extra_kw={"disease","treatment","clinical","medical","patient",
                    "therapy","drug","cancer","neural","brain"},
          max_docs=3000)
      record_source("corpus_C_technical", "arXiv Medical/Biology Papers",
                    "https://arxiv.org", approx_mb([""]*n),
                    "Medical research: tissues, neuroscience, cell biology", n)
      total += n; log.info(f"  [MED-6] arXiv: {n} docs")

      n = gutenberg_load([
          (12026, "Gray's Anatomy — Gray"),
          (15576, "The Human Body — Hutchinson"),
          (4949,  "The Science of Human Nature — Pyle"),
          (16522, "Manual of Surgery — Thomson"),
          (38877, "The Practice of Medicine — Osler"),
          (24746, "The Chemistry of Plant Life — Miller"),
          (25163, "Handbook of Medical Entomology"),
          (9728,  "Personal Memoirs of a Physician — Hahnemann"),
          (15583, "The Art of Perfumery — Piesse"),
          (18492, "Folk Medicine — Vogel"),
      ], bucket, "med_gutenberg")
      record_source("corpus_C_technical", "Gutenberg Medicine Textbooks",
                    "https://www.gutenberg.org", approx_mb([""]*n),
                    "Gray's Anatomy, Osler, Surgery, folk/herbal medicine texts", n)
      total += n; log.info(f"  [MED-7] Gutenberg: {n} docs")

      n = openalex_fetch([
          "traditional Chinese medicine acupuncture randomized trial",
          "Ayurveda herbal medicine clinical evidence",
          "integrative medicine cancer complementary therapy",
          "mindfulness meditation clinical outcomes",
          "immunotherapy checkpoint inhibitor clinical trial",
          "precision medicine genomics personalized treatment",
          "antibiotic resistance mechanisms clinical",
          "diabetes type 2 management outcomes",
          "cardiovascular disease prevention treatment",
          "neurodegenerative disease biomarkers treatment",
      ], bucket, "med_openalex", label="Medical Research")
      record_source("corpus_C_technical", "OpenAlex Medical Papers",
                    "https://openalex.org", approx_mb([""]*n),
                    "Open-access medical research: clinical trials, treatment outcomes", n)
      total += n; log.info(f"  [MED-8] OpenAlex: {n} docs")

      log.info(f"  >>> Medicine total: {total:,} docs")
      return total


  # ══════════════════════════════════════════════════════════════════════════════
  # DOMAIN 6 — ADVANCED ELECTRICAL ENGINEERING
  # ══════════════════════════════════════════════════════════════════════════════
  def load_advanced_electrical_engineering():
      log.info("=" * 65)
      log.info("DOMAIN 6: Advanced Electrical Engineering")
      log.info("=" * 65)
      bucket = RAW / "corpus_C_technical"
      total  = 0

      wiki_queries_adv_ee = [
          "power system analysis load flow Newton-Raphson",
          "electric power transmission grid stability",
          "synchronous generator excitation control",
          "induction motor speed torque characteristics",
          "power electronics converter topologies",
          "high voltage engineering insulation breakdown",
          "electromagnetic compatibility EMC shielding",
          "RF microwave circuit design Smith chart",
          "antenna design radiation pattern gain",
          "optical fiber communication WDM",
          "laser semiconductor physics diode",
          "analog IC design operational amplifier",
          "CMOS digital circuit design",
          "phase locked loop frequency synthesis",
          "analog digital converter architecture",
          "signal integrity PCB high speed",
          "power quality harmonic distortion",
          "electric vehicle battery management system",
          "smart grid energy management systems",
          "control theory state space methods",
          "digital signal processing filter banks",
          "radar signal processing Doppler",
          "satellite communication link budget",
          "5G NR radio access technology",
          "photovoltaic solar cell efficiency",
          "electric machine design optimization",
      ]

      n = wiki_api_targeted(wiki_queries_adv_ee, bucket, "advee_wiki_api", max_per_query=20)
      record_source("corpus_C_technical", "Wikipedia Advanced EE API",
                    "https://en.wikipedia.org", approx_mb([""]*n),
                    "Power systems, RF, photonics, analog IC, power electronics depth", n)
      total += n; log.info(f"  [ADV-EE-1] Wiki API: {n} docs")

      n = wiki_stream_filtered(ADV_EE_KW, bucket, "advee_wiki_stream", max_docs=4000)
      record_source("corpus_C_technical", "Wikipedia Advanced EE Stream",
                    "https://huggingface.co/datasets/wikimedia/wikipedia",
                    approx_mb([""]*n), "Deep EE keyword stream: RF, power, photonics", n)
      total += n; log.info(f"  [ADV-EE-2] Wiki stream: {n} docs")

      n = stackexchange_qa(["electronics","electrical"],
                           bucket, "advee_se",
                           extra_kw_filter={
                               "power","rf","antenna","high voltage","signal integrity",
                               "mosfet","igbt","converter","inverter","filter","amplifier",
                               "noise","impedance","transmission line","microwave","pll",
                           }, max_docs=3000)
      record_source("corpus_C_technical", "Advanced EE StackExchange Q&A",
                    "https://electronics.stackexchange.com", approx_mb([""]*n),
                    "Advanced EE Q&A: power electronics, RF design, signal integrity", n)
      total += n; log.info(f"  [ADV-EE-3] StackExchange: {n} docs")

      n = arxiv_filtered(
          ["eess.SP","eess.PE","eess.SY","eess.IV","cs.SY"],
          bucket, "advee_arxiv", max_docs=3000)
      record_source("corpus_C_technical", "arXiv Advanced EE Papers",
                    "https://arxiv.org", approx_mb([""]*n),
                    "EE research: signal processing, power electronics, control", n)
      total += n; log.info(f"  [ADV-EE-4] arXiv: {n} docs")

      n = openalex_fetch([
          "wide bandgap semiconductor power electronics",
          "machine learning power system stability",
          "electric vehicle fast charging grid",
          "millimeter wave 5G antenna design",
          "neural network control adaptive",
          "photonic integrated circuit silicon",
          "digital twin power grid simulation",
          "high frequency transformer design",
          "quantum computing superconducting qubit",
          "energy harvesting piezoelectric wireless",
      ], bucket, "advee_openalex", label="EE Research")
      record_source("corpus_C_technical", "OpenAlex Advanced EE Papers",
                    "https://openalex.org", approx_mb([""]*n),
                    "Open-access advanced EE research papers", n)
      total += n; log.info(f"  [ADV-EE-5] OpenAlex: {n} docs")

      log.info(f"  >>> Advanced EE total: {total:,} docs")
      return total


  # ══════════════════════════════════════════════════════════════════════════════
  # DOMAIN 7 — ADVANCED MECHANICAL ENGINEERING
  # ══════════════════════════════════════════════════════════════════════════════
  def load_advanced_mechanical_engineering():
      log.info("=" * 65)
      log.info("DOMAIN 7: Advanced Mechanical Engineering")
      log.info("=" * 65)
      bucket = RAW / "corpus_C_technical"
      total  = 0

      wiki_queries_adv_me = [
          "fracture mechanics stress intensity factor",
          "fatigue failure crack propagation Paris law",
          "creep high temperature deformation",
          "computational fluid dynamics turbulence models",
          "boundary layer separation flow control",
          "compressible aerodynamics shock waves",
          "combustion engine thermodynamics Otto Diesel",
          "gas turbine Brayton cycle efficiency",
          "heat exchanger design NTU effectiveness",
          "advanced materials titanium superalloy properties",
          "composite materials carbon fiber mechanics",
          "additive manufacturing metal 3D printing",
          "topology optimization structural design",
          "tribology lubrication wear mechanisms",
          "vibration control active passive isolation",
          "acoustic noise reduction enclosure design",
          "precision engineering tolerances metrology",
          "corrosion mechanisms electrochemical",
          "MEMS fabrication silicon micromachining",
          "nano-manufacturing surface roughness",
          "reliability engineering Weibull failure analysis",
          "condition monitoring predictive maintenance",
          "aeroelasticity flutter aircraft structures",
          "wind tunnel aerodynamic testing",
          "fuel cell hydrogen energy conversion",
      ]

      n = wiki_api_targeted(wiki_queries_adv_me, bucket, "advme_wiki_api", max_per_query=20)
      record_source("corpus_C_technical", "Wikipedia Advanced ME API",
                    "https://en.wikipedia.org", approx_mb([""]*n),
                    "Fracture, CFD, combustion, advanced materials, tribology, MEMS", n)
      total += n; log.info(f"  [ADV-ME-1] Wiki API: {n} docs")

      n = wiki_stream_filtered(ADV_ME_KW, bucket, "advme_wiki_stream", max_docs=4000)
      record_source("corpus_C_technical", "Wikipedia Advanced ME Stream",
                    "https://huggingface.co/datasets/wikimedia/wikipedia",
                    approx_mb([""]*n), "Deep ME keyword stream: fracture, CFD, thermo", n)
      total += n; log.info(f"  [ADV-ME-2] Wiki stream: {n} docs")

      n = stackexchange_qa(["engineering","physics","mechanics"],
                           bucket, "advme_se",
                           extra_kw_filter={
                               "fracture","fatigue","cfd","turbulence","combustion",
                               "composite","creep","tribology","bearing","vibration",
                               "stress","strain","heat transfer","fluid","aerodynamic",
                           }, max_docs=3000)
      record_source("corpus_C_technical", "Advanced ME StackExchange Q&A",
                    "https://engineering.stackexchange.com", approx_mb([""]*n),
                    "Advanced ME Q&A: fracture mechanics, CFD, composites, tribology", n)
      total += n; log.info(f"  [ADV-ME-3] StackExchange: {n} docs")

      n = arxiv_filtered(
          ["physics.flu-dyn","cond-mat.mtrl-sci","cond-mat.soft",
           "physics.app-ph","cs.CE"],
          bucket, "advme_arxiv", max_docs=3000)
      record_source("corpus_C_technical", "arXiv Advanced ME Papers",
                    "https://arxiv.org", approx_mb([""]*n),
                    "ME research: fluid dynamics, materials science, applied physics", n)
      total += n; log.info(f"  [ADV-ME-4] arXiv: {n} docs")

      n = openalex_fetch([
          "topology optimization additive manufacturing",
          "machine learning turbulence prediction CFD",
          "high entropy alloy mechanical properties",
          "hydrogen combustion engine emissions",
          "digital twin manufacturing process",
          "carbon fiber composite structural health",
          "tribology surface texture lubrication",
          "MEMS sensor fabrication characterization",
          "aerodynamic shape optimization aircraft",
          "predictive maintenance vibration analysis",
      ], bucket, "advme_openalex", label="ME Research")
      record_source("corpus_C_technical", "OpenAlex Advanced ME Papers",
                    "https://openalex.org", approx_mb([""]*n),
                    "Open-access advanced mechanical engineering research", n)
      total += n; log.info(f"  [ADV-ME-5] OpenAlex: {n} docs")

      log.info(f"  >>> Advanced ME total: {total:,} docs")
      return total


  # ══════════════════════════════════════════════════════════════════════════════
  # DOMAIN 8 — SYSTEMS DESIGN (ONLINE / DIGITAL)
  # ══════════════════════════════════════════════════════════════════════════════
  def load_systems_design_online():
      log.info("=" * 65)
      log.info("DOMAIN 8: Systems Design — Online / Digital")
      log.info("=" * 65)
      bucket = RAW / "corpus_C_technical"
      total  = 0

      wiki_queries_sys_online = [
          "digital logic design combinational circuits",
          "FPGA architecture look-up table routing",
          "ASIC design flow RTL synthesis",
          "microprocessor pipeline architecture",
          "cache memory hierarchy coherence",
          "out-of-order execution speculative",
          "embedded systems real-time scheduling",
          "RTOS FreeRTOS interrupt latency",
          "TCP/IP protocol suite networking",
          "software defined networking OpenFlow",
          "distributed systems consensus Raft Paxos",
          "CAP theorem consistency availability",
          "microservices architecture service mesh",
          "event driven architecture Kafka",
          "control theory PID tuning Ziegler-Nichols",
          "model predictive control optimization",
          "digital signal processing FFT algorithms",
          "FIR IIR filter design windowing",
          "OFDM multicarrier modulation 5G LTE",
          "MIMO beamforming antenna arrays",
          "power electronics buck boost converter",
          "PWM modulation switching frequency",
          "software architecture design patterns",
          "database design normalization indexing",
          "fault tolerant systems redundancy",
      ]

      n = wiki_api_targeted(wiki_queries_sys_online, bucket, "sys_on_wiki_api", max_per_query=20)
      record_source("corpus_C_technical", "Wikipedia Systems Online API",
                    "https://en.wikipedia.org", approx_mb([""]*n),
                    "FPGA, ASIC, distributed, control, DSP, comms systems design", n)
      total += n; log.info(f"  [SYS-ON-1] Wiki API: {n} docs")

      n = wiki_stream_filtered(SYS_ONLINE_KW, bucket, "sys_on_wiki_stream", max_docs=3000)
      record_source("corpus_C_technical", "Wikipedia Systems Online Stream",
                    "https://huggingface.co/datasets/wikimedia/wikipedia",
                    approx_mb([""]*n), "Digital systems keyword stream", n)
      total += n; log.info(f"  [SYS-ON-2] Wiki stream: {n} docs")

      n = stackexchange_qa(["electronics","cs","softwareengineering","dsp"],
                           bucket, "sys_on_se",
                           extra_kw_filter={
                               "fpga","verilog","vhdl","pipeline","cache","rtos",
                               "distributed","consensus","microservices","filter",
                               "controller","pid","bode","nyquist","ofdm","mimo",
                           }, max_docs=3000)
      record_source("corpus_C_technical", "Systems Online StackExchange Q&A",
                    "https://electronics.stackexchange.com", approx_mb([""]*n),
                    "Digital & online systems Q&A: FPGA, RTOS, distributed, DSP", n)
      total += n; log.info(f"  [SYS-ON-3] StackExchange: {n} docs")

      n = arxiv_filtered(
          ["cs.SY","eess.SY","cs.DC","cs.NI","cs.AR","cs.PF"],
          bucket, "sys_on_arxiv", max_docs=3000)
      record_source("corpus_C_technical", "arXiv Systems Online Papers",
                    "https://arxiv.org", approx_mb([""]*n),
                    "Systems research: control, distributed, networking, architecture", n)
      total += n; log.info(f"  [SYS-ON-4] arXiv: {n} docs")

      log.info(f"  >>> Systems Design Online total: {total:,} docs")
      return total


  # ══════════════════════════════════════════════════════════════════════════════
  # DOMAIN 9 — SYSTEMS DESIGN (MECHANICAL)
  # ══════════════════════════════════════════════════════════════════════════════
  def load_systems_design_mechanical():
      log.info("=" * 65)
      log.info("DOMAIN 9: Systems Design — Mechanical")
      log.info("=" * 65)
      bucket = RAW / "corpus_C_technical"
      total  = 0

      wiki_queries_sys_mech = [
          "mechanism design four-bar linkage synthesis",
          "gear design involute profile bending stress",
          "rolling element bearing selection life",
          "shaft design bending torsion combined loading",
          "bolted joint preload fatigue",
          "weld design throat area fillet",
          "hydraulic system pump valve actuator",
          "proportional directional control valve",
          "pneumatic system compressor circuit design",
          "heat exchanger design shell tube plate",
          "refrigeration cycle COP optimization",
          "HVAC system psychrometric chart",
          "manufacturing process selection casting forging",
          "CNC machining tool path programming",
          "lean manufacturing value stream mapping",
          "six sigma DMAIC quality control",
          "mechatronics servo system design",
          "vibration isolation mount design",
          "composite laminate structural analysis",
          "tribology surface roughness friction",
          "corrosion protection coating systems",
          "precision assembly tolerance stack-up GD&T",
      ]

      n = wiki_api_targeted(wiki_queries_sys_mech, bucket, "sys_me_wiki_api", max_per_query=20)
      record_source("corpus_C_technical", "Wikipedia Systems Mech API",
                    "https://en.wikipedia.org", approx_mb([""]*n),
                    "Mechanism design, hydraulics, thermal, manufacturing systems", n)
      total += n; log.info(f"  [SYS-ME-1] Wiki API: {n} docs")

      n = wiki_stream_filtered(SYS_MECH_KW, bucket, "sys_me_wiki_stream", max_docs=3000)
      record_source("corpus_C_technical", "Wikipedia Systems Mech Stream",
                    "https://huggingface.co/datasets/wikimedia/wikipedia",
                    approx_mb([""]*n), "Mechanical systems keyword stream", n)
      total += n; log.info(f"  [SYS-ME-2] Wiki stream: {n} docs")

      n = stackexchange_qa(["engineering","physics"],
                           bucket, "sys_me_se",
                           extra_kw_filter={
                               "gear","bearing","shaft","hydraulic","pneumatic",
                               "heat exchanger","hvac","mechanism","linkage",
                               "manufacturing","cnc","tolerance","composite","servo",
                           }, max_docs=3000)
      record_source("corpus_C_technical", "Systems Mech StackExchange Q&A",
                    "https://engineering.stackexchange.com", approx_mb([""]*n),
                    "Mechanical systems Q&A: mechanism, hydraulics, thermal, manufacturing", n)
      total += n; log.info(f"  [SYS-ME-3] StackExchange: {n} docs")

      n = arxiv_filtered(
          ["cs.CE","physics.class-ph","physics.flu-dyn","cond-mat.mtrl-sci"],
          bucket, "sys_me_arxiv",
          extra_kw={"mechanism","hydraulic","thermal","manufacturing","design",
                    "optimization","composite","bearing","gear"},
          max_docs=2000)
      record_source("corpus_C_technical", "arXiv Systems Mech Papers",
                    "https://arxiv.org", approx_mb([""]*n),
                    "Mechanical systems research papers", n)
      total += n; log.info(f"  [SYS-ME-4] arXiv: {n} docs")

      n = gutenberg_load([
          (20218, "Machine Design: Kinematics of Machinery"),
          (25864, "Hydraulics and Fluid Mechanics"),
          (41568, "Heat Engines — Steam & Internal Combustion"),
          (40780, "Materials of Construction"),
          (38876, "Thermodynamics — Bryan"),
          (20417, "Practical Mechanics for Boys"),
          (14725, "The Mechanical Properties of Fluids"),
          (33272, "The Steam Engine — Rankine"),
          (4367,  "Engineering Descriptive Geometry"),
          (21352, "Principles of Scientific Management — Taylor"),
      ], bucket, "sys_me_gutenberg")
      record_source("corpus_C_technical", "Gutenberg Mechanical Systems Textbooks",
                    "https://www.gutenberg.org", approx_mb([""]*n),
                    "Machine design, hydraulics, heat engines, materials, Taylor", n)
      total += n; log.info(f"  [SYS-ME-5] Gutenberg: {n} docs")

      log.info(f"  >>> Systems Design Mechanical total: {total:,} docs")
      return total


  # ══════════════════════════════════════════════════════════════════════════════
  # MAIN
  # ══════════════════════════════════════════════════════════════════════════════
  def main():
      log.info("TitanAI Premium Science Corpus Loader — Starting")
      log.info("9 Domains: Chemistry | Biology | Mathematics | Physics | Medicine")
      log.info("           Adv-EE | Adv-ME | Systems-Online | Systems-Mech")
      log.info(f"Target directory: {RAW}")
      log.info("")

      results = {}
      results["advanced_chemistry"]         = load_advanced_chemistry()
      results["biology"]                    = load_biology()
      results["advanced_mathematics"]       = load_advanced_mathematics()
      results["advanced_physics"]           = load_advanced_physics()
      results["medicine_east_west"]         = load_medicine()
      results["advanced_electrical_eng"]    = load_advanced_electrical_engineering()
      results["advanced_mechanical_eng"]    = load_advanced_mechanical_engineering()
      results["systems_design_online"]      = load_systems_design_online()
      results["systems_design_mechanical"]  = load_systems_design_mechanical()

      # Persist
      with open(INVENTORY, "w") as f:
          json.dump(inventory, f, indent=2)
      log.info(f"Source inventory saved: {INVENTORY}")
      with open(EXCLUSION_LOG, "w") as f:
          for exc in exclusions:
              f.write(json.dumps(exc) + "\n")
      log.info(f"Exclusion log: {EXCLUSION_LOG}")

      log.info("")
      log.info("=" * 65)
      log.info("PREMIUM SCIENCE CORPUS LOAD — COMPLETE")
      log.info("=" * 65)
      grand = 0
      for domain, n in results.items():
          log.info(f"  {domain:<35}: {n:>7,} docs")
          grand += n
      log.info(f"  {'GRAND TOTAL':<35}: {grand:>7,} docs")
      log.info("")
      log.info("Next steps on instance:")
      log.info("  python scripts/rebalance_corpus.py")
      log.info("  python scripts/generate_shards.py")
      log.info("  python scripts/pretrain_titan_v3.py --config configs/titan_1b.yaml")

  if __name__ == "__main__":
      main()
  