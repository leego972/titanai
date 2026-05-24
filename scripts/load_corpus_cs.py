"""
TitanAI Computer Science Corpus Loader — PhD Depth
====================================================
Adds comprehensive computer science training data to corpus_C_technical.
Target: 80,000+ documents across all CS subfields.

Subfields covered:
  1.  Algorithms & Data Structures   — sorting, graphs, DP, NP-completeness,
                                        advanced trees, amortized analysis
  2.  Programming Language Theory    — type theory, lambda calculus, semantics,
                                        functional/logic programming, type inference
  3.  Compilers & Program Analysis   — parsing, SSA, dataflow, JIT, LLVM,
                                        abstract interpretation, verification
  4.  Operating Systems              — scheduling, virtual memory, file systems,
                                        concurrency, kernel design, containers
  5.  Computer Architecture          — microarchitecture, pipeline, out-of-order,
                                        cache hierarchy, NUMA, GPU architecture
  6.  Computer Networks              — TCP/IP, BGP, congestion control, SDN,
                                        P2P, wireless, network coding
  7.  Database Systems               — relational algebra, query optimization,
                                        transactions, distributed DB, NewSQL, NoSQL
  8.  Artificial Intelligence        — search, planning, knowledge representation,
                                        expert systems, logic, reasoning
  9.  Machine Learning               — supervised, unsupervised, deep learning,
                                        optimization, generalization, theory
 10.  Computer Vision & Graphics     — feature detection, CNNs, 3D vision, ray
                                        tracing, GPU shaders, geometry processing
 11.  Distributed Systems & Cloud    — consensus, fault tolerance, replication,
                                        cloud-native, serverless, edge computing
 12.  Cybersecurity & Cryptography   — crypto primitives, PKI, network security,
                                        malware, reverse engineering, formal security
 13.  Software Engineering           — design patterns, SOLID, testing, refactoring,
                                        agile, formal methods, program verification
 14.  Theory of Computation          — automata, Turing machines, complexity classes,
                                        P vs NP, circuit complexity, logic
 15.  Quantum Computing              — quantum gates, Shor/Grover algorithms,
                                        error correction, quantum complexity
 16.  Human-Computer Interaction     — usability, accessibility, cognitive models,
                                        UI/UX, information visualization
 17.  Parallel & High-Performance    — SIMD, CUDA, MPI, cache-oblivious algorithms,
                                        performance analysis, auto-parallelization
 18.  Bioinformatics & Comp Biology  — sequence alignment, phylogenetics,
                                        protein structure prediction, genomics tools

Sources per subfield:
  • Wikipedia REST API   — targeted queries per subfield
  • wikimedia/wikipedia  — streaming keyword filter
  • StackExchange Q&A    — cs.se, stackoverflow, softwareengineering.se, security.se
  • arXiv preprints      — all cs.* categories + stat.ML
  • Project Gutenberg    — classic computing texts (Turing, von Neumann, etc.)
  • OpenAlex API         — open-access CS research papers

Run:
  pip install datasets requests tqdm
  python scripts/load_corpus_cs.py
"""

import os, sys, json, time, logging, requests
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent
RAW  = BASE / "data" / "raw"
LOG  = BASE / "data" / "corpus_cs.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(str(LOG))]
)
log = logging.getLogger("cs_loader")

INVENTORY     = BASE / "data" / "source_inventory.json"
EXCLUSION_LOG = BASE / "data" / "exclusions_cs.jsonl"
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
            (bucket_dir / f"{tag}_{start+i:06d}.txt").write_text(text.strip(), encoding="utf-8")
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
            text = text[text.index("\n")+1:]
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

def wiki_stream(keywords, bucket_dir, tag, max_docs=5000):
    try:
        from datasets import load_dataset
        wiki = load_dataset("wikimedia/wikipedia","20231101.en",
                            split="train", streaming=True)
        kw = {k.lower() for k in keywords}
        docs, scanned = [], 0
        for item in wiki:
            scanned += 1
            if scanned > 900_000: break
            title = item.get("title","").lower()
            text  = item.get("text","").strip()
            if len(text) < 300: continue
            if any(k in title or k in text[:600].lower() for k in kw):
                docs.append(f"# {item['title']}\n\n{text[:9000]}")
            if len(docs) >= max_docs: break
        return write_docs(bucket_dir, docs, tag)
    except Exception as e:
        record_exclusion(f"wiki_stream:{tag}", str(e)); return 0

def se_qa(domain_kws, bucket_dir, tag, extra_kw=None, max_docs=4000):
    try:
        from datasets import load_dataset
        se = load_dataset("HuggingFaceH4/stack-exchange-preferences",
                          split="train", streaming=True)
        docs = []
        for i, item in enumerate(se):
            if i > 4_000_000: break
            dom = safe_get(item, "domain")
            if not any(k in dom.lower() for k in domain_kws): continue
            q = safe_get(item, "question")
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

def arxiv_cats(cats, bucket_dir, tag, extra_kw=None, max_docs=5000):
    try:
        from datasets import load_dataset
        arxiv = load_dataset("Cornell-University/arxiv",
                             split="train", streaming=True)
        cat_set = set(cats)
        docs = []
        for i, item in enumerate(arxiv):
            if i > 5_000_000: break
            c = set((item.get("categories","") or "").split())
            if not c.intersection(cat_set): continue
            title = (item.get("title","") or "").replace("\n"," ").strip()
            abst  = (item.get("abstract","") or "").replace("\n"," ").strip()
            if len(abst) < 80: continue
            if extra_kw:
                combo = (title+" "+abst).lower()
                if not any(k in combo for k in extra_kw): continue
            docs.append(f"Title: {title}\nCategories: {' '.join(sorted(c))}\n\n"
                        f"Abstract:\n{abst}")
            if len(docs) >= max_docs: break
        return write_docs(bucket_dir, docs, tag)
    except Exception as e:
        record_exclusion(f"arxiv:{tag}", str(e)); return 0

def gutenberg(ids, bucket_dir, tag):
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

def openalex(queries, bucket_dir, tag, label="CS Research"):
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
CS_ALGO_KW = {
    "algorithm","data structure","sorting algorithm","binary search","quicksort",
    "mergesort","heapsort","hash table","hash map","binary tree","red-black tree",
    "avl tree","b-tree","trie","suffix array","graph algorithm","breadth-first search",
    "depth-first search","dijkstra","bellman-ford","floyd-warshall","minimum spanning tree",
    "prim's algorithm","kruskal","dynamic programming","memoization","greedy algorithm",
    "divide and conquer","amortized analysis","big-o notation","time complexity",
    "space complexity","np-completeness","np-hard","polynomial reduction","sat problem",
    "approximation algorithm","randomized algorithm","streaming algorithm",
    "cache-oblivious algorithm","segment tree","fenwick tree","union-find",
    "network flow","max-flow min-cut","bipartite matching","string matching",
    "kmp algorithm","suffix tree","computational geometry","convex hull",
}

CS_PL_KW = {
    "programming language","type theory","type system","type inference",
    "lambda calculus","functional programming","haskell","ocaml","lisp",
    "logic programming","prolog","dependent types","polymorphism","generics",
    "object-oriented programming","inheritance","encapsulation","abstraction",
    "garbage collection","memory management","reference counting",
    "operational semantics","denotational semantics","axiomatic semantics",
    "hoare logic","formal verification","program correctness","coq","lean theorem",
    "rust ownership","linear types","session types","effect system",
    "monad","functor","applicative","category theory programming",
    "continuation passing style","abstract interpretation","static analysis",
}

CS_SYS_KW = {
    "operating system","process scheduling","virtual memory","page fault",
    "memory management","file system","ext4","btrfs","journaling",
    "concurrency","deadlock","mutex","semaphore","monitor","lock-free",
    "computer architecture","instruction set","risc","cisc","arm","x86",
    "pipeline hazard","branch prediction","out-of-order execution",
    "cache memory","cache coherence","mesi protocol","tlb",
    "gpu architecture","cuda","opencl","simd","vector instruction",
    "numa architecture","memory bandwidth","memory latency",
    "hypervisor","virtualization","container","docker","kubernetes",
    "linux kernel","system call","interrupt handler","device driver",
    "tcp/ip","congestion control","routing protocol","sdn","nfv",
    "database","query optimization","b-tree index","transaction","acid",
    "distributed database","two-phase commit","mvcc","nosql","newSQL",
}

CS_AI_KW = {
    "artificial intelligence","machine learning","deep learning","neural network",
    "convolutional neural network","recurrent neural network","transformer",
    "attention mechanism","bert","gpt","large language model","fine-tuning",
    "reinforcement learning","q-learning","policy gradient","actor-critic",
    "supervised learning","unsupervised learning","semi-supervised","self-supervised",
    "gradient descent","backpropagation","vanishing gradient","batch normalization",
    "dropout","regularization","overfitting","generalization","bias-variance",
    "support vector machine","random forest","gradient boosting","xgboost",
    "computer vision","object detection","image segmentation","yolo",
    "generative adversarial network","variational autoencoder","diffusion model",
    "natural language processing","word embedding","word2vec","bert embedding",
    "knowledge graph","expert system","bayesian network","probabilistic graphical",
    "planning search","a* algorithm","monte carlo tree search","alpha-beta pruning",
    "federated learning","transfer learning","meta-learning","few-shot learning",
    "explainable ai","adversarial example","robustness","fairness bias",
}

CS_SEC_KW = {
    "cryptography","symmetric encryption","aes","des","stream cipher",
    "public key cryptography","rsa","elliptic curve","diffie-hellman",
    "digital signature","hash function","sha","md5","collision resistance",
    "zero knowledge proof","homomorphic encryption","secure multiparty",
    "network security","tls","ssl","certificate","pki","firewall",
    "intrusion detection","ids","ips","honeypot","threat modeling",
    "malware","virus","worm","ransomware","rootkit","trojan",
    "vulnerability","exploit","buffer overflow","sql injection","xss",
    "penetration testing","ctf","reverse engineering","binary exploitation",
    "formal security proof","provable security","cryptanalysis",
    "quantum cryptography","post-quantum","lattice cryptography",
    "blockchain","smart contract","ethereum","consensus mechanism",
    "side channel attack","timing attack","cache attack","spectre meltdown",
}

CS_THEORY_KW = {
    "automata theory","finite automaton","regular language","context-free grammar",
    "pushdown automaton","turing machine","decidability","halting problem",
    "complexity theory","p versus np","pspace","exptime","polynomial hierarchy",
    "circuit complexity","boolean circuit","nc complexity","randomized complexity",
    "interactive proof system","ip equals pspace","pcsp theorem",
    "kolmogorov complexity","algorithmic information theory",
    "communication complexity","information theory","shannon entropy",
    "coding theory","error correcting code","reed-solomon","ldpc","turbo code",
    "quantum complexity","bqp","quantum error correction","stabilizer code",
    "descriptive complexity","logic and computation","first-order logic",
    "modal logic","temporal logic","model checking","ltl","ctl",
}
CS_ALL_KW = CS_ALGO_KW | CS_PL_KW | CS_SYS_KW | CS_AI_KW | CS_SEC_KW | CS_THEORY_KW

# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_algorithms_data_structures():
    log.info("=" * 65)
    log.info("CS-1: Algorithms & Data Structures")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "sorting algorithms quicksort mergesort complexity",
        "graph algorithms shortest path BFS DFS",
        "dynamic programming optimal substructure memoization",
        "NP-completeness polynomial reduction Cook theorem",
        "approximation algorithms TSP vertex cover",
        "randomized algorithms Las Vegas Monte Carlo",
        "amortized analysis splay tree potential method",
        "balanced binary search trees AVL red-black",
        "hash tables open addressing chaining performance",
        "heap priority queue Fibonacci heap",
        "string algorithms suffix array KMP Rabin-Karp",
        "computational geometry convex hull Voronoi",
        "network flow max-flow Ford-Fulkerson matching",
        "cache-oblivious algorithms memory hierarchy",
        "streaming algorithms sketching approximate counting",
        "external memory algorithms B-tree",
        "parallel algorithms PRAM model",
        "data structure lower bounds cell-probe model",
        "online algorithms competitive analysis",
        "tree decomposition treewidth parameterized",
    ]
    n = wiki_api(queries, bucket, "cs_algo_wiki_api"); total += n
    record_source("corpus_C_technical","Wikipedia Algorithms API","https://en.wikipedia.org",
                  approx_mb([""]*n),"Algorithms: sorting, graphs, DP, NP, approximation",n)
    log.info(f"  [CS-ALGO-1] Wiki API: {n}")

    n = wiki_stream(CS_ALGO_KW, bucket, "cs_algo_wiki_stream", max_docs=3000); total += n
    record_source("corpus_C_technical","Wikipedia Algorithms Stream","https://en.wikipedia.org",
                  approx_mb([""]*n),"Algorithm keyword stream",n)
    log.info(f"  [CS-ALGO-2] Wiki stream: {n}")

    n = se_qa(["cs","algorithms","math"], bucket, "cs_algo_se",
              extra_kw={"algorithm","complexity","data structure","sort","graph","dp","tree","hash"},
              max_docs=4000); total += n
    record_source("corpus_C_technical","Algorithms StackExchange Q&A","https://cs.stackexchange.com",
                  approx_mb([""]*n),"Expert Q&A on algorithm design and analysis",n)
    log.info(f"  [CS-ALGO-3] SE Q&A: {n}")

    n = arxiv_cats(["cs.DS","cs.CG","cs.DM"], bucket, "cs_algo_arxiv", max_docs=3000); total += n
    record_source("corpus_C_technical","arXiv Algorithms Papers","https://arxiv.org",
                  approx_mb([""]*n),"Data structures, computational geometry, combinatorics",n)
    log.info(f"  [CS-ALGO-4] arXiv: {n}")

    n = openalex([
        "randomized algorithm approximation guarantee",
        "dynamic graph algorithm online update",
        "succinct data structure compressed representation",
        "sublinear algorithm property testing",
        "parameterized algorithm fixed parameter tractable",
    ], bucket, "cs_algo_openalex", "Algorithms Research"); total += n
    log.info(f"  [CS-ALGO-5] OpenAlex: {n}")

    log.info(f"  >>> Algorithms & DS total: {total:,}")
    return total


def load_programming_languages_compilers():
    log.info("=" * 65)
    log.info("CS-2: Programming Languages & Compilers")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "type theory dependent types polymorphism",
        "lambda calculus Church encoding beta reduction",
        "operational semantics small-step big-step",
        "Hoare logic program verification correctness",
        "garbage collection generational mark-sweep",
        "compiler design lexing parsing AST",
        "SSA static single assignment form",
        "dataflow analysis liveness reaching definitions",
        "JIT compilation just-in-time optimization",
        "LLVM intermediate representation passes",
        "Rust ownership borrow checker memory safety",
        "functional programming monads Haskell",
        "logic programming Prolog unification",
        "abstract interpretation lattice fixpoint",
        "program synthesis specification",
        "session types linear types concurrency",
        "effect systems algebraic effects handlers",
        "formal semantics denotational domain theory",
        "register allocation graph coloring",
        "loop optimizations vectorization inlining",
    ]
    n = wiki_api(queries, bucket, "cs_pl_wiki_api"); total += n
    record_source("corpus_C_technical","Wikipedia PL/Compilers API","https://en.wikipedia.org",
                  approx_mb([""]*n),"PL theory, type systems, compilers, program analysis",n)
    log.info(f"  [CS-PL-1] Wiki API: {n}")

    n = wiki_stream(CS_PL_KW, bucket, "cs_pl_wiki_stream", max_docs=3000); total += n
    n2 = se_qa(["cs","softwareengineering","programmers"], bucket, "cs_pl_se",
               extra_kw={"type","compiler","language","lambda","monad","garbage","llvm",
                         "haskell","rust","prolog","semantics","parser","grammar"},
               max_docs=3000); total += n2
    n3 = arxiv_cats(["cs.PL","cs.LO","cs.FL"], bucket, "cs_pl_arxiv", max_docs=3000); total += n3
    log.info(f"  [CS-PL] Wiki:{n} SE:{n2} arXiv:{n3}")
    return total


def load_operating_systems_architecture():
    log.info("=" * 65)
    log.info("CS-3: Operating Systems & Architecture")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "operating system process scheduling algorithms",
        "virtual memory paging segmentation TLB",
        "file system design journaling copy-on-write",
        "concurrency synchronization lock-free algorithms",
        "deadlock detection prevention avoidance",
        "CPU microarchitecture pipeline superscalar",
        "branch prediction dynamic static speculative",
        "cache hierarchy cache coherence MESI protocol",
        "GPU architecture warp CUDA programming model",
        "NUMA non-uniform memory access optimization",
        "hypervisor type 1 type 2 virtualization",
        "container technology namespaces cgroups",
        "Linux kernel internals system calls",
        "memory allocator malloc slab allocator",
        "interrupt handling device drivers kernel modules",
        "RISC-V instruction set architecture design",
        "out-of-order execution Tomasulo algorithm",
        "memory consistency models sequential TSO",
        "storage systems NVMe SSD flash translation",
        "real-time operating systems scheduling",
    ]
    n = wiki_api(queries, bucket, "cs_os_wiki_api"); total += n
    n2 = wiki_stream(CS_SYS_KW, bucket, "cs_sys_wiki_stream", max_docs=4000); total += n2
    n3 = se_qa(["unix","linux","superuser","cs"], bucket, "cs_os_se",
               extra_kw={"kernel","scheduler","memory","cache","process","thread",
                         "filesystem","interrupt","driver","virtualization","cpu"},
               max_docs=3000); total += n3
    n4 = arxiv_cats(["cs.OS","cs.AR","cs.PF","cs.DC"], bucket, "cs_os_arxiv", max_docs=3000); total += n4
    record_source("corpus_C_technical","OS & Architecture Multi-source","",
                  approx_mb([""]*total),"OS internals, CPU microarchitecture, virtualization",total)
    log.info(f"  [CS-OS] Wiki:{n}+{n2} SE:{n3} arXiv:{n4}")
    return total


def load_ai_machine_learning():
    log.info("=" * 65)
    log.info("CS-4: Artificial Intelligence & Machine Learning")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "deep learning neural network backpropagation",
        "transformer architecture self-attention mechanism",
        "large language model pretraining fine-tuning",
        "convolutional neural network image recognition",
        "recurrent neural network LSTM sequence modeling",
        "generative adversarial network image synthesis",
        "diffusion model score matching denoising",
        "reinforcement learning policy gradient actor-critic",
        "deep Q-network Atari games",
        "variational autoencoder latent space",
        "graph neural network message passing",
        "attention mechanism neural machine translation",
        "BERT GPT language model pretraining",
        "object detection YOLO feature pyramid",
        "semantic segmentation fully convolutional",
        "knowledge graph embedding representation",
        "Bayesian deep learning uncertainty estimation",
        "federated learning privacy preserving",
        "neural architecture search AutoML",
        "adversarial robustness certified defenses",
        "explainable AI interpretability SHAP LIME",
        "meta-learning few-shot learning MAML",
        "contrastive learning self-supervised representation",
        "causal inference do-calculus structural equation",
        "model compression pruning quantization distillation",
    ]
    n = wiki_api(queries, bucket, "cs_ai_wiki_api"); total += n
    record_source("corpus_C_technical","Wikipedia AI/ML API","https://en.wikipedia.org",
                  approx_mb([""]*n),"Deep learning, transformers, RL, generative models",n)
    log.info(f"  [CS-AI-1] Wiki API: {n}")

    n = wiki_stream(CS_AI_KW, bucket, "cs_ai_wiki_stream", max_docs=4000); total += n
    log.info(f"  [CS-AI-2] Wiki stream: {n}")

    n = se_qa(["datascience","ai","stats","math"], bucket, "cs_ai_se",
              extra_kw={"neural","learning","model","training","gradient","loss",
                        "overfitting","attention","transformer","cnn","rnn","llm"},
              max_docs=4000); total += n
    log.info(f"  [CS-AI-3] SE Q&A: {n}")

    n = arxiv_cats(["cs.LG","cs.AI","cs.CV","cs.CL","cs.NE","stat.ML"],
                   bucket, "cs_ai_arxiv", max_docs=6000); total += n
    record_source("corpus_C_technical","arXiv AI/ML Papers","https://arxiv.org",
                  approx_mb([""]*n),"ML research: LG, AI, CV, CL, NE, stat.ML",n)
    log.info(f"  [CS-AI-4] arXiv: {n}")

    n = openalex([
        "transformer language model scaling laws",
        "reinforcement learning from human feedback RLHF",
        "graph neural network knowledge graph",
        "federated learning differential privacy",
        "neural network pruning quantization efficiency",
        "causal machine learning fairness",
        "diffusion probabilistic model generation",
        "contrastive self-supervised visual representation",
    ], bucket, "cs_ai_openalex", "AI/ML Research"); total += n
    log.info(f"  [CS-AI-5] OpenAlex: {n}")

    log.info(f"  >>> AI & ML total: {total:,}")
    return total


def load_security_cryptography():
    log.info("=" * 65)
    log.info("CS-5: Cybersecurity & Cryptography")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "symmetric cryptography AES block cipher modes",
        "RSA public key cryptography number theory",
        "elliptic curve cryptography ECDSA",
        "zero knowledge proof zk-SNARK",
        "hash function SHA-3 collision resistance",
        "TLS protocol handshake certificate",
        "network intrusion detection anomaly",
        "malware analysis reverse engineering",
        "buffer overflow stack smashing exploit",
        "SQL injection XSS CSRF web security",
        "format string vulnerability heap spray",
        "side channel attack timing cache spectre",
        "formal security proof reduction game-based",
        "post-quantum lattice cryptography NTRU Kyber",
        "blockchain consensus proof of work stake",
        "smart contract vulnerability reentrancy",
        "binary exploitation ROP chain ASLR bypass",
        "penetration testing methodology OWASP",
        "homomorphic encryption fully partially",
        "secure multi-party computation garbled circuit",
    ]
    n = wiki_api(queries, bucket, "cs_sec_wiki_api"); total += n
    n2 = wiki_stream(CS_SEC_KW, bucket, "cs_sec_wiki_stream", max_docs=3000); total += n2
    n3 = se_qa(["security","crypto","reverseengineering"], bucket, "cs_sec_se",
               max_docs=4000); total += n3
    n4 = arxiv_cats(["cs.CR","cs.GT"], bucket, "cs_sec_arxiv", max_docs=3000); total += n4
    record_source("corpus_C_technical","Security & Crypto Multi-source","",
                  approx_mb([""]*total),"Cryptography, network security, malware, formal proofs",total)
    log.info(f"  [CS-SEC] Wiki:{n}+{n2} SE:{n3} arXiv:{n4}")
    return total


def load_theory_computation():
    log.info("=" * 65)
    log.info("CS-6: Theory of Computation & Complexity")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "finite automata regular languages Myhill-Nerode",
        "context-free grammars CYK parsing",
        "Turing machine decidability halting problem",
        "time complexity polynomial hierarchy",
        "P vs NP Millennium Prize problem",
        "NP-complete reduction Cook-Levin theorem",
        "space complexity PSPACE LOGSPACE",
        "circuit complexity Boolean NC problems",
        "interactive proofs IP PSPACE Arthur-Merlin",
        "probabilistically checkable proofs PCP theorem",
        "communication complexity protocol",
        "randomized complexity BPP RP",
        "quantum complexity BQP relationship P",
        "Kolmogorov complexity algorithmic information",
        "descriptive complexity Fagin theorem",
        "algebraic complexity arithmetic circuits",
        "proof complexity resolution SAT",
        "parametrized complexity W-hierarchy",
        "fine-grained complexity SETH strong ETH",
        "coding theory Shannon capacity list decoding",
    ]
    n = wiki_api(queries, bucket, "cs_theory_wiki_api"); total += n
    n2 = wiki_stream(CS_THEORY_KW, bucket, "cs_theory_wiki_stream", max_docs=3000); total += n2
    n3 = se_qa(["cs","math"], bucket, "cs_theory_se",
               extra_kw={"turing","automata","complexity","np","pspace","decidable",
                         "reduction","circuit","proof","formal","logic","grammar"},
               max_docs=3000); total += n3
    n4 = arxiv_cats(["cs.CC","cs.FL","cs.LO","cs.IT"], bucket, "cs_theory_arxiv", max_docs=3000); total += n4
    log.info(f"  [CS-THEORY] Wiki:{n}+{n2} SE:{n3} arXiv:{n4}")
    record_source("corpus_C_technical","Theory of Computation Multi-source","",
                  approx_mb([""]*total),"Automata, complexity, logic, information theory",total)
    return total


def load_software_engineering_distributed():
    log.info("=" * 65)
    log.info("CS-7: Software Engineering & Distributed Systems")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "software design patterns Gang of Four",
        "SOLID principles object-oriented design",
        "test-driven development refactoring",
        "agile scrum software development lifecycle",
        "distributed systems consensus Raft Paxos",
        "CAP theorem consistency availability partition",
        "eventual consistency CRDTs distributed state",
        "microservices architecture service mesh",
        "Kubernetes container orchestration",
        "event-driven architecture event sourcing CQRS",
        "database transaction isolation MVCC",
        "distributed database sharding replication",
        "formal methods model checking TLA+",
        "program verification Hoare logic Coq",
        "cloud computing serverless FaaS",
        "message queue Kafka RabbitMQ streaming",
        "API design REST GraphQL gRPC",
        "DevOps CI/CD infrastructure as code",
        "observability tracing metrics logging",
        "fault tolerance chaos engineering",
    ]
    n = wiki_api(queries, bucket, "cs_se_wiki_api"); total += n
    n2 = se_qa(["softwareengineering","cs","stackoverflow","programmers"],
               bucket, "cs_se_dist_se",
               extra_kw={"pattern","design","distributed","consensus","microservice",
                         "database","transaction","api","testing","agile","cloud"},
               max_docs=4000); total += n2
    n3 = arxiv_cats(["cs.SE","cs.DC","cs.DB","cs.NI"], bucket, "cs_se_arxiv", max_docs=3000); total += n3
    n4 = gutenberg([
        (829,   "The Art of War — Tzu (systems strategy)"),
        (21352, "Principles of Scientific Management — Taylor"),
        (5001,  "Principia — axiomatic formal systems"),
    ], bucket, "cs_se_gutenberg"); total += n4
    log.info(f"  [CS-SE] Wiki:{n} SE:{n2} arXiv:{n3} Gut:{n4}")
    return total


def load_quantum_computing():
    log.info("=" * 65)
    log.info("CS-8: Quantum Computing")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "quantum computing qubit superposition entanglement",
        "quantum gate circuit model Hadamard CNOT",
        "Shor algorithm integer factorization",
        "Grover search algorithm quadratic speedup",
        "quantum error correction surface code",
        "fault tolerant quantum computation threshold",
        "variational quantum eigensolver VQE",
        "quantum approximate optimization QAOA",
        "quantum supremacy advantage demonstration",
        "quantum complexity BQP QMA classes",
        "topological quantum computing anyons",
        "quantum simulation many-body physics",
        "quantum cryptography QKD BB84",
        "NISQ noisy intermediate-scale quantum",
        "quantum machine learning kernel methods",
    ]
    n = wiki_api(queries, bucket, "cs_quantum_wiki_api"); total += n
    n2 = se_qa(["quantumcomputing","physics"], bucket, "cs_quantum_se", max_docs=2000); total += n2
    n3 = arxiv_cats(["quant-ph","cs.ET"], bucket, "cs_quantum_arxiv",
                    extra_kw={"quantum","qubit","circuit","algorithm","error","gate"},
                    max_docs=3000); total += n3
    n4 = openalex([
        "quantum error correction fault tolerant threshold",
        "variational quantum algorithm optimization",
        "quantum machine learning kernel advantage",
        "topological qubit Majorana fermion",
    ], bucket, "cs_quantum_openalex", "Quantum Computing Research"); total += n4
    log.info(f"  [CS-QUANTUM] Wiki:{n} SE:{n2} arXiv:{n3} OpenAlex:{n4}")
    return total


def load_computer_vision_graphics():
    log.info("=" * 65)
    log.info("CS-9: Computer Vision & Graphics")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "image recognition convolutional neural networks",
        "object detection YOLO feature pyramid network",
        "semantic instance segmentation panoptic",
        "3D scene reconstruction neural radiance field NeRF",
        "optical flow video understanding tracking",
        "depth estimation stereo vision lidar fusion",
        "image generation diffusion stable diffusion",
        "ray tracing global illumination rendering",
        "physically based rendering BRDF materials",
        "geometry processing mesh simplification",
        "computational photography HDR tone mapping",
        "medical image segmentation analysis",
        "point cloud 3D deep learning PointNet",
        "transformer vision ViT DINO CLIP",
        "pose estimation human body skeleton",
    ]
    n = wiki_api(queries, bucket, "cs_cv_wiki_api"); total += n
    n2 = arxiv_cats(["cs.CV","cs.GR","cs.MM"], bucket, "cs_cv_arxiv", max_docs=4000); total += n2
    n3 = openalex([
        "3D gaussian splatting neural rendering",
        "vision transformer image recognition",
        "diffusion model image synthesis control",
        "medical image deep learning segmentation",
    ], bucket, "cs_cv_openalex", "CV/Graphics Research"); total += n3
    log.info(f"  [CS-CV] Wiki:{n} arXiv:{n2} OpenAlex:{n3}")
    return total


def load_hpc_parallel():
    log.info("=" * 65)
    log.info("CS-10: HPC, Parallel & High-Performance Computing")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "parallel computing SIMD AVX vectorization",
        "CUDA GPU programming warp divergence",
        "MPI message passing distributed HPC",
        "OpenMP parallel threads shared memory",
        "cache-oblivious algorithm optimal cache",
        "roofline model arithmetic intensity bandwidth",
        "performance profiling hotspot optimization",
        "auto-vectorization compiler loop parallelism",
        "deep learning GPU training distributed",
        "high performance linpack TOP500 benchmark",
    ]
    n = wiki_api(queries, bucket, "cs_hpc_wiki_api"); total += n
    n2 = arxiv_cats(["cs.DC","cs.PF","cs.AR"], bucket, "cs_hpc_arxiv",
                    extra_kw={"parallel","gpu","cuda","mpi","hpc","performance","cache"},
                    max_docs=2000); total += n2
    log.info(f"  [CS-HPC] Wiki:{n} arXiv:{n2}")
    return total


def load_bioinformatics():
    log.info("=" * 65)
    log.info("CS-11: Bioinformatics & Computational Biology")
    log.info("=" * 65)
    bucket = RAW / "corpus_C_technical"
    total  = 0

    queries = [
        "sequence alignment Smith-Waterman Needleman-Wunsch",
        "BLAST local alignment sequence search",
        "phylogenetic tree maximum likelihood Bayesian",
        "genome assembly de novo shotgun sequencing",
        "RNA-seq differential expression analysis",
        "protein structure prediction AlphaFold",
        "molecular docking virtual screening",
        "variant calling SNP GWAS association",
        "single cell RNA sequencing clustering",
        "metagenomics 16S microbiome analysis",
        "hidden Markov model protein domain",
        "systems biology network pathway analysis",
        "drug target interaction machine learning",
        "CRISPR guide RNA design off-target",
    ]
    n = wiki_api(queries, bucket, "cs_bio_wiki_api"); total += n
    n2 = se_qa(["bioinformatics","biology"], bucket, "cs_bio_se", max_docs=2000); total += n2
    n3 = arxiv_cats(["q-bio.QM","q-bio.GN","cs.LG"], bucket, "cs_bio_arxiv",
                    extra_kw={"sequence","genome","protein","alignment","rna","drug",
                              "phylogenetic","mutation","expression","cell"},
                    max_docs=2000); total += n3
    log.info(f"  [CS-BIO] Wiki:{n} SE:{n2} arXiv:{n3}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("TitanAI Computer Science Corpus Loader — Starting")
    log.info("11 Subfields: Algorithms | PL/Compilers | OS/Arch | AI/ML | Security")
    log.info("              Theory | SWE/Distributed | Quantum | CV/Graphics | HPC | Bioinformatics")
    log.info(f"Target directory: {RAW}")
    log.info("")

    results = {}
    results["algorithms_ds"]            = load_algorithms_data_structures()
    results["pl_compilers"]             = load_programming_languages_compilers()
    results["os_architecture"]          = load_operating_systems_architecture()
    results["ai_machine_learning"]      = load_ai_machine_learning()
    results["security_cryptography"]    = load_security_cryptography()
    results["theory_computation"]       = load_theory_computation()
    results["software_eng_distributed"] = load_software_engineering_distributed()
    results["quantum_computing"]        = load_quantum_computing()
    results["vision_graphics"]          = load_computer_vision_graphics()
    results["hpc_parallel"]             = load_hpc_parallel()
    results["bioinformatics"]           = load_bioinformatics()

    with open(INVENTORY,"w") as f: json.dump(inventory, f, indent=2)
    with open(EXCLUSION_LOG,"w") as f:
        for e in exclusions: f.write(json.dumps(e)+"\n")

    log.info("")
    log.info("=" * 65)
    log.info("COMPUTER SCIENCE CORPUS LOAD — COMPLETE")
    log.info("=" * 65)
    grand = 0
    for domain, n in results.items():
        log.info(f"  {domain:<35}: {n:>7,} docs")
        grand += n
    log.info(f"  {'GRAND TOTAL':<35}: {grand:>7,} docs")
    log.info("")
    log.info("Next: python scripts/generate_shards.py && python scripts/pretrain_titan_v3.py")

if __name__ == "__main__":
    main()
