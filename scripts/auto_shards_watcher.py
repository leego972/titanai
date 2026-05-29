import time, os, subprocess

LOG         = "/workspace/titanai/data/corpus_master.log"
SCRIPT_POL  = "/workspace/titanai/scripts/load_corpus_world_politics.py"
SCRIPT_APP  = "/workspace/titanai/scripts/load_corpus_applied_sciences.py"
SCRIPT_GEN  = "/workspace/titanai/scripts/generate_shards.py"
DONE_POL    = "/workspace/titanai/data/.world_politics_loaded"
DONE_APP    = "/workspace/titanai/data/.applied_sciences_loaded"
DONE_SHARDS = "/workspace/titanai/data/.shards_generated"
WATCH_LOG   = "/workspace/titanai/data/shards_watcher.log"
POL_LOG     = "/workspace/titanai/data/corpus_world_politics.log"
APP_LOG     = "/workspace/titanai/data/corpus_applied_sciences.log"
GEN_LOG     = "/workspace/titanai/data/generate_shards.log"
PYTHON      = "/opt/conda/bin/python3"


def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = "[{}] {}".format(ts, msg)
    print(line, flush=True)
    with open(WATCH_LOG, "a") as f:
        f.write(line + "\n")


def loaders_finished():
    if not os.path.exists(LOG):
        return False
    with open(LOG) as f:
        return "ALL LOADERS COMPLETE" in f.read()


def run_script(script, logfile, done_flag, label):
    if os.path.exists(done_flag):
        log("{} already done — skipping".format(label))
        return True
    log("Starting {} ...".format(label))
    with open(logfile, "a") as lf:
        proc = subprocess.run([PYTHON, script], stdout=lf, stderr=lf)
    if proc.returncode == 0:
        open(done_flag, "w").close()
        log("{} completed OK (rc=0)".format(label))
        return True
    else:
        log("ERROR: {} exited rc={}".format(label, proc.returncode))
        return False


log("Watcher v3 started — chain: world_politics -> applied_sciences -> generate_shards")
log("Polling every 3 min for ALL LOADERS COMPLETE")

while True:
    if loaders_finished():
        log("ALL LOADERS COMPLETE detected — starting chain")
        ok = run_script(SCRIPT_POL, POL_LOG, DONE_POL, "load_corpus_world_politics")
        if ok:
            ok2 = run_script(SCRIPT_APP, APP_LOG, DONE_APP, "load_corpus_applied_sciences")
            if ok2:
                run_script(SCRIPT_GEN, GEN_LOG, DONE_SHARDS, "generate_shards")
                log("Full chain complete. Watcher exiting.")
                break
            else:
                log("applied_sciences failed — still running generate_shards")
                run_script(SCRIPT_GEN, GEN_LOG, DONE_SHARDS, "generate_shards")
                break
        else:
            log("world_politics failed — aborting chain")
            break
    time.sleep(180)
