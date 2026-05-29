import time, os, subprocess

SCRIPT_NEW   = "/workspace/titanai/scripts/load_corpus_cyber_web_psych_motors.py"
SCRIPT_GEN   = "/workspace/titanai/scripts/generate_shards.py"
DONE_PHASE1  = "/workspace/titanai/data/.shards_generated"
DONE_NEW     = "/workspace/titanai/data/.cyber_web_psych_motors_loaded"
DONE_PHASE2  = "/workspace/titanai/data/.shards_phase2_generated"
LOG_NEW      = "/workspace/titanai/data/corpus_cyber_web_psych_motors.log"
LOG_GEN2     = "/workspace/titanai/data/generate_shards_phase2.log"
WATCH_LOG    = "/workspace/titanai/data/phase2_watcher.log"
PYTHON       = "/opt/conda/bin/python3"


def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = "[{}] {}".format(ts, msg)
    print(line, flush=True)
    with open(WATCH_LOG, "a") as f:
        f.write(line + "\n")


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


log("Phase-2 Watcher started — waiting for Phase 1 (.shards_generated)")
log("Chain: load_corpus_cyber_web_psych_motors -> generate_shards (phase 2)")
log("Polling every 3 min for .shards_generated flag ...")

while True:
    if os.path.exists(DONE_PHASE1):
        log(".shards_generated detected — starting Phase 2 chain")
        ok = run_script(SCRIPT_NEW, LOG_NEW, DONE_NEW,
                        "load_corpus_cyber_web_psych_motors")
        if ok:
            run_script(SCRIPT_GEN, LOG_GEN2, DONE_PHASE2,
                       "generate_shards (phase 2)")
        else:
            log("New loader failed — running generate_shards anyway")
            run_script(SCRIPT_GEN, LOG_GEN2, DONE_PHASE2,
                       "generate_shards (phase 2)")
        log("Phase 2 complete. Watcher exiting.")
        break
    time.sleep(180)
