#!/usr/bin/env python3
"""
sync_dropbox_to_github.py — runs every 30 min via background loop
1. Dropbox /workspace/titanai/scripts  -> GitHub leego972/titanai/scripts   (new/changed, post-cutoff only)
2. Dropbox /workspace/titanai/configs  -> GitHub leego972/titanai/configs   (new/changed, post-cutoff only)
3. Dropbox /workspace/titanai/data     -> GitHub data/dropbox_sync/         (small text files only)
4. Instance live status + logs         -> Dropbox /workspace/titanai_status/ (backup)

Safety: files already on GitHub are only overwritten if Dropbox server_modified >= SYNC_CUTOFF
        so we never clobber recent GitHub edits with old Dropbox copies.
"""
import os, sys, json, base64, glob, hashlib, urllib.request, urllib.parse, urllib.error, time
from datetime import datetime, timezone

SYNC_CUTOFF = datetime(2026, 5, 1, tzinfo=timezone.utc)

ENV = '/workspace/titanai/.dropbox_sync_env'
cfg = {}
if os.path.exists(ENV):
    for line in open(ENV):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip()

DBX_TOKEN   = cfg.get('DBX_ACCESS_TOKEN', '')
DBX_REFRESH = cfg.get('DBX_REFRESH_TOKEN', '')
APP_KEY     = cfg.get('DBX_APP_KEY', 'pqbzwk2xdjlp3wd')
APP_SECRET  = cfg.get('DBX_APP_SECRET', 'vjsf0a5wiwlk39d')
GH_TOKEN    = cfg.get('GH_TOKEN', '')
REPO        = 'leego972/titanai'


def log(msg):
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    print(f'[{ts}] {msg}', flush=True)


def dbx_request(endpoint, body):
    global DBX_TOKEN
    req = urllib.request.Request(
        f'https://api.dropboxapi.com/2/{endpoint}',
        data=json.dumps(body).encode()
    )
    req.add_header('Authorization', f'Bearer {DBX_TOKEN}')
    req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        if 'expired_access_token' in err and DBX_REFRESH:
            log('Token expired — refreshing...')
            if refresh_token():
                return dbx_request(endpoint, body)
        log(f'Dropbox error {e.code}: {err[:200]}')
        return None
    except Exception as e:
        log(f'Dropbox request error: {e}')
        return None


def dbx_download(path):
    global DBX_TOKEN
    req = urllib.request.Request('https://content.dropboxapi.com/2/files/download')
    req.add_header('Authorization', f'Bearer {DBX_TOKEN}')
    req.add_header('Dropbox-API-Arg', json.dumps({'path': path}))
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except Exception as e:
        log(f'Download error {path}: {e}')
        return None


def dbx_upload(path, content):
    global DBX_TOKEN
    req = urllib.request.Request(
        'https://content.dropboxapi.com/2/files/upload',
        data=content
    )
    req.add_header('Authorization', f'Bearer {DBX_TOKEN}')
    req.add_header('Content-Type', 'application/octet-stream')
    req.add_header('Dropbox-API-Arg', json.dumps({'path': path, 'mode': 'overwrite', 'autorename': False}))
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except Exception as e:
        log(f'Upload error {path}: {e}')
        return None


def refresh_token():
    global DBX_TOKEN
    if not DBX_REFRESH:
        return False
    creds = base64.b64encode(f'{APP_KEY}:{APP_SECRET}'.encode()).decode()
    data = urllib.parse.urlencode({'grant_type': 'refresh_token', 'refresh_token': DBX_REFRESH}).encode()
    req = urllib.request.Request('https://api.dropbox.com/oauth2/token', data=data)
    req.add_header('Authorization', f'Basic {creds}')
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            d = json.loads(resp.read())
            DBX_TOKEN = d['access_token']
            lines = open(ENV).readlines()
            with open(ENV, 'w') as f:
                for ln in lines:
                    f.write(f'DBX_ACCESS_TOKEN={DBX_TOKEN}\n' if ln.startswith('DBX_ACCESS_TOKEN=') else ln)
            log('Token refreshed OK')
            return True
    except Exception as e:
        log(f'Refresh failed: {e}')
        return False


def gh_get(path):
    req = urllib.request.Request(f'https://api.github.com/repos/{REPO}/contents/{path}')
    req.add_header('Authorization', f'token {GH_TOKEN}')
    req.add_header('Accept', 'application/vnd.github.v3+json')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return None
    except Exception:
        return None


def gh_put(path, content_bytes, message, sha=None):
    body = {'message': message, 'content': base64.b64encode(content_bytes).decode()}
    if sha:
        body['sha'] = sha
    req = urllib.request.Request(
        f'https://api.github.com/repos/{REPO}/contents/{path}',
        data=json.dumps(body).encode(), method='PUT'
    )
    req.add_header('Authorization', f'token {GH_TOKEN}')
    req.add_header('Content-Type', 'application/json')
    req.add_header('Accept', 'application/vnd.github.v3+json')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        log(f'GitHub put error {path}: {e}')
        return None


def sync_dropbox_folder_to_github(dbx_folder, gh_prefix, max_size=500*1024, exts=None):
    result = dbx_request('files/list_folder', {'path': dbx_folder, 'recursive': False})
    if not result:
        log(f'  Could not list {dbx_folder}')
        return 0
    pushed = skipped_old = 0
    for entry in result.get('entries', []):
        if entry['.tag'] != 'file':
            continue
        name = entry['name']
        size = entry.get('size', 0)
        dbx_modified = datetime.strptime(
            entry['server_modified'], '%Y-%m-%dT%H:%M:%SZ'
        ).replace(tzinfo=timezone.utc)
        if size > max_size:
            continue
        if exts and not any(name.endswith(e) for e in exts):
            continue
        content = dbx_download(entry['path_display'])
        if content is None:
            continue
        gh_path = f'{gh_prefix}/{name}'
        existing = gh_get(gh_path)
        sha = existing.get('sha') if existing else None
        if existing:
            try:
                remote_content = base64.b64decode(
                    existing.get('content', '').replace('\n', '')
                )
            except Exception:
                remote_content = b''
            if remote_content == content:
                continue
            if dbx_modified < SYNC_CUTOFF:
                skipped_old += 1
                continue
        r = gh_put(gh_path, content, f'sync(dropbox): {name}', sha)
        if r and r.get('commit'):
            log(f'  Pushed {gh_path} ({size}B)')
            pushed += 1
        time.sleep(0.5)
    if skipped_old:
        log(f'  Skipped {skipped_old} old Dropbox files (GitHub version kept)')
    return pushed


def push_status_to_dropbox():
    files = [
        ('/workspace/titanai/data/live_status.json',
         '/workspace/titanai_status/live_status.json'),
        ('/workspace/titanai/data/training_log_latest.txt',
         '/workspace/titanai_status/training_log_latest.txt'),
        ('/workspace/titanai/data/corpus_progress_latest.txt',
         '/workspace/titanai_status/corpus_progress_latest.txt'),
    ]
    for pat in glob.glob('/workspace/titanai/logs/titan_1b/upgrade_*/training.log'):
        stage = os.path.basename(os.path.dirname(pat))
        files.append((pat, f'/workspace/titanai_status/{stage}_training.log'))
    uploaded = 0
    for local, remote in files:
        if not os.path.exists(local):
            continue
        content = open(local, 'rb').read()
        if len(content) > 2 * 1024 * 1024:
            content = content[-500*1024:]
        if dbx_upload(remote, content):
            uploaded += 1
    if uploaded:
        log(f'Pushed {uploaded} status files to Dropbox')


if __name__ == '__main__':
    if not DBX_TOKEN:
        log('ERROR: No DBX_ACCESS_TOKEN — aborting')
        sys.exit(1)
    if not GH_TOKEN:
        log('ERROR: No GH_TOKEN — aborting')
        sys.exit(1)

    log('=== Dropbox -> GitHub sync starting ===')

    n = sync_dropbox_folder_to_github(
        '/workspace/titanai/scripts', 'scripts',
        exts=['.py', '.sh', '.yaml', '.yml', '.json', '.md', '.txt']
    )
    log(f'Scripts: {n} synced to GitHub')

    n = sync_dropbox_folder_to_github(
        '/workspace/titanai/configs', 'configs',
        exts=['.yaml', '.yml', '.json', '.toml', '.cfg', '.md']
    )
    log(f'Configs: {n} synced to GitHub')

    n = sync_dropbox_folder_to_github(
        '/workspace/titanai/data', 'data/dropbox_sync',
        max_size=200*1024,
        exts=['.json', '.md', '.txt', '.log', '.yaml']
    )
    log(f'Data: {n} synced to GitHub')

    push_status_to_dropbox()

    log('=== Sync complete ===')
