#!/usr/bin/env python3
  """
  sync_dropbox_to_github.py
  Runs every 30 min on the Vast.ai instance.
    1. Dropbox /workspace/titanai/scripts  -> GitHub leego972/titanai/scripts
    2. Dropbox /workspace/titanai/configs  -> GitHub leego972/titanai/configs
    3. Dropbox /workspace/titanai/data (logs/json only, <=500KB) -> GitHub data/dropbox_sync/
    4. Instance training logs + status     -> Dropbox /workspace/titanai_status/
  """
  import os, sys, json, base64, hashlib, urllib.request, urllib.parse, urllib.error, time

  # ── credentials (written by install_dropbox_sync.sh) ───────────────────────
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

  def dbx_request(endpoint, body=None, content_type='application/json'):
      global DBX_TOKEN
      url = f'https://api.dropboxapi.com/2/{endpoint}'
      data = json.dumps(body).encode() if body is not None else b''
      req = urllib.request.Request(url, data=data)
      req.add_header('Authorization', f'Bearer {DBX_TOKEN}')
      req.add_header('Content-Type', content_type)
      try:
          with urllib.request.urlopen(req) as resp:
              return json.loads(resp.read())
      except urllib.error.HTTPError as e:
          err = e.read().decode()
          if 'expired_access_token' in err and DBX_REFRESH:
              log('Token expired — refreshing...')
              if refresh_token():
                  return dbx_request(endpoint, body, content_type)
          log(f'Dropbox error {e.code}: {err[:200]}')
          return None

  def dbx_download(path):
      global DBX_TOKEN
      url = 'https://content.dropboxapi.com/2/files/download'
      arg = json.dumps({'path': path})
      req = urllib.request.Request(url)
      req.add_header('Authorization', f'Bearer {DBX_TOKEN}')
      req.add_header('Dropbox-API-Arg', arg)
      try:
          with urllib.request.urlopen(req) as resp:
              return resp.read()
      except Exception as e:
          log(f'Download error {path}: {e}')
          return None

  def dbx_upload(path, content):
      global DBX_TOKEN
      url = 'https://content.dropboxapi.com/2/files/upload'
      arg = json.dumps({'path': path, 'mode': 'overwrite', 'autorename': False})
      req = urllib.request.Request(url, data=content)
      req.add_header('Authorization', f'Bearer {DBX_TOKEN}')
      req.add_header('Content-Type', 'application/octet-stream')
      req.add_header('Dropbox-API-Arg', arg)
      try:
          with urllib.request.urlopen(req) as resp:
              return json.loads(resp.read())
      except Exception as e:
          log(f'Upload error {path}: {e}')
          return None

  def refresh_token():
      global DBX_TOKEN
      if not DBX_REFRESH:
          return False
      creds = base64.b64encode(f'{APP_KEY}:{APP_SECRET}'.encode()).decode()
      url = 'https://api.dropbox.com/oauth2/token'
      data = urllib.parse.urlencode({'grant_type': 'refresh_token', 'refresh_token': DBX_REFRESH}).encode()
      req = urllib.request.Request(url, data=data)
      req.add_header('Authorization', f'Basic {creds}')
      req.add_header('Content-Type', 'application/x-www-form-urlencoded')
      try:
          with urllib.request.urlopen(req) as resp:
              d = json.loads(resp.read())
              DBX_TOKEN = d['access_token']
              # Update .env file
              lines = open(ENV).readlines()
              with open(ENV, 'w') as f:
                  for line in lines:
                      if line.startswith('DBX_ACCESS_TOKEN='):
                          f.write(f'DBX_ACCESS_TOKEN={DBX_TOKEN}\n')
                      else:
                          f.write(line)
              log('Token refreshed OK')
              return True
      except Exception as e:
          log(f'Refresh failed: {e}')
          return False

  def gh_get(path):
      url = f'https://api.github.com/repos/{REPO}/contents/{path}'
      req = urllib.request.Request(url)
      req.add_header('Authorization', f'token {GH_TOKEN}')
      req.add_header('Accept', 'application/vnd.github.v3+json')
      try:
          with urllib.request.urlopen(req) as resp:
              return json.loads(resp.read())
      except urllib.error.HTTPError as e:
          if e.code == 404:
              return None
          raise

  def gh_put(path, content_bytes, message, sha=None):
      url = f'https://api.github.com/repos/{REPO}/contents/{path}'
      body = {'message': message, 'content': base64.b64encode(content_bytes).decode()}
      if sha:
          body['sha'] = sha
      data = json.dumps(body).encode()
      req = urllib.request.Request(url, data=data, method='PUT')
      req.add_header('Authorization', f'token {GH_TOKEN}')
      req.add_header('Content-Type', 'application/json')
      req.add_header('Accept', 'application/vnd.github.v3+json')
      try:
          with urllib.request.urlopen(req) as resp:
              return json.loads(resp.read())
      except Exception as e:
          log(f'GitHub put error {path}: {e}')
          return None

  def sync_dropbox_folder_to_github(dbx_folder, gh_prefix, max_size=500*1024, exts=None):
      """Sync a Dropbox folder into GitHub. Skips files >max_size or wrong extension."""
      result = dbx_request('files/list_folder', {'path': dbx_folder, 'recursive': False})
      if not result:
          return 0
      pushed = 0
      for entry in result.get('entries', []):
          if entry['.tag'] != 'file':
              continue
          name = entry['name']
          if entry.get('size', 0) > max_size:
              continue
          if exts and not any(name.endswith(e) for e in exts):
              continue
          content = dbx_download(entry['path_display'])
          if content is None:
              continue
          gh_path = f'{gh_prefix}/{name}'
          existing = gh_get(gh_path)
          sha = existing.get('sha') if existing else None
          # Only push if content changed
          if existing:
              remote_content = base64.b64decode(existing.get('content', '').replace('\n', ''))
              if remote_content == content:
                  continue
          msg = f'sync(dropbox): update {gh_path}'
          r = gh_put(gh_path, content, msg, sha)
          if r and r.get('commit'):
              log(f'  Pushed {gh_path} ({len(content)} bytes)')
              pushed += 1
          time.sleep(0.3)
      return pushed

  def push_status_to_dropbox():
      """Push live training status + logs from instance to Dropbox."""
      files = [
          ('/workspace/titanai/data/live_status.json',         '/workspace/titanai_status/live_status.json'),
          ('/workspace/titanai/data/training_log_latest.txt',  '/workspace/titanai_status/training_log_latest.txt'),
          ('/workspace/titanai/data/corpus_progress_latest.txt','/workspace/titanai_status/corpus_progress_latest.txt'),
      ]
      # Also grab latest training log
      import glob
      for pat in ['/workspace/titanai/logs/titan_1b/upgrade_*/training.log']:
          for f in glob.glob(pat):
              files.append((f, f'/workspace/titanai_status/{os.path.basename(os.path.dirname(f))}_training.log'))

      uploaded = 0
      for local, remote in files:
          if not os.path.exists(local):
              continue
          content = open(local, 'rb').read()
          if len(content) > 2 * 1024 * 1024:
              content = content[-500*1024:]  # last 500KB only
          r = dbx_upload(remote, content)
          if r:
              uploaded += 1
      if uploaded:
          log(f'Pushed {uploaded} status files to Dropbox')

  if __name__ == '__main__':
      if not DBX_TOKEN:
          log('ERROR: No DBX_ACCESS_TOKEN in .dropbox_sync_env — aborting')
          sys.exit(1)
      if not GH_TOKEN:
          log('ERROR: No GH_TOKEN in .dropbox_sync_env — aborting')
          sys.exit(1)

      log('=== Dropbox ↔ GitHub sync starting ===')

      # 1. Dropbox scripts → GitHub
      n = sync_dropbox_folder_to_github(
          '/workspace/titanai/scripts', 'scripts',
          exts=['.py', '.sh', '.yaml', '.yml', '.json', '.md', '.txt']
      )
      log(f'Scripts: {n} files synced')

      # 2. Dropbox configs → GitHub
      n = sync_dropbox_folder_to_github(
          '/workspace/titanai/configs', 'configs',
          exts=['.yaml', '.yml', '.json', '.toml', '.ini', '.cfg', '.md']
      )
      log(f'Configs: {n} files synced')

      # 3. Dropbox data (small files) → GitHub data/dropbox_sync/
      n = sync_dropbox_folder_to_github(
          '/workspace/titanai/data', 'data/dropbox_sync',
          max_size=200*1024,
          exts=['.json', '.md', '.txt', '.log', '.yaml']
      )
      log(f'Data files: {n} files synced')

      # 4. Instance status → Dropbox
      push_status_to_dropbox()

      log('=== Sync complete ===')
  