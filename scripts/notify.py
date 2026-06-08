#!/usr/bin/env python3
"""
notify.py — TitanAI training email + GitHub status notifications.
Called after each training phase completes or if training fails.

Usage:
    python3 scripts/notify.py --phase "Phase 1 complete" --status ok --detail "305k steps done"
    python3 scripts/notify.py --phase "Training FAILED" --status error --detail "OOM at step 12000"

Env vars required for email:
    NOTIFY_TO    = leego972@gmail.com   (set automatically)
    SMTP_USER    = your-sender@gmail.com
    SMTP_PASS    = your-gmail-app-password  (16-char App Password, not your login password)
    GITHUB_TOKEN = TITAN_GITHUB_TOKEN (for GitHub status push — fallback always works)
"""
import argparse, os, json, urllib.request, smtplib, base64
from email.mime.text import MIMEText
from datetime import datetime, timezone

NOTIFY_TO     = os.environ.get("NOTIFY_TO",    "leego972@gmail.com")
SMTP_USER     = os.environ.get("SMTP_USER",    "")
SMTP_PASS     = os.environ.get("SMTP_PASS",    "")
GH_TOKEN      = os.environ.get("TITAN_GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN", ""))
REPO          = "leego972/titanai"
STATUS_FILE   = "TRAINING_STATUS.md"

def send_email(subject: str, body: str) -> bool:
    if not SMTP_USER or not SMTP_PASS:
        print("[notify] SMTP_USER/SMTP_PASS not set — skipping email")
        return False
    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = NOTIFY_TO
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as s:
            s.ehlo()
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_USER, [NOTIFY_TO], msg.as_string())
        print(f"[notify] Email sent to {NOTIFY_TO}: {subject}")
        return True
    except Exception as e:
        print(f"[notify] Email failed: {e}")
        return False

def push_github_status(phase: str, status: str, detail: str) -> bool:
    if not GH_TOKEN:
        print("[notify] No GH_TOKEN — skipping GitHub push")
        return False
    try:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        icon = "✅" if status == "ok" else "❌" if status == "error" else "🔄"
        content = f"# TitanAI Training Status\n\nLast updated: {ts}\n\n## {icon} {phase}\n\n{detail}\n"
        encoded = base64.b64encode(content.encode()).decode()

        # Get current SHA of status file (if it exists)
        sha = None
        try:
            req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/contents/{STATUS_FILE}")
            req.add_header("Authorization", f"token {GH_TOKEN}")
            req.add_header("Accept", "application/vnd.github.v3+json")
            with urllib.request.urlopen(req, timeout=8) as r:
                sha = json.loads(r.read()).get("sha")
        except Exception:
            pass

        payload = {"message": f"[training] {phase}", "content": encoded}
        if sha:
            payload["sha"] = sha
        req2 = urllib.request.Request(
            f"https://api.github.com/repos/{REPO}/contents/{STATUS_FILE}",
            data=json.dumps(payload).encode(), method="PUT"
        )
        req2.add_header("Authorization", f"token {GH_TOKEN}")
        req2.add_header("Accept", "application/vnd.github.v3+json")
        req2.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req2, timeout=12) as r:
            resp = json.loads(r.read())
        print(f"[notify] GitHub status pushed: {resp['commit']['sha'][:10]}")
        return True
    except Exception as e:
        print(f"[notify] GitHub push failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",  required=True)
    parser.add_argument("--status", default="ok", choices=["ok", "error", "info"])
    parser.add_argument("--detail", default="")
    args = parser.parse_args()

    ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    icon  = {"ok": "✅", "error": "❌", "info": "🔄"}.get(args.status, "🔄")
    subj  = f"TitanAI | {icon} {args.phase}"
    body  = (
        f"TitanAI Training Notification\n"
        f"{'='*40}\n"
        f"Phase   : {args.phase}\n"
        f"Status  : {args.status.upper()}\n"
        f"Time    : {ts}\n"
        f"Detail  : {args.detail}\n"
        f"{'='*40}\n"
        f"Check logs: /workspace/logs/titanai_full/master.log\n"
    )

    email_sent  = send_email(subj, body)
    github_sent = push_github_status(args.phase, args.status, args.detail or "No detail provided.")

    if not email_sent and not github_sent:
        print(f"[notify] WARNING — no notification sent. Phase: {args.phase}")
    else:
        print(f"[notify] Done. email={email_sent} github={github_sent}")

if __name__ == "__main__":
    main()
