import os
from pathlib import Path

PRIVATE_PROMPT_PATH = "/workspace/titanai/prompts/system_prompt.txt"
PUBLIC_PROMPT_PATH  = "/workspace/titanai/prompts/public_system_prompt.txt"

# Inline fallback for public mode — Australian law applies when not in cyber/owner mode
PUBLIC_PROMPT_INLINE = """You are Titan, an AI assistant built by Lee Ida.

You are operating in Public Mode. In this mode you abide by the laws of Australia at all times. You answer questions helpfully, directly, and honestly within those legal bounds. You do not assist with any activity that is illegal under Australian federal or state law. You are straightforward and do not add unnecessary caveats or apologies."""

def get_system_prompt(is_private=False):
    if is_private:
        if os.path.exists(PRIVATE_PROMPT_PATH):
            with open(PRIVATE_PROMPT_PATH, 'r') as f:
                return f.read()
        return ""

    if os.path.exists(PUBLIC_PROMPT_PATH):
        with open(PUBLIC_PROMPT_PATH, 'r') as f:
            return f.read()

    return PUBLIC_PROMPT_INLINE

def is_request_private(request_headers, auth_token):
    from .identity import check_identity
    if auth_token and check_identity(auth_token):
        return True

    origin  = request_headers.get("Origin",  "").lower()
    referer = request_headers.get("Referer", "").lower()

    public_domains = ["archibaldtitan.com", "virelle.life"]
    for domain in public_domains:
        if domain in origin or domain in referer:
            return False

    return False
