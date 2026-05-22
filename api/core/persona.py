import os
from pathlib import Path

# Paths to the different prompts
PRIVATE_PROMPT_PATH = "/workspace/titanai/prompts/system_prompt.txt"
PUBLIC_PROMPT_PATH = "/workspace/titanai/prompts/public_system_prompt.txt"

def get_system_prompt(is_private=False):
    """
    Returns the appropriate system prompt based on the authentication level.
    """
    if is_private:
        if os.path.exists(PRIVATE_PROMPT_PATH):
            with open(PRIVATE_PROMPT_PATH, 'r') as f:
                return f.read()
    
    # Default to Public Prompt if not private or private prompt is missing
    if os.path.exists(PUBLIC_PROMPT_PATH):
        with open(PUBLIC_PROMPT_PATH, 'r') as f:
            return f.read()
            
    return "You are a helpful AI assistant."

def is_request_private(request_headers, auth_token):
    """
    Logic to determine if the request is from the owner (Private) or a website user (Public).
    """
    # 1. Check if the special identity passphrase was used (Owner Override)
    # This allows the owner to use private mode even on public websites
    from .identity import check_identity
    if check_identity(auth_token):
        return True
        
    # 2. Check for Public Website Origins
    origin = request_headers.get("Origin", "").lower()
    referer = request_headers.get("Referer", "").lower()
    
    public_domains = ["archibaldtitan.com", "virelle.life"]
    
    for domain in public_domains:
        if domain in origin or domain in referer:
            return False # Public Mode for these websites
            
    # Default to Public Mode for safety if not explicitly verified
    return False
