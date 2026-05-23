#!/usr/bin/env python3
import os
import sys
import json
import time
import requests
import argparse

# Secrets are read from environment variables for security
VAST_API_KEY = os.environ.get("VAST_AI_API_KEY")
GITHUB_TOKEN = os.environ.get("TITAN_GITHUB_TOKEN")
TITAN_API_KEY = "TitanArcV1-SecureKey-XK9-2026-Production"

def vast_api_call(method, endpoint, data=None, params=None):
    url = f"https://console.vast.ai/api/v0{endpoint}"
    headers = {"Authorization": f"Bearer {VAST_API_KEY}"}
    if method == "GET":
        response = requests.get(url, headers=headers, params=params)
    elif method == "PUT":
        response = requests.put(url, headers=headers, json=data)
    elif method == "POST":
        response = requests.post(url, headers=headers, json=data)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    if response.status_code not in [200, 201]:
        print(f"Error calling {url}: {response.status_code} {response.text}")
        return None
    return response.json()

def find_best_offer():
    # Search for 2x RTX 4090, verified, rentable
    query = {
        "gpu_name": {"eq": "RTX 4090"},
        "num_gpus": {"eq": 2},
        "verified": {"eq": True},
        "rentable": {"eq": True}
    }
    # The bundles endpoint expects 'q'
    params = {"q": json.dumps(query)}
    results = vast_api_call("GET", "/bundles/", params=params)
    
    if results and results.get("offers"):
        # Manual sort by dph_total to get the cheapest
        results["offers"].sort(key=lambda x: x.get("dph_total", 999))
        
    if not results or not results.get("offers"):
        print("No suitable 2x RTX 4090 verified offers found.")
        return None
    
    # Filter for reliability > 98%
    offers = [o for o in results["offers"] if o.get("reliability", 0) > 0.98]
    if not offers:
        offers = results["offers"] # fallback to best available
        
    return offers[0]

def launch_instance(offer_id):
    # Setup onstart script to clone and start 13b training
    onstart = f"""
mkdir -p /root/.ssh && chmod 700 /root/.ssh
echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCd/3CKl7qBcmIsA5YZEs7xOcCpeWvJ1dOeS/m+tJLfKIoYiL96HCODA7oVexuLU/3dSgPP3VYf2+iUOCAJHJNzfcbUbbk8ArdAbyVsMAdEpl4rmf/rcih+A8C07d2snoFlVTsdOMzXJHE3m19DMPWk/hoys+31a0jh5ZQfh9YnrOPOqYnqAv2h7CgZvUFeRgttxbii3wcmZBW74r6KzqYmf13+uKmM3Eu9z0Hf934xCbGQEL5Z13Awt35WC+Hz/mqtPDurNfjldaUiZxCytfh8jQuNr+PhDG7QxkEylrf0dDeQldOMFLeArCeJHbTPpJAcqhApFxPqPV+Idu4VMymV+oSxGAOAcgcI2VA3X2wBqYhxAsBhQLIPhKoIm2Prrpt5072eqEc5IKS161F0wapdVMYAb2bilzTgO00lwEwuxotylG68IXX432eFptMEril+ebIWDeTWly47inTn8UTyVEuIhuejp1NXlnv/t3XZQzI0i5V3F3AoMLZj2Lu+AybSQ7hja4z1s8U1X985iYOgPkCZDuCp3BqYW4frsh+Mn/hRl2Sl5hNq+szyIzQkVyGB2nQvVUCoGkbAq3q7mI4aBL2ji8+Aqbh+HLw9NRZz0wO4kWMKuB4UklsHfDGlcrdjwLSkNKNsiNE+139RvJMm571QcqGiPo1B0F0zsOqt4Q== titanai-agent" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
service ssh restart 2>/dev/null || true
apt-get update -qq && apt-get install -y -qq git
mkdir -p /workspace
git clone https://{GITHUB_TOKEN}@github.com/leego972/titanai.git /workspace/titanai
cd /workspace/titanai
export TITAN_REQUIRE_AUTH=true
export TITAN_API_KEY={TITAN_API_KEY}
export TITAN_CONFIG=configs/titan_13b.yaml
export TITAN_CHECKPOINT_DIR=checkpoints/titan_13b_pretrain
export TITAN_GITHUB_TOKEN={GITHUB_TOKEN}
bash scripts/vast_startup.sh 2>&1 | tee /var/log/titanai.log
    """
    
    data = {
        "image": "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel",
        "disk": 100.0,
        "label": "titanai-13b-auto",
        "onstart": onstart.strip(),
        "runtype": "ssh"
    }
    
    print(f"Renting offer {offer_id}...")
    result = vast_api_call("PUT", f"/asks/{offer_id}/", data=data)
    if result and result.get("success"):
        print(f"Successfully rented instance! Contract ID: {result.get('new_contract')}")
        return result.get("new_contract")
    return None

def main():
    if not VAST_API_KEY or not GITHUB_TOKEN:
        print("Error: VAST_AI_API_KEY and TITAN_GITHUB_TOKEN environment variables must be set.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Automate TitanAI 13B training launch on Vast.ai")
    parser.add_argument("--dry-run", action="store_true", help="Search but don't rent")
    args = parser.parse_args()
    
    print("Searching for the best 2x RTX 4090 verified offer...")
    offer = find_best_offer()
    
    if not offer:
        print("No suitable offer found.")
        sys.exit(1)
        
    print(f"Found best offer: ID {offer['id']} | GPUs: {offer['num_gpus']}x {offer['gpu_name']} | Price: ${offer['dph_total']:.3f}/hr | Reliability: {offer['reliability']:.2%}")
    
    if args.dry_run:
        print("Dry run complete. No instance rented.")
    else:
        contract_id = launch_instance(offer['id'])
        if contract_id:
            print(f"13B Training launch initiated on instance {contract_id}.")
        else:
            print("Failed to launch instance.")

if __name__ == "__main__":
    main()
