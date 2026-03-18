# meridian/ingest/pubmed_fetcher.py
import requests, time, json
from pathlib import Path

DOMAINS = [
    "clinical diagnosis AND treatment",
    "drug interactions pharmacology",
    "emergency medicine acute care",
    "internal medicine differential diagnosis",
    "neurology clinical",
    "cardiology clinical management",
    "infectious disease treatment",
    "pharmacology drug dosage",
    "medical education clinical skills",
    "rehabilitation medicine prosthetics",
]

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def search_pubmed(query, max_results=5000):
    """Get PMIDs for a query."""
    r = requests.get(f"{BASE}/esearch.fcgi", params={
        "db": "pubmed", "term": query,
        "retmax": max_results, "retmode": "json",
        "usehistory": "y", "sort": "relevance"
    }, timeout=30)
    return r.json()["esearchresult"]["idlist"]

def fetch_abstracts(pmids, batch=100):
    """Fetch abstracts in batches. Returns list of dicts."""
    results = []
    for i in range(0, len(pmids), batch):
        batch_ids = pmids[i:i+batch]
        r = requests.get(f"{BASE}/efetch.fcgi", params={
            "db": "pubmed", "id": ",".join(batch_ids),
            "rettype": "abstract", "retmode": "xml"
        }, timeout=60)
        # Parse XML — extract title + abstract text
        # (full parser in next task)
        results.append(r.text)
        time.sleep(0.35)  # stay under 3 req/sec limit
        if i % 1000 == 0:
            print(f"  Fetched {i}/{len(pmids)}")
    return results

if __name__ == "__main__":
    all_pmids = []
    for domain in DOMAINS:
        print(f"Searching: {domain}")
        pmids = search_pubmed(domain, max_results=5000)
        all_pmids.extend(pmids)
        print(f"  Got {len(pmids)} PMIDs")
    
    # Deduplicate
    all_pmids = list(set(all_pmids))
    print(f"Total unique PMIDs: {len(all_pmids)}")
    
    Path("./data/raw").mkdir(parents=True, exist_ok=True)
    with open("./data/raw/pmids.json", "w") as f:
        json.dump(all_pmids, f)
    print("Saved PMIDs. Run fetcher next.")