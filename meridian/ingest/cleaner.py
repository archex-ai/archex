# meridian/ingest/cleaner.py
import hashlib, re

def is_good_abstract(title, abstract):
    """Returns True if this abstract is worth training on."""
    # Minimum length
    if len(abstract) < 150: return False
    if len(title) < 10: return False
    
    # Must be mostly English
    english_chars = sum(1 for c in abstract if ord(c) < 128)
    if english_chars / len(abstract) < 0.85: return False
    
    # Must contain at least some clinical content signals
    clinical_signals = [
        "patient", "treatment", "diagnosis", "clinical",
        "symptoms", "therapy", "mg", "dose", "risk",
        "study", "results", "conclusion", "evidence"
    ]
    abstract_lower = abstract.lower()
    signal_count = sum(1 for s in clinical_signals if s in abstract_lower)
    if signal_count < 2: return False
    
    # No pure methods papers
    methods_only = ["we describe a method", "algorithm for", "software for"]
    if any(m in abstract_lower for m in methods_only): return False
    
    return True

def deduplicate(pairs):
    """Remove near-duplicate abstracts using title hash."""
    seen = set()
    unique = []
    for p in pairs:
        h = hashlib.md5(p["user"][:100].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(p)
    print(f"Dedup: {len(pairs)} → {len(unique)} unique pairs")
    return unique