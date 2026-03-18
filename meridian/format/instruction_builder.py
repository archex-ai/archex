# meridian/format/instruction_builder.py
import xml.etree.ElementTree as ET
import json, re
from pathlib import Path

def parse_pubmed_xml(xml_text):
    """Extract title and abstract from PubMed XML."""
    try:
        root = ET.fromstring(xml_text)
        articles = []
        for article in root.findall(".//PubmedArticle"):
            title = article.findtext(".//ArticleTitle", "").strip()
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(p.text or "" for p in abstract_parts).strip()
            if title and abstract and len(abstract) > 100:
                articles.append({"title": title, "abstract": abstract})
        return articles
    except:
        return []

def abstract_to_instruction(title, abstract):
    """Convert a PubMed abstract into a training instruction pair."""
    # Multiple question formats for variety
    templates = [
        {
            "user": f"What does current medical literature say about: {title}?",
            "assistant": f"Based on published research: {abstract[:600]}"
        },
        {
            "user": f"Summarize the key clinical findings regarding {title.lower()}.",
            "assistant": f"The key findings are: {abstract[:500]}"
        },
        {
            "user": f"As a medical student, I need to understand: {title}. Can you explain?",
            "assistant": f"Here is a clinical summary: {abstract[:600]}"
        },
    ]
    import random
    t = random.choice(templates)
    return {
        "system": (
            "You are ARCHEX MedLM, a medical AI assistant. "
            "You provide accurate, evidence-based medical information. "
            "Always remind users to consult a qualified doctor for personal medical decisions."
        ),
        "user": t["user"],
        "assistant": t["assistant"]
    }

def build_dataset(raw_dir, output_path, max_samples=50000):
    """Build full training dataset from raw PubMed XML files."""
    pairs = []
    for xml_file in Path(raw_dir).glob("*.xml"):
        articles = parse_pubmed_xml(xml_file.read_text())
        for a in articles:
            pairs.append(abstract_to_instruction(a["title"], a["abstract"]))
            if len(pairs) >= max_samples:
                break
    
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    
    print(f"Built {len(pairs)} instruction pairs")
    print(f"Saved to {output_path}")