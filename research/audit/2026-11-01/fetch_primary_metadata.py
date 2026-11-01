"""Fetch a dated bibliographic snapshot; do not download or reproduce full papers.

Run from the repository root with network access. arXiv supplies latest and
selected-version metadata; Crossref supplies registration metadata for publication
DOIs. A failed request is recorded as a retrieval failure, not evidence of fraud.
"""

import concurrent.futures
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}


def fetch(url):
    request = urllib.request.Request(url, headers={
        "User-Agent": "ThesisBibliographyAudit/1.0 (bibliographic verification)"
    })
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def arxiv_query(ids):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "id_list": ",".join(ids), "max_results": len(ids)
    })
    feed = ET.fromstring(fetch(url))
    result = []
    for entry in feed.findall("a:entry", NS):
        value = lambda tag: " ".join((entry.findtext(tag, default="", namespaces=NS)).split())
        result.append({
            "id": value("a:id"), "title": value("a:title"),
            "authors": [n.findtext("a:name", namespaces=NS) for n in entry.findall("a:author", NS)],
            "published": value("a:published"), "updated": value("a:updated"),
            "comment": value("x:comment"), "journal_ref": value("x:journal_ref"),
            "doi": value("x:doi"),
            "links": [n.attrib for n in entry.findall("a:link", NS)],
        })
    return {"source_url": url, "entries": result}


def doi_query(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    try:
        message = json.loads(fetch(url))["message"]
        fields = ("DOI", "title", "author", "type", "publisher", "container-title",
                  "published", "published-online", "published-print", "event", "page",
                  "volume", "issue", "relation", "update-to", "updated-by")
        return {"doi": doi, "source_url": url, "status": "retrieved",
                "metadata": {k: message[k] for k in fields if k in message}}
    except Exception as exc:
        return {"doi": doi, "source_url": url, "status": "retrieval_failed",
                "error": f"{type(exc).__name__}: {exc}"}


def main():
    registry = json.loads((ROOT / "references/papers.json").read_text())
    records = registry["papers"]
    arxiv_ids = [p["arxiv_id"] for p in records if p.get("arxiv_id")]
    snapshot = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "metadata only; no replication or proof of methodological reliability",
                "latest_batches": [], "selected_batches": []}
    for start in range(0, len(arxiv_ids), 30):
        if snapshot["latest_batches"]:
            time.sleep(3)
        snapshot["latest_batches"].append(arxiv_query(arxiv_ids[start:start + 30]))
    latest = {}
    for batch in snapshot["latest_batches"]:
        for entry in batch["entries"]:
            match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?$", entry["id"])
            if match:
                latest[match.group(1)] = entry
    selected_ids = []
    unversioned = []
    changes = []
    for record in records:
        arxiv_id = record.get("arxiv_id")
        if not arxiv_id:
            continue
        entry = latest.get(arxiv_id)
        if not record.get("arxiv_version"):
            unversioned.append({"id": record["id"], "arxiv_id": arxiv_id,
                                "latest": entry["id"] if entry else None})
            continue
        selected = arxiv_id + (record.get("arxiv_version") or "")
        if entry and not entry["id"].endswith("/" + selected):
            selected_ids.append(selected)
            changes.append({"id": record["id"], "selected": selected,
                            "latest": entry["id"], "registry_title": record["title"],
                            "latest_title": entry["title"]})
    for start in range(0, len(selected_ids), 30):
        time.sleep(3)
        snapshot["selected_batches"].append(arxiv_query(selected_ids[start:start + 30]))
    snapshot["versions_differing_from_registry"] = changes
    snapshot["unversioned_arxiv_records"] = unversioned
    (OUT / "primary-metadata.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    dois = {p["doi"] for p in records if p.get("doi") and not p["doi"].lower().startswith("10.48550/")}
    for entry in latest.values():
        doi = entry["doi"]
        if doi.startswith("10.") and not doi.lower().startswith("10.48550/"):
            dois.add(doi)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        publisher = list(pool.map(doi_query, sorted(dois)))
    (OUT / "publisher-metadata.json").write_text(json.dumps({
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": publisher,
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"registry_records": len(records), "arxiv_ids": len(arxiv_ids),
                      "arxiv_entries_retrieved": len(latest),
                      "different_latest_versions": len(changes),
                      "publication_dois": len(publisher),
                      "publication_dois_retrieved": sum(p["status"] == "retrieved" for p in publisher)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
