"""Dependency-free structural checks for the checked-in static website."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
import hashlib
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
VOID = set("area base br col embed hr img input link meta param source track wbr".split())


class AuditParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.ids, self.references, self.placeholders = [], set(), [], set()
        self.errors, self.external, self.images = [], set(), []
        self.h1_count = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag not in VOID:
            self.stack.append(tag)
        if "id" in a:
            if a["id"] in self.ids:
                self.errors.append("Duplicate id: " + a["id"])
            self.ids.add(a["id"])
        self.h1_count += tag == "h1"
        if "data-placeholder" in a:
            self.placeholders.add(a["data-placeholder"])
        for key in ("src", "href", "data-zoom", "srcset"):
            if a.get(key):
                self.references.append((tag, key, a[key]))
        for key in ("aria-labelledby", "aria-describedby", "data-copy"):
            for value in a.get(key, "").split():
                self.references.append((tag, "idref", "#" + value))
        if tag == "img":
            self.images.append(a)
            if "alt" not in a or (a.get("src") and not a["alt"].strip()):
                self.errors.append("Missing image alternative text")
        if tag in ("script", "link") and a.get("src", a.get("href", "")).startswith("http"):
            self.errors.append("External runtime dependency")

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append("Unbalanced end tag: " + tag)
        else:
            self.stack.pop()


def main():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    parser = AuditParser()
    parser.feed(html)
    errors = parser.errors
    if parser.stack:
        errors.append("Unclosed tags: " + repr(parser.stack))
    if parser.h1_count != 1:
        errors.append("Expected exactly one h1")
    if not html.lower().startswith("<!doctype html>") or '<html lang="en">' not in html:
        errors.append("Missing HTML5 declaration / language")
    local_files = set()
    for tag, attr, ref in parser.references:
        url = urlsplit(ref)
        if url.scheme in ("http", "https"):
            parser.external.add(ref)
            continue
        if url.scheme or ref.startswith(("/", "\\")):
            errors.append("Non-relative local resource: " + ref)
            continue
        if url.path:
            target = (ROOT / unquote(url.path)).resolve()
            if not target.is_relative_to(ROOT) or not target.is_file():
                errors.append("Missing / out-of-docs file: " + ref)
            local_files.add(ref)
        if not url.path and url.fragment and url.fragment not in parser.ids:
            errors.append("Unresolved anchor/id reference: " + ref)
    css = (ROOT / "assets/css/style.css").read_text(encoding="utf-8")
    if "@import" in css or re.search(r"url\s*\(", css):
        errors.append("Unexpected CSS resource dependency; extend audit first")
    registry = json.loads((ROOT / "assets/data/project.json").read_text(encoding="utf-8"))
    registered = {p["id"] for p in registry["placeholders"]}
    if parser.placeholders - registered:
        errors.append("Unregistered placeholders: " + repr(parser.placeholders - registered))
    readme = (ROOT.parent / "README.md").read_text(encoding="utf-8")
    for ident in set(re.findall(r"TODO_[A-Z0-9_]+", readme)) - registered:
        errors.append("Unregistered README placeholder: " + ident)
    for group in registry["datasets"]:
        expected = '<tr><th scope="row">' + group["name"] + '</th>'
        expected += ''.join(f'<td>{group["records"][s]:,}</td>' for s in ("train", "val", "test")) + '</tr>'
        if expected not in html:
            errors.append("HTML/JSON inventory drift: " + group["name"])
    for split in ("train", "val", "test"):
        if sum(g["records"][split] for g in registry["datasets"]) != registry["inventory_totals"][split]:
            errors.append("Inventory total mismatch: " + split)
    for path in (ROOT / "assets/images").rglob("*.svg"):
        svg = path.read_text(encoding="utf-8")
        tree = ET.fromstring(svg)
        if not tree.tag.endswith("svg") or any(x in svg.lower() for x in ("<image", "<script", "foreignobject", "base64", "file://")):
            errors.append("Unexpected embedded raster/executable SVG content: " + str(path))
    provenance = json.loads((ROOT / "qa/asset_provenance.json").read_text(encoding="utf-8"))
    for asset in provenance["assets"]:
        expected_hash = asset.get("copy_sha256", asset.get("sha256"))
        if hashlib.sha256((ROOT / asset["path"]).read_bytes()).hexdigest() != expected_hash:
            errors.append("Asset provenance drift: " + asset["path"])
    secret_patterns = [r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", r"gh[pousr]_[A-Za-z0-9]{25,}", r"sk-[A-Za-z0-9_-]{24,}", r"hf_[A-Za-z0-9]{25,}"]
    for path in [ROOT / "index.html", ROOT.parent / "README.md", *list((ROOT / "assets").rglob("*"))]:
        if path.is_file() and path.suffix in (".html", ".md", ".js", ".css", ".svg", ".json"):
            content = path.read_text(encoding="utf-8")
            if any(re.search(pattern, content) for pattern in secret_patterns):
                errors.append("Possible credential pattern in " + str(path))
    report = {"date": "2026-11-02", "passed": not errors, "errors": errors,
              "html_structure": "balanced tags, unique IDs, single h1, language and doctype",
              "checked_local_resources": sorted(local_files), "anchors_and_idrefs": "all resolved" if not errors else "see errors",
              "external_links": sorted(parser.external), "registered_pending_items": len(registered),
              "asset_provenance": "hashes checked", "scope": "Structural/static check, not W3C validation or clinical/privacy certification"}
    (ROOT / "qa/static_checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
