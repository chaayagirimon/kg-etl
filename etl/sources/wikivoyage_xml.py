# etl/wikivoyage_xml.py
from __future__ import annotations

import bz2
import io
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import lxml.etree as ET
import pandas as pd

# Templates commonly used for POIs in Wikivoyage pages
LISTING_TEMPLATES = {"see", "do", "eat", "drink", "sleep", "buy", "listing", "marker"}

# Optional: use mwparserfromhell if available (more robust than regex for nested braces)
try:
    import mwparserfromhell  # type: ignore
    HAS_MWPH = True
except Exception:
    HAS_MWPH = False


# --------------------------
# Helpers
# --------------------------
def _open_xml(path: str | os.PathLike) -> io.BufferedReader:
    p = str(path)
    if p.endswith(".bz2"):
        return bz2.open(p, "rb")
    return open(p, "rb")


def _tag(name: str) -> str:
    """Wildcard namespace tag, e.g. '{*}page' works for export-0.10/0.11/etc."""
    return "{*}" + name


def _get_text(elem: ET._Element, xpath: str) -> Optional[str]:
    node = elem.find(xpath)
    if node is not None and node.text is not None:
        return node.text
    return None


def _iter_pages(xml_file: str | os.PathLike) -> Iterator[ET._Element]:
    """Stream over <page> elements with namespace-agnostic matching."""
    with _open_xml(xml_file) as fh:
        for _, elem in ET.iterparse(fh, events=("end",), tag=_tag("page")):
            yield elem
            # Free memory: clear element and trim processed siblings
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]


_float_rx = re.compile(r"[-+]?\d+(?:\.\d+)?")

def _to_float_maybe(s: object) -> Optional[float]:
    if s is None:
        return None
    t = str(s).strip()
    # Extract first float-looking number (handles '48.2°N' or embedded text)
    m = _float_rx.search(t)
    try:
        return float(m.group(0)) if m else None
    except Exception:
        return None


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-:_.,]+", "", s)
    return s[:120]


# --------------------------
# Wikitext parsing
# --------------------------
def _parse_wikitext_regex(text: str) -> List[Dict[str, str]]:
    """
    Lightweight parser for {{template|k=v|k=v}} blocks.
    Not perfect for all nested cases, but fast and works well enough.
    """
    results: List[Dict[str, str]] = []
    for m in re.finditer(r"\{\{([^\{\}\|]+)\|([^\}]*)\}\}", text, flags=re.IGNORECASE | re.DOTALL):
        tname = m.group(1).strip().lower()
        if tname not in LISTING_TEMPLATES:
            continue
        params = m.group(2)
        fields: Dict[str, str] = {"_template": tname}
        for part in re.split(r"\s*\|\s*", params):
            if "=" in part:
                k, v = part.split("=", 1)
                fields[k.strip().lower()] = v.strip()
        results.append(fields)
    return results


def _parse_wikitext_mwph(text: str) -> List[Dict[str, str]]:
    code = mwparserfromhell.parse(text)
    results: List[Dict[str, str]] = []
    for tpl in code.filter_templates(recursive=True):
        name = str(tpl.name).strip().lower()
        if name not in LISTING_TEMPLATES:
            continue
        fields: Dict[str, str] = {"_template": name}
        for p in tpl.params:
            k = str(p.name).strip().lower()
            v = str(p.value).strip()
            fields[k] = v
        results.append(fields)
    return results


def _extract_listings_from_text(wikitext: str) -> List[Dict[str, str]]:
    if not wikitext:
        return []
    if HAS_MWPH:
        try:
            return _parse_wikitext_mwph(wikitext)
        except Exception:
            # fallback to regex if mwparser fails on odd constructs
            pass
    return _parse_wikitext_regex(wikitext)


# --------------------------
# Public loader for unify_sqlite.py
# --------------------------
def load_wikivoyage_xml(xml_path: Optional[str]) -> pd.DataFrame:
    """
    Parse a Wikivoyage dump and return a DataFrame with:
      place_id, source, name, lat, lon, type, address, city_hint

    - Accepts .xml or .xml.bz2
    - Uses namespace-agnostic iterparse so tag version changes don't break ingestion
    - Accepts both 'long' and 'lon' for longitude
    - Includes '{{marker}}' template (often used inline)
    """
    if not xml_path:
        return pd.DataFrame(columns=["place_id","source","name","lat","lon","type","address","city_hint"])

    xml_path = str(xml_path)
    if not Path(xml_path).exists():
        print(f"[wikivoyage] file not found: {xml_path}")
        return pd.DataFrame(columns=["place_id","source","name","lat","lon","type","address","city_hint"])

    rows: List[Dict[str, object]] = []
    n_pages = 0

    for page in _iter_pages(xml_path):
        n_pages += 1

        # Only main namespace
        ns = _get_text(page, _tag("ns"))
        if ns and ns.strip() != "0":
            continue

        # Skip redirects
        if page.find(_tag("redirect")) is not None:
            continue

        title = _get_text(page, _tag("title")) or ""
        text = _get_text(page, ".//" + _tag("text"))
        if not text:
            continue

        for t in _extract_listings_from_text(text):
            # Pull core fields
            name = t.get("name") or t.get("alt") or t.get("title")
            lat = _to_float_maybe(t.get("lat"))
            lon = _to_float_maybe(t.get("long") or t.get("lon"))  # prefer 'long'
            if not name or lat is None or lon is None:
                continue

            addr = (
                t.get("address")
                or t.get("addr")
                or t.get("street")
                or t.get("directions")
                or None
            )
            ttype = t.get("_template") or "listing"

            # Build a reasonably unique id (title + name + rounded coords)
            pid = f"wv:{_slug(title)}:{_slug(str(name))}:{lat:.5f},{lon:.5f}"

            # NEW: capture best-effort description
            desc = None
            for k in ("content", "description", "desc", "summary", "alt"):
                val = t.get(k)
                if val and str(val).strip():
                    desc = str(val).strip()
                    break

            rows.append(
                {
                    "place_id": pid,
                    "source": "wikivoyage",
                    "name": str(name),
                    "lat": float(lat),
                    "lon": float(lon),
                    "type": ttype,
                    "address": addr,
                    "city_hint": title,  # helps your assign_city_slug fallback
                    "desc": desc, 
                }
            )

    df = pd.DataFrame(rows, columns=["place_id","source","name","lat","lon","type","address","city_hint","desc"])
    print(f"[wikivoyage] pages_scanned={n_pages} listings_parsed={len(df)} from {os.path.basename(xml_path)}")
    return df
