import json
from pathlib import Path
from typing import Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert "cities" in cfg and isinstance(cfg["cities"], list), "Config must contain 'cities' list"
    cfg.setdefault("defaults", {})
    cfg.setdefault("overrides", {})
    return cfg
