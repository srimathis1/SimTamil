
import json
from pathlib import Path

base = Path("export/participants/Mounees")

files = [
    base / "all.json",
    base / "wave2" / "wave2.json",
]


def show_structure(obj, indent=0, limit=5):
    prefix = " " * indent

    if isinstance(obj, dict):
        print(prefix + f"dict keys: {list(obj.keys())[:20]}")
        for key, value in list(obj.items())[:limit]:
            print(prefix + f"[{key}]")
            show_structure(value, indent + 2, limit)
    elif isinstance(obj, list):
        print(prefix + f"list length: {len(obj)}")
        for item in obj[:2]:
            show_structure(item, indent + 2, limit)
    else:
        print(prefix + repr(obj)[:180])


for file in files:
    print("\n" + "=" * 70)
    print(f"FILE: {file}")
    print("=" * 70)

    if not file.exists():
        print("FILE NOT FOUND")
        continue

    with file.open(encoding="utf-8") as f:
        data = json.load(f)

    show_structure(data)
