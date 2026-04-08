import pathlib
import re
import sys


PATTERNS = {
    "google_api_key_like": re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    "openai_key_like": re.compile(r"sk-[0-9A-Za-z]{20,}"),
}


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent
    hits = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name in {".git", ".cursor"}:
            continue
        if p.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".txt", ".json", ".toml", ".ini"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for name, rx in PATTERNS.items():
            if rx.search(text):
                hits.append((name, str(p)))

    if hits:
        print("FOUND_POSSIBLE_SECRETS")
        for name, path in hits:
            print(f"- {name}: {path}")
        return 2

    print("OK_NO_SECRETS_FOUND")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

