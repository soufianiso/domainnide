#!/usr/bin/env python3
"""Two-word domain idea generator: find_words + tldx.

Takes two seed words, expands each into a list of related words with
find_words (the 7 GB model is loaded once and queried per seed), then
hands both lists to tldx --cross, which checks the availability of
every combination in both orders (a+b and b+a).

Usage (any of these work; the script switches to the venv python itself):
    python domainide.py scale node
    ./domainide.py scale node --top 5 --tlds com,io
    ./domainide.py scale node --no-nouns-verbs
    ./domainide.py scale node --no-dict --dry-run

The generated word lists and the tldx command go to stderr, so stdout
carries only the tldx report and can be piped to a file.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def ensure_venv():
    """Run under the project venv python (it has fasttext/nltk)."""
    venv = SCRIPT_DIR / "venv"
    if (venv / "bin" / "python").is_file():
        if Path(sys.prefix).resolve() != venv.resolve():
            os.execv(str(venv / "bin" / "python"),
                     [str(venv / "bin" / "python"), str(Path(__file__).resolve())] + sys.argv[1:])
        return  # already running under the project venv
    try:  # no venv: a system python with the dependencies also works
        import fasttext  # noqa: F401
    except ImportError:
        sys.exit("error: no venv and fasttext is not installed — run ./setup.sh first")


ensure_venv()
sys.path.insert(0, str(SCRIPT_DIR / "lib"))  # find_words.py lives in lib/

import find_words  # noqa: E402 (needs the lib/ path set up above)

DEFAULT_TOP = 10


def build_parser():
    p = argparse.ArgumentParser(
        description="Find available two-word domain ideas: find_words expands "
                    "each seed word into related words, then tldx --cross "
                    "checks every combination of the two lists (both orders)."
    )
    p.add_argument("seeds", nargs=2, metavar=("WORD1", "WORD2"),
                   help="two seed words, e.g. scale node")
    p.add_argument("--top", type=int, default=DEFAULT_TOP,
                   help=f"related words per seed (default {DEFAULT_TOP})")
    p.add_argument("--no-nouns-verbs", action="store_true",
                   help="keep all word types, not just nouns and verbs")
    p.add_argument("--no-dict", action="store_true",
                   help="skip the dictionary check (keeps modern words like 'kubernetes'); "
                        "implies --no-nouns-verbs")
    p.add_argument("--tlds",
                   help="TLDs for tldx, comma-separated (default: tldx's own default, .com)")
    p.add_argument("--tld-preset", dest="tld_preset",
                   help="tldx TLD preset (e.g. popular, tech, startup)")
    p.add_argument("--limit", type=int, default=0,
                   help="stop after this many available domains (0 = no limit)")
    p.add_argument("-a", "--only-available", action="store_true",
                   help="show only available domains")
    p.add_argument("--dry-run", action="store_true",
                   help="print the domains tldx would check, without network calls")
    p.add_argument("--model", default=find_words.DEFAULT_MODEL,
                   help=f"fastText model path (default {find_words.DEFAULT_MODEL}, "
                        "kept at the project root)")
    return p


def normalize_seed(raw, parser):
    """Lowercase and validate one seed word; exit via parser on bad input."""
    word = raw.strip().lower()
    if not word.isalpha():
        parser.error(f"seeds must be single alphabetic words: {raw!r}")
    return word


def resolve_tldx(parser):
    """Find tldx: in the project folder, next to it, or on PATH."""
    for candidate in (SCRIPT_DIR / "tldx", SCRIPT_DIR.parent / "tldx"):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("tldx")
    if not found:
        parser.error("tldx not found: expected in the project folder, next to it, or on PATH")
    return found


def build_settings(args):
    """Translate domainide flags into find_words.related() settings.

    The fixed values mirror find_words.py's own CLI defaults.
    """
    nouns_verbs = not (args.no_nouns_verbs or args.no_dict)
    if args.no_dict and not args.no_nouns_verbs:
        print("note: --no-dict disables the nouns/verbs filter (it needs the dictionary)",
              file=sys.stderr)
    return argparse.Namespace(
        pool=200, min_score=0.45, min_len=3, max_len=8, typo_dist=None,
        top=args.top, nouns_verbs=nouns_verbs, no_dict=args.no_dict,
    )


def find_related_words(model, seed, settings):
    """Related words for one seed; [] on failure so the caller can fall back."""
    try:
        return [w for w, _ in find_words.related(model, seed, settings)]
    except Exception as exc:  # any expansion failure degrades to seed-only
        print(f"warning: find_words failed for {seed!r} ({exc}); keeping only the seed word",
              file=sys.stderr)
        return []


def build_word_list(seed, words):
    """The seed word first, then each related word once."""
    out = [seed]
    seen = {seed}
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def write_word_lists(word_lists, tmp):
    """Write each word list to its own file in tmp; returns the file paths."""
    files = []
    for i, words in enumerate(word_lists, 1):
        path = Path(tmp) / f"list{i}.txt"
        path.write_text("".join(f"{w}\n" for w in words), encoding="utf-8")
        files.append(path)
    return files


def build_tldx_command(tldx, files, args):
    """Assemble the tldx invocation from the two list files and the flags."""
    cmd = [tldx, "-i", str(files[0]), "-i", str(files[1]), "--cross"]
    if args.tlds:
        cmd += ["-t", args.tlds]
    if args.tld_preset:
        cmd += ["--tld-preset", args.tld_preset]
    if args.limit:
        cmd += ["-l", str(args.limit)]
    if args.only_available:
        cmd += ["-a"]
    if args.dry_run:
        cmd += ["--dry-run"]
    return cmd


def main():
    parser = build_parser()
    args = parser.parse_args()
    seeds = [normalize_seed(s, parser) for s in args.seeds]
    if seeds[0] == seeds[1]:
        parser.error("give two different words")

    tldx = resolve_tldx(parser)
    settings = build_settings(args)
    find_words.ensure_wordnet()

    try:
        model = find_words.load_model(args.model)  # one load, both seeds
    except FileNotFoundError as exc:
        parser.error(str(exc))

    word_lists = []
    for seed in seeds:
        words = find_related_words(model, seed, settings)
        word_lists.append(build_word_list(seed, words))
        print(f"[{seed} -> {len(word_lists[-1])} words] "
              + ", ".join(word_lists[-1]), file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="domainide-") as tmp:
        files = write_word_lists(word_lists, tmp)
        cmd = build_tldx_command(tldx, files, args)
        print(f"\nrunning: {' '.join(cmd)}\n", file=sys.stderr)
        result = subprocess.run(cmd)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
