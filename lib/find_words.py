#!/usr/bin/env python3
"""Related-word finder for domain brainstorming (fastText + WordNet).

fastText supplies candidates ranked by similarity; WordNet cleans them
(real words only, no antonyms, one word per word-family, optional POS filter).

Output:
    screen      a tidy table per keyword (rank, word, score) plus a summary
    --out FILE  a clean list: one word per line, no headers or scores,
                each word once across all keywords
    --words-only  print that same clean list to the screen instead of tables

Usage:
    python find_words.py fast
    python find_words.py fast scale orbit --top 15 --max-len 7
    python find_words.py --file data.txt --nouns-verbs --out words.txt
    python find_words.py --file data.txt --words-only --sort

data.txt: one keyword per line. Blank lines and lines starting with # are ignored.
"""
import argparse
import contextlib
import io
import os
import shutil
import sys

import fasttext
import fasttext.util
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import SnowballStemmer

DEFAULT_MODEL = "cc.en.300.bin"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # lib/ -> project root
stem = SnowballStemmer("english").stem


def ensure_wordnet():
    """Download the WordNet data on first run if it is missing."""
    try:
        wn.synsets("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)


def resolve_model_path(path):
    """The default model lives at the project root; custom paths stay as given."""
    if path == DEFAULT_MODEL:
        return os.path.join(PROJECT_ROOT, DEFAULT_MODEL)
    return path


def load_model(path):
    """Load the fastText model, downloading the default one if it is missing."""
    path = resolve_model_path(path)
    if not os.path.isfile(path):
        if path != resolve_model_path(DEFAULT_MODEL):
            raise FileNotFoundError(f"model not found: {path}")
        downloaded = fasttext.util.download_model("en", if_exists="ignore")  # ~7 GB, run once
        if os.path.abspath(downloaded) != os.path.abspath(path):
            shutil.move(downloaded, path)  # keep the model at the project root
    with contextlib.redirect_stderr(io.StringIO()):  # hides the harmless load_model warning
        return fasttext.load_model(path)


def family(word):
    """Collapse inflections and derived forms: quick/quicker/quickly -> 'quick'."""
    return stem(wn.morphy(word) or word)


def antonym_names(seed):
    return {
        a.name().lower()
        for s in wn.synsets(seed)
        for lemma in s.lemmas()
        for a in lemma.antonyms()
    }


def osa_distance(a, b):
    """Edit distance where swapping two neighbouring letters counts as one edit."""
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[len(b)]


def load_seeds(args, parser):
    """Combine command-line seeds and the --file keywords, deduplicated, in order."""
    raw = list(args.seeds)
    if args.file:
        try:
            with open(args.file, encoding="utf-8") as fh:
                raw += [line.strip() for line in fh]
        except OSError as e:
            parser.error(f"cannot read {args.file}: {e}")
    seeds, seen = [], set()
    for item in raw:
        word = item.strip().lower()
        if not word or word.startswith("#"):
            continue
        if " " in word:
            print(f"skipping multi-word entry: {word!r}", file=sys.stderr)
            continue
        if word not in seen:
            seen.add(word)
            seeds.append(word)
    if not seeds:
        parser.error("no seed words: pass some on the command line or use --file")
    return seeds


def related(model, seed, args):
    # Word families we never want to return: the seed's own and its antonyms'.
    blocked = {family(seed)} | {family(a) for a in antonym_names(seed)}
    seen = {seed}
    out = []
    # Spelling-variant limit: 1 edit for short seeds, 2 for longer ones (0 disables).
    max_typo = args.typo_dist if args.typo_dist is not None else (1 if len(seed) < 6 else 2)
    for score, w in model.get_nearest_neighbors(seed, k=args.pool):
        if score < args.min_score:
            break  # neighbors come sorted by score, so the rest are weaker
        if not (w.isalpha() and w.islower()):  # drops names, acronyms, junk
            continue
        if not args.min_len <= len(w) <= args.max_len:
            continue
        if w in seen or seed in w or w in seed:
            continue
        if osa_distance(seed, w) <= max_typo:  # typo of the seed: clould, sytem, desgin
            continue
        if len(w) >= 5 and any(len(o) >= 5 and osa_distance(o, w) <= 1 for o, _ in out):
            continue  # near-duplicate of a word already kept: cumulus / cumulous
        fam = family(w)
        if fam in blocked:  # inflection, same family, or antonym
            continue
        if not args.no_dict and not wn.synsets(w):  # real dictionary words only
            continue
        if args.nouns_verbs and not (
            wn.synsets(w, pos=wn.NOUN) or wn.synsets(w, pos=wn.VERB)
        ):
            continue
        seen.add(w)
        blocked.add(fam)
        out.append((w, score))
        if len(out) == args.top:
            break
    return out


def format_block(seed, results):
    """One tidy table for a keyword: rank, word, score."""
    count = f"{len(results)} word{'s' if len(results) != 1 else ''}"
    head = f"{seed}  ({count})"
    lines = [head, "-" * len(head)]
    if not results:
        lines.append("  (nothing passed the filters)")
    else:
        width = max(len(w) for w, _ in results)
        for i, (w, score) in enumerate(results, 1):
            lines.append(f"  {i:>2}. {w:<{width}}  {score:.3f}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Find related words with fastText + WordNet.")
    p.add_argument("seeds", nargs="*", help="seed words (optional if --file is used)")
    p.add_argument("-f", "--file", help="text file with one keyword per line")
    p.add_argument("-o", "--out", help="save a clean word list (one word per line) to this file")
    p.add_argument("--sort", action="store_true", help="alphabetize the saved/printed word list")
    p.add_argument("--top", type=int, default=10, help="results per seed (default 10)")
    p.add_argument("--min-score", type=float, default=0.45, help="minimum similarity (default 0.45)")
    p.add_argument("--min-len", type=int, default=3, help="minimum word length (default 3)")
    p.add_argument("--max-len", type=int, default=8, help="maximum word length (default 8)")
    p.add_argument("--typo-dist", type=int, default=None,
                   help="drop neighbors within this many letter edits of the seed "
                        "(default: 1 for seeds under 6 letters, else 2; 0 turns it off)")
    p.add_argument("--pool", type=int, default=200, help="neighbors to fetch before filtering (default 200)")
    p.add_argument("--nouns-verbs", action="store_true",
                   help="keep only words that can be a noun or verb (drops adverbs, most adjectives)")
    p.add_argument("--no-dict", action="store_true",
                   help="skip the WordNet dictionary check (keeps modern words like 'kubernetes')")
    p.add_argument("--words-only", action="store_true",
                   help="print the clean word list (one per line) instead of tables")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"path to the model (default {DEFAULT_MODEL})")
    args = p.parse_args()

    if args.no_dict and args.nouns_verbs:
        p.error("--nouns-verbs needs the WordNet dictionary, so it cannot be combined with --no-dict")

    seeds = load_seeds(args, p)  # fail early, before the slow model load
    ensure_wordnet()

    try:
        model = load_model(args.model)  # loaded once for all seeds
    except FileNotFoundError as exc:
        p.error(str(exc))

    collected, seen_words, empty = [], set(), []
    for seed in seeds:
        results = related(model, seed, args)
        if not results:
            empty.append(seed)
        for w, _ in results:  # each word kept once across all keywords
            if w not in seen_words:
                seen_words.add(w)
                collected.append(w)
        if not args.words_only:
            print(format_block(seed, results))
            print()

    words = sorted(collected) if args.sort else collected

    if args.words_only:
        for w in words:
            print(w)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("".join(f"{w}\n" for w in words))

    summary = [f"{len(seeds)} keyword{'s' if len(seeds) != 1 else ''}, {len(words)} unique words"]
    if empty:
        summary.append(f"no results for: {', '.join(empty)}")
    if args.out:
        summary.append(f"saved to {args.out}")
    # Keep stdout clean when it carries the word list, so it can be piped.
    print("Done: " + "; ".join(summary), file=sys.stderr if args.words_only else sys.stdout)


if __name__ == "__main__":
    main()

