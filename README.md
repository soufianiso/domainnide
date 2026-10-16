# domainide

Two-word domain name idea generator: give it two seed words, and it finds
related words for each, then checks the availability of every two-word
combination (in both orders) across TLDs.

```
"scale" + "node"
   |                         |
   | find_words              |
   | (fastText + WordNet)    |
   v                         v
scale, fortify, ...      node, vertex, topology, ...
   \                       /
    \     tldx --cross    /   (both orders: a+b and b+a)
     v                   v
✅ scaletopology.com is available
✅ topologyscale.com is available
```

## How it works

1. **Expansion** — each seed word grows into a list of related words:
   fastText semantic neighbors, cleaned by WordNet (real dictionary words
   only, no antonyms, one word per word family). The seed itself leads
   its list.
2. **Cross** — every word from list A is combined with every word from
   list B, in both orders (`scalenode` and `nodescale`).
3. **Check** — the bundled `tldx` binary checks availability per TLD
   (default `.com`).

If a seed yields no related words, the tool falls back to the seed word
alone — a run never dies on an empty list.

## Setup

Requires Python 3, plus `g++` and `make` (fasttext is built from the
bundled source — PyPI has no wheel for Python 3.14).

```bash
git clone <repo-url> domainide && cd domainide
./setup.sh
```

The first real run downloads the 7 GB fastText model (`cc.en.300.bin`)
automatically — it is not part of the repo.

## Usage

```bash
./domainide.py scale node -a                    # available .com domains only
./domainide.py scale node --top 5 --tlds com,io,ai -a
./domainide.py scale node --tld-preset popular -a
./domainide.py scale node --dry-run             # preview domains, no network
./domainide.py scale node -a --limit 20         # stop after 20 available
```

Seeds must be two different single words (letters only).

The word lists and the tldx command go to stderr; the report goes to
stdout, so it can be piped: `./domainide.py scale node -a > results.txt`

Any python works — the script switches to the project venv itself.

### Flags

| Flag | Meaning | Default |
|---|---|---|
| `--top N` | related words per seed | 10 |
| `--no-nouns-verbs` | keep all word types, not just nouns/verbs | off |
| `--no-dict` | allow non-dictionary words (e.g. "kubernetes"); implies `--no-nouns-verbs` | off |
| `--tlds a,b` | TLDs to check | tldx default (`.com`) |
| `--tld-preset P` | tldx preset (`popular`, `tech`, `startup`, ...) | — |
| `--limit N` | stop after N available domains | no limit |
| `-a` | only show available domains | off |
| `--dry-run` | list the domains without checking | off |
| `--model PATH` | fastText model path | bundled default |

Volume: `--top N` produces (N+1)² × 2 domain checks (both orders).

## Project layout

```
domainide.py        the tool
lib/find_words.py  the word engine (also usable as a standalone CLI)
tldx                the availability checker (Go binary)
fastText-0.9.2/     fasttext source, used by setup.sh to build the venv
setup.sh            creates venv/ and installs the dependencies
```

## find_words.py standalone

The engine also works on its own, e.g.:

```bash
venv/bin/python lib/find_words.py node --nouns-verbs --top 10
```
