"""Milestone 1: gather a corpus of real sports news and prepare it.

The articles come from Wikinews (https://en.wikinews.org), whose text is licensed CC BY 2.5. The source of
every article is listed in corpus/SOURCES.tsv. Fetching needs the network and is only run to rebuild the
corpus; the cleaned corpus is committed so everything else works offline.
`python -m app.corpus` rebuilds corpus/sports_news.txt.
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikinews.org/w/api.php"
ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "corpus" / "sports_news.txt"
SOURCES_PATH = ROOT / "corpus" / "SOURCES.tsv"
CACHE_PATH = ROOT / "data" / "wikinews_cache.json"  # raw wikitext, git-ignored
EOT = "<|endoftext|>"  # separates articles, so the model learns where one ends and the next begins
CATEGORY = "Category:Sports"
# plain-text headings after which an article only has links and navigation
TAIL_HEADINGS = {"sources", "related news", "related articles", "external links", "references",
                 "see also", "related", "notes", "further reading", "sister links", "related links"}
# inline templates whose displayed text is part of the sentence: {{w|Gary Neville}} is just "Gary Neville"
KEEP_TEMPLATES = {"w", "wikipedia", "wp", "nowrap", "abbr", "sic", "lang"}


def _call(retries=8, **params):
    "One API request. Waits and retries when the server asks us to slow down (HTTP 429/503)."
    params["format"] = "json"
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": "sports-play-by-play-generator/1.0"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as err:
            if err.code not in (429, 503) or attempt == retries:
                raise
            time.sleep(max(float(err.headers.get("Retry-After") or 0), 10 * attempt))


def fetch_titles(category=CATEGORY, limit=3000, pause=0.5):
    "Titles of the articles in a Wikinews category (main namespace only)."
    titles, cont = [], {}
    while len(titles) < limit:
        data = _call(action="query", list="categorymembers", cmtitle=category, cmlimit=500,
                     cmnamespace=0, cmtype="page", **cont)
        titles += [m["title"] for m in data["query"]["categorymembers"]]
        if "continue" not in data:
            break
        cont = data["continue"]
        time.sleep(pause)
    return titles[:limit]


def fetch_texts(titles, batch=50, pause=3.0, cache=CACHE_PATH):
    """{title: raw wikitext}, 50 articles per request. Already-fetched articles are kept in a cache file, so an
    interrupted run (or a rate limit) can simply be started again."""
    out = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [t for t in titles if t not in out]
    for i in range(0, len(todo), batch):
        data = _call(action="query", prop="revisions", rvprop="content", rvslots="main",
                     titles="|".join(todo[i:i + batch]))
        for page in data["query"]["pages"].values():
            slot = (page.get("revisions") or [{}])[0].get("slots", {}).get("main", {})
            out[page["title"]] = slot.get("*", slot.get("content", ""))
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(out), encoding="utf-8")
        print(f"fetched {min(i + batch, len(todo))}/{len(todo)} articles", flush=True)
        time.sleep(pause)
    return {t: out[t] for t in titles if t in out}


def _template_text(match):
    "The text a template shows in the sentence: its first (or, for {{w|page|shown}}, last) plain argument."
    name, *args = (p.strip() for p in match.group(1).split("|"))
    args = [a for a in args if a and "=" not in a]
    if name.lower() not in KEEP_TEMPLATES or not args:
        return ""  # pull quotes, source lists, notices and so on
    return args[-1] if name.lower() in ("w", "wikipedia", "wp") else args[0]


def wikitext_to_text(wiki):
    "Turn a Wikinews article's wikitext into plain paragraphs (templates, links, references and markup removed)."
    t = re.sub(r"<!--.*?-->", "", wiki, flags=re.S)
    t = re.sub(r"<ref[^>]*?/>|<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<(gallery|noinclude|includeonly|nowiki|math|poem)[^>]*>.*?</\1>", "", t, flags=re.S)
    while True:  # templates can nest, so resolve the innermost ones until none are left
        t, n = re.subn(r"\{\{([^{}]*)\}\}", _template_text, t)
        if not n:
            break
    t = re.sub(r"\{\|.*?\|\}", "", t, flags=re.S)                                   # tables
    t = re.sub(r"\[\[(?:Category|File|Image|Media):[^\]]*\]\]", "", t, flags=re.I)  # categories and pictures
    t = re.sub(r"\[\[(?:w|wikipedia|wikt|n|s|c|commons):([^\]|]*)\]\]", r"\1", t, flags=re.I)  # [[w:Page]]
    t = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", t)                             # [[page|shown text]]
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)                                       # [[page]]
    t = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", t)                              # [url shown text]
    t = re.sub(r"\[https?://[^\]\s]*\]|https?://\S+", "", t)
    t = re.sub(r"<[^>]+>", "", t)                                                   # remaining html tags
    t = re.sub(r"'{2,}", "", t)                                                     # bold and italics marks
    t = re.sub(r"^\s*=+\s*(.*?)\s*=+\s*$", r"\1", t, flags=re.M)                    # == Heading == -> Heading
    return re.sub(r"^[*#:;]+\s*", "", t, flags=re.M)


def to_ascii(text):
    "Straighten quotes and dashes and keep plain ASCII only, which keeps the byte-pair vocabulary small."
    for old, new in ((" ", " "), ("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                     ("–", "-"), ("—", " - ")):
        text = text.replace(old, new)
    return "".join(c for c in text if c == "\n" or 32 <= ord(c) < 127)


def clean_article(text):
    """Keep the news body: drop link sections and navigation, markup and stray symbols, and normalise spaces."""
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower() in TAIL_HEADINGS:
            break  # everything after this heading is references and links
        if not stripped or re.search(r"https?://|\{\{|\}\}|\[\[|\]\]", stripped):
            continue
        if len(stripped) < 40 and not stripped.endswith((".", "?", "!", '"')):
            continue  # a heading or a caption, not news text
        kept.append(re.sub(r"\s+", " ", stripped))
    return to_ascii("\n".join(kept))


def build_corpus(articles, min_chars=600, max_chars=6000):
    "Convert and clean raw wikitext articles, drop short and duplicate ones, and join them with the end-of-text marker."
    docs, kept, seen = [], [], set()
    for title in sorted(articles):
        body = clean_article(wikitext_to_text(articles[title]))[:max_chars]
        key = body[:200]
        if len(body) < min_chars or key in seen:
            continue
        seen.add(key)
        kept.append(title)
        docs.append(f"{to_ascii(title).strip()}\n{body}")
    return f"\n{EOT}\n".join(docs) + f"\n{EOT}\n", kept


def article_url(title):
    return "https://en.wikinews.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))


def load_corpus(path=CORPUS_PATH):
    return path.read_text(encoding="utf-8")


def main():
    titles = fetch_titles()
    print(f"found {len(titles)} articles in {CATEGORY}")
    corpus, kept = build_corpus(fetch_texts(titles))
    CORPUS_PATH.parent.mkdir(exist_ok=True)
    CORPUS_PATH.write_text(corpus, encoding="utf-8")
    SOURCES_PATH.write_text("title\turl\n" + "\n".join(f"{t}\t{article_url(t)}" for t in kept) + "\n",
                            encoding="utf-8")
    print(f"kept {len(kept)} articles, {len(corpus):,} characters -> {CORPUS_PATH}")


if __name__ == "__main__":
    main()
