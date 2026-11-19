"""Milestone 1: the sports news corpus and how it is cleaned."""
import re

from app.corpus import EOT, SOURCES_PATH, build_corpus, clean_article, load_corpus, wikitext_to_text

WIKI = """{{date|November 10, 2024}}{{Publish}}
'''London''' - The [[England national football team|English team]] won the [[Euro]] final.<ref>cite</ref>
[[Category:Sports]]

== Sources ==
* [http://example.com BBC report]
{{Archive}}"""


def test_wikitext_is_turned_into_plain_text():
    out = wikitext_to_text(WIKI)
    assert "London - The English team won the Euro final." in out
    assert not re.search(r"\{\{|\[\[|<ref|'''|http", out)
    assert "Sources" in out and "BBC report" in out  # clean_article later cuts everything after this heading

RAW = """Spain beat the Netherlands 1-0 in the final.

Match report

The only goal came in extra time, ten minutes from the end of the match, and the crowd erupted. See https://example.org

Sources
Some source line that must not survive.
Related news
Another link."""


def test_clean_article_keeps_the_news_and_drops_links_and_headings():
    out = clean_article(RAW)
    assert "Spain beat the Netherlands 1-0 in the final." in out
    assert "The only goal came in extra time" not in out or "https" not in out  # link line removed
    assert "Match report" not in out          # a heading, not news text
    assert "Sources" not in out and "source line" not in out and "Related news" not in out


def test_clean_article_normalises_to_plain_ascii():
    out = clean_article("The coach said “we played well” – and the team’s fans agreed with him today.")
    assert out == "The coach said \"we played well\" - and the team's fans agreed with him today."
    assert all(ord(c) < 128 for c in clean_article("Café owners cheered the home team all through the long evening."))


def test_build_corpus_filters_short_and_duplicate_articles():
    long_body = "The home side scored twice in the second half and held on for the win. " * 12
    articles = {"Good match": long_body, "Same match again": long_body, "Tiny": "Too short to be news."}
    text, kept = build_corpus(articles)
    assert kept == ["Good match"]  # the duplicate and the short article are dropped
    assert text.startswith("Good match\n") and text.endswith(EOT + "\n")


def test_committed_corpus_is_clean_and_large_enough():
    text = load_corpus()
    articles = [a for a in text.split(EOT) if a.strip()]
    assert len(articles) >= 500 and len(text) > 1_000_000
    assert all(c == "\n" or 32 <= ord(c) < 127 for c in text)
    assert not re.search(r"https?://|\{\{|\[\[", text)


def test_every_article_has_a_listed_source():
    rows = SOURCES_PATH.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "title\turl" and len(rows) - 1 >= 500
    assert all(r.split("\t")[1].startswith("https://en.wikinews.org/wiki/") for r in rows[1:])
