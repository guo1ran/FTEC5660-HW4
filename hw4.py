"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================


# Lexical normalization is shared by index construction and query scoring.
def _retrieval_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = re.sub(r"\bcfo\b", "chief financial officer", text)
    text = re.sub(r"\bceo\b", "chief executive officer", text)
    aliases = {
        "headquartered": "headquarters", "headcount": "employees",
        "buyback": "repurchase", "buybacks": "repurchase",
        "borrowing": "debt", "borrowings": "debt",
        "guidance": "outlook", "guide": "outlook", "guided": "outlook",
        "forecasts": "estimate", "forecast": "estimate", "estimates": "estimate",
        "audited": "audit", "auditor": "audit", "auditors": "audit",
    }
    stop = set("a an the of to in at on for from by with and or its their this that was were is are be been did does do what which who how much more higher held hold had have has report reported give gives set generate generated earn earned spend spent end ended fiscal year fy usd million millions percent value amount company companies management fourth quarter full q q4 per than as it s city firm large announce announced".split())
    result = []
    for token in re.findall(r"[a-z]+|\d+(?:[.,]\d+)*", text):
        token = aliases.get(token, token)
        if token in stop or token.isdigit():
            continue
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        result.append(token)
    return result


def _retrieval_company_key(name: str) -> str:
    words = re.findall(r"[a-z0-9]+", name.casefold())
    while words and words[-1] in {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}:
        words.pop()
    return " ".join(words)


def _retrieve_config(index, question, *, metadata=True, rerank=True, balanced=True, abstain=True):
    if not index or not index["chunks"]:
        return []
    if not metadata:
        scores = index["bm25"].get_scores(index["tokenizer"](question))
        ids = sorted(range(len(scores)), key=lambda i: (float(scores[i]), i), reverse=True)
        return [index["chunks"][i] for i in ids[:MAX_RETRIEVED]]
    parsed = parse_query(question)
    names = parsed["companies"]
    if not names or len(names) > MAX_RETRIEVED:
        return []

    topic = question
    if rerank:
        for name in sorted(names, key=len, reverse=True):
            topic = re.sub(re.escape(name), " ", topic, flags=re.I)
        topic = re.sub(r"\bFY\s*\d{4}\b|\b\d{4}\b|\b31\s+December\b|\b12-month\b", " ", topic, flags=re.I)
        topic = re.sub(r"\b(?:earnings\s+(?:conference\s+)?call|results\s+review)\b", " ", topic, flags=re.I)
        # Broker names describe a source, rather than the requested financial fact.
        topic = re.sub(r"\bdoes\s+.+?\s+(?=forecast|set|give)", " ", topic, flags=re.I)
    query_tokens = _retrieval_tokens(topic)
    query_set = set(query_tokens)
    if not query_set:
        return []
    bm25_scores = index["bm25"].get_scores(query_tokens)
    year = parsed["fiscal_year"]
    doc_type = parsed["doc_type"]
    ranked_groups = []

    for name in names:
        key = _retrieval_company_key(name)
        candidates = []
        for i in index["companies"].get(key, []):
            chunk = index["chunks"][i]
            meta = chunk.metadata
            if meta.get("doc_type") != doc_type:
                continue
            doc_year = meta.get("fiscal_year")
            if year is not None:
                if doc_type == "annual_report" and doc_year not in (year, year + 1):
                    continue
                if doc_type != "annual_report" and doc_year != year and not re.search(rf"\b(?:FY)?{year}(?:[AE])?\b", chunk.page_content, re.I):
                    continue
            candidates.append(i)
        if not candidates:
            return []

        # Reject a requested concept absent from the candidate documents.
        available = set().union(*(index["tokens"][i] for i in candidates))
        if abstain and not query_set.issubset(available):
            return []

        ranked = []
        for i in candidates:
            chunk = index["chunks"][i]
            words = index["tokens"][i]
            coverage = len(query_set & words) / len(query_set)
            score = float(bm25_scores[i]) + (10.0 * coverage if rerank else 0.0)
            if rerank:
                normalized = " ".join(_retrieval_tokens(chunk.page_content))
                for a, b in zip(query_tokens, query_tokens[1:]):
                    if f"{a} {b}" in normalized:
                        score += 2.0
                if year is not None and chunk.metadata.get("fiscal_year") == year:
                    score += 3.0
                if year is None:
                    score += 0.01 * int(chunk.metadata.get("fiscal_year", 0))
                # Exact table values are preferable to rounded narrative figures.
                if doc_type == "annual_report" and " | " in chunk.page_content and not re.search(r"\bwho\b|headquarter|audit", question, re.I):
                    score += 3.0
            ranked.append((score, i))
        ranked.sort(reverse=True)
        ranked_groups.append(ranked)

    if not balanced:
        pooled = sorted((item for group in ranked_groups for item in group), reverse=True)
        return [index["chunks"][i] for _, i in pooled[:MAX_RETRIEVED]]

    # Round-robin selection reserves evidence for every compared company.
    selected = []
    seen = set()
    for depth in range(MAX_RETRIEVED):
        for ranked in ranked_groups:
            if depth >= len(ranked):
                continue
            i = ranked[depth][1]
            chunk = index["chunks"][i]
            identity = (chunk.metadata.get("source"), chunk.page_content)
            if identity not in seen:
                selected.append(chunk)
                seen.add(identity)
                if len(selected) == MAX_RETRIEVED:
                    return selected
    return selected


def _chunk_document_config(file_name, html, *, contextual=True):
    from bs4 import BeautifulSoup
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup.find_all(["nav", "script", "style", "head"]):
        tag.decompose()
    body = soup.body or soup
    h1 = body.find("h1")
    if h1 is None:
        raise ValueError(f"Missing company heading in {file_name}")
    company = re.sub(r"\s*\([^()]*\)\s*$", "", h1.get_text(" ", strip=True)).strip()

    heading_text = " ".join(
        tag.get_text(" ", strip=True)
        for tag in body.find_all(["h1", "h2", "h3"], limit=4)
    )
    identity = title + " " + heading_text
    if re.search(r"earnings\s+(?:conference\s+)?call", identity, re.I):
        doc_type = "earnings_call"
        year_match = re.search(r"full\s+year\s+(20\d{2})", identity, re.I)
        if year_match is None:
            year_match = re.search(r"\b(20\d{2})\b", title)
    elif re.search(r"results\s+review|equity\s+research", identity, re.I):
        doc_type = "research_report"
        year_match = re.search(r"FY\s*(20\d{2})\s+results\s+review", identity, re.I)
    else:
        doc_type = "annual_report"
        year_match = re.search(r"annual\s+report\s+(20\d{2})", title + " " + body.get_text(" ", strip=True)[:1000], re.I)
    if year_match is None:
        raise ValueError(f"Missing fiscal year in {file_name}")

    metadata = {
        "source": file_name,
        "company": company,
        "fiscal_year": int(year_match.group(1)),
        "doc_type": doc_type,
    }
    chunks = []

    def add_chunks(text, section):
        text = text.strip()
        if not text:
            return
        # Section headings and table captions are repeated as local context.
        prefix = section.strip()[:300] if contextual else ""
        prefix = prefix + "\n" if prefix else ""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_CHARS - len(prefix),
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        for part in splitter.split_text(text):
            chunks.append(Document(
                page_content=prefix + part,
                metadata={**metadata, "section": section},
            ))

    section = ""
    pending = []
    for tag in body.find_all(["h1", "h2", "h3", "p", "li", "table"]):
        # Only consume each block once; tables are handled as a whole.
        if tag.find_parent(["table", "p", "li"]) is not None:
            continue
        if tag.name in ("h1", "h2", "h3"):
            add_chunks("\n\n".join(pending), section)
            pending = []
            section = tag.get_text(" ", strip=True)
            pending.append(section)
        elif tag.name == "table":
            add_chunks("\n\n".join(pending), section)
            pending = []
            table_text = table_to_text(str(tag))
            caption = tag.find("caption")
            context = section
            if caption is not None:
                context += "\n" + caption.get_text(" ", strip=True)
            add_chunks(table_text, context)
        else:
            pending.append(tag.get_text(" ", strip=True))
    add_chunks("\n\n".join(pending), section)
    return chunks


def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    ### YOUR CODE HERE
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    if table is None:
        return ""

    lines = []
    caption = table.find("caption")
    if caption is not None:
        lines.append(caption.get_text(" ", strip=True))

    rows = table.find_all("tr")
    if not rows:
        return "\n".join(lines)

    headers = [
        cell.get_text(" ", strip=True)
        for cell in rows[0].find_all(["th", "td"], recursive=False)
    ]

    for row in rows[1:]:
        cells = [
            cell.get_text(" ", strip=True)
            for cell in row.find_all(["th", "td"], recursive=False)
        ]
        if not cells:
            continue

        parts = [cells[0]]
        for column, value in enumerate(cells[1:], start=1):
            header = headers[column] if column < len(headers) else ""
            parts.append(f"{header}: {value}" if header else value)
        lines.append(" | ".join(parts))

    return "\n".join(lines)


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    ### YOUR CODE HERE
    return _chunk_document_config(file_name, html)


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    ### YOUR CODE HERE
    question = question.strip().replace("’", "'")
    lower = question.lower()
    year_match = re.search(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", question, re.I)
    fiscal_year = int(year_match.group(1)) if year_match else None

    if re.search(r"earnings\s+(?:conference\s+)?call", lower):
        doc_type = "earnings_call"
    elif re.search(r"target\s+price|\brating\b|\bforecast\w*\b|results\s+review|\bestimat\w*\b|research\s+(?:report|note)|broker", lower):
        doc_type = "research_report"
    else:
        doc_type = "annual_report"

    companies = []

    def add_company(name):
        name = name.strip(" \t\n,?.:;\"'")
        name = re.sub(r"^(?:the\s+)?(?:value|amount)\s+of\s+", "", name, flags=re.I)
        name = re.split(r"'s\b|'(?=\s)", name, maxsplit=1)[0]
        if name and name.casefold() not in {n.casefold() for n in companies}:
            companies.append(name)

    # A comparison explicitly introduces both subjects after the colon.
    comparison = re.search(r":\s*(.+?)\s*\?*$", question)
    if comparison and re.search(r"\bwhich\s+company\b", lower):
        for name in re.split(r"\s+(?:or|versus|vs\.?|and)\s+", comparison.group(1), flags=re.I):
            add_company(name)

    # Grammatical subject/object patterns also retain companies absent from
    # the corpus; matching only known names would lose those questions.
    patterns = [
        r"\bdid\s+(.+?)\s+(?:hold|report|earn|generate|spend|have|pay|record|announce)\b",
        r"\b(?:give|gives|assign|assigns)\s+(.+?)\s+(?:in\s+its|a\s+rating|an\s+investment)\b",
        r"\b(?:is|was)\s+(.+?)\s+headquartered\b",
        r"\b(?:forecast\w*|estimat\w*)\s+for\s+(.+?)(?=,|\?|$)",
        r"\b(?:set|sets)\s+for\s+(.+?)(?:'s|')\s",
        r"\b(?:what|who)\s+(?:was|were|is|are)\s+(.+?)(?:'s\b|'(?=\s))",
        r"\baudited\s+(.+?)(?:'s\b|'(?=\s))",
    ]
    if not companies:
        for pattern in patterns:
            match = re.search(pattern, question, re.I)
            if match:
                add_company(match.group(1))
                break

    # Possessive company names cover subjects embedded in longer questions.
    if not companies:
        name_pattern = r"[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*)*"
        for match in re.finditer(r"\b(" + name_pattern + r")(?:'s\b|'(?=\s))", question):
            add_company(match.group(1))

    return {
        "companies": companies,
        "fiscal_year": fiscal_year,
        "doc_type": doc_type,
    }


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    ### YOUR CODE HERE
    from rank_bm25 import BM25Okapi

    chunks = list(chunks)
    tokens = [_retrieval_tokens(chunk.page_content) for chunk in chunks]
    groups = {}
    for i, chunk in enumerate(chunks):
        key = _retrieval_company_key(str(chunk.metadata.get("company", "")))
        groups.setdefault(key, []).append(i)
    return {
        "chunks": chunks,
        "tokenizer": _retrieval_tokens,
        "tokens": [set(row) for row in tokens],
        "bm25": BM25Okapi([row or ["__empty__"] for row in tokens]) if tokens else None,
        "companies": groups,
    }


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    ### YOUR CODE HERE
    return _retrieve_config(index, question)


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
