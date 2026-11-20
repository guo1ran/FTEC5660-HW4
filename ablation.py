"""Task 5: cumulative, single-component ablations of hw4.py.

Run from the repository root:
    .venv/bin/python ablation.py
Each variant builds an in-memory index from its input corpus. No index is saved.
Gold labels are read only for evaluation, after retrieval; they never affect it.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import hw4

VARIANTS = {'B0': 'Naive get_text + fixed chunks + whitespace BM25 global top-5', 'V1': '+ table serialization only', 'V2': '+ navigation/head/style/script removal only', 'V3': '+ section/table boundaries for chunking only', 'V4': '+ repeated local section/caption context only', 'V5': '+ lexical token normalization only', 'V6': '+ query parsing and metadata candidate filtering only', 'V7': '+ topic-focused lexical reranking only', 'V8': '+ per-company round-robin allocation only', 'V9': '+ missing-concept abstention only (production configuration)'}


def metadata(file_name, html):
    """Read metadata from the document itself, without calling table_to_text."""
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.title.get_text(' ', strip=True) if soup.title else ''
    body = soup.body or soup
    h1 = body.find('h1')
    company = re.sub(r'\s*\([^()]*\)\s*$', '', h1.get_text(' ', strip=True)).strip()
    headings = ' '.join(t.get_text(' ', strip=True) for t in body.find_all(['h1', 'h2', 'h3'], limit=4))
    identity = title + ' ' + headings
    if re.search(r'earnings\s+(?:conference\s+)?call', identity, re.I):
        kind = 'earnings_call'
        match = re.search(r'full\s+year\s+(20\d{2})', identity, re.I) or re.search(r'\b(20\d{2})\b', title)
    elif re.search(r'results\s+review|equity\s+research', identity, re.I):
        kind = 'research_report'
        match = re.search(r'FY\s*(20\d{2})\s+results\s+review', identity, re.I)
    else:
        kind = 'annual_report'
        match = re.search(r'annual\s+report\s+(20\d{2})', title + ' ' + body.get_text(' ', strip=True)[:1000], re.I)
    return {'source': file_name, 'company': company, 'fiscal_year': int(match.group(1)), 'doc_type': kind}


def fixed_chunks(file_name, html, serialize_tables=False, clean=False):
    meta = metadata(file_name, html)
    soup = BeautifulSoup(html, 'html.parser')
    if serialize_tables:
        for table in soup.find_all('table'):
            table.replace_with(NavigableString('\n' + hw4.table_to_text(str(table)) + '\n'))
    if clean:
        for tag in soup.find_all(['nav', 'script', 'style', 'head']):
            tag.decompose()
        soup = soup.body or soup
    # B0 is deliberately naive: this discards HTML table row/column structure.
    text = soup.get_text(' ', strip=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    return [Document(page_content=part, metadata=dict(meta)) for part in splitter.split_text(text)]


def build_index(chunks, normalized=False):
    if normalized:
        return hw4.build_index(chunks)
    tokenizer = lambda text: text.lower().split()
    tokens = [tokenizer(c.page_content) for c in chunks]
    groups = {}
    for i, c in enumerate(chunks):
        groups.setdefault(hw4._retrieval_company_key(c.metadata['company']), []).append(i)
    return {'chunks': chunks, 'tokens': [set(row) for row in tokens],
            'tokenizer': tokenizer, 'bm25': BM25Okapi(tokens), 'companies': groups}


def delta(before, after):
    return {k: after[k] - before[k] for k in before if k != 'cost_usd'}


def evaluate(variant, corpus, questions, gold, docs_meta, tables, output, generate_answers=True):
    level = list(VARIANTS).index(variant)
    started = time.perf_counter()
    before = hw4.GENERATOR_USAGE.summary()
    chunks_by_file = {
        f: (hw4._chunk_document_config(f, html, contextual=level >= 4) if level >= 3
            else fixed_chunks(f, html, serialize_tables=level >= 1, clean=level >= 2))
        for f, html in corpus.items()
    }
    chunks = [c for cs in chunks_by_file.values() for c in cs]
    index = build_index(chunks, normalized=level >= 5)
    parsed = {qid: hw4.parse_query(q) if level >= 6 else {} for qid, q in questions.items()}
    contexts = {qid: hw4._retrieve_config(index, q, metadata=level >= 6,
                rerank=level >= 7, balanced=level >= 8, abstain=level >= 9)
                for qid, q in questions.items()}
    retrieval_seconds = time.perf_counter() - started
    all_text = {c.page_content for c in chunks}
    assert all(len(cs) <= 5 and all(c.page_content in all_text for c in cs) for cs in contexts.values())
    errors = {}

    def answer(qid):
        if not generate_answers:
            return ''
        try:
            return hw4.generate(questions[qid], contexts[qid])
        except Exception as exc:
            errors[qid] = str(exc)
            return ''

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    rows = []
    for qid, q in questions.items():
        g = gold[qid]
        cs = contexts[qid]
        correct = hw4.answer_correct(answers[qid], g) if generate_answers else ''
        rows.append({'id': qid, 'category': g['category'], 'question': q,
            'parsed': json.dumps(parsed[qid]), 'parse_score': hw4.parse_score(parsed[qid], g),
            'retrieved': ';'.join(sorted({c.metadata['source'] for c in cs})),
            'valid': len(cs) <= 5 and all(c.page_content in all_text for c in cs),
            'retrieval_hit': hw4.retrieval_hit(cs, g) if g['evidence'] else '',
            'answer': answers[qid], 'correct': correct, 'expected': '' if correct or not generate_answers else g['answer']})
    folder = output / variant
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    answerable = [r for r in rows if gold[r['id']]['evidence']]
    unanswerable = [r for r in rows if not gold[r['id']]['evidence']]
    valid = sum(bool(contexts[r['id']]) and r['valid'] for r in answerable) / len(answerable)
    recall = sum(r['retrieval_hit'] for r in answerable) / len(answerable)
    kept = sum(bool(contexts[r['id']]) for r in answerable) / len(answerable)
    rejected = sum(not contexts[r['id']] for r in unanswerable) / len(unanswerable)
    accuracy = sum(r['correct'] for r in rows) / len(rows) if generate_answers else None
    cells = sum(hw4.table_score(hw4.table_to_text(t['html']), t) if level >= 1 else
                hw4.table_score(BeautifulSoup(t['html'], 'html.parser').get_text(' ', strip=True), t)
                for t in tables) / len(tables)
    doc_problems = {f: problems for f, meta in docs_meta.items()
                    if (problems := hw4.document_problems(chunks_by_file.get(f), None, meta, f))}
    doc_pass = 1 - len(doc_problems) / len(docs_meta)
    parse_score = sum(r['parse_score'] for r in rows) / len(rows)
    points = {'1.1': 8*cells, '1.2': 7*doc_pass, '2.1': 5*valid, '2.2': 10*parse_score,
              '2.3': 15*max(0, min(1, (recall-.5)/.4)), '3': 10*max(0, kept+rejected-1),
              '4': 20*accuracy if accuracy is not None else None}
    usage = delta(before, hw4.GENERATOR_USAGE.summary())
    usage['cost_usd'] = (usage['llm_input_tokens'] * hw4.PRICE_LLM_INPUT + usage['llm_output_tokens'] * hw4.PRICE_LLM_OUTPUT)/1e6
    metrics = {'variant': variant, 'change': VARIANTS[variant], 'chunks': len(chunks),
        'table_score': cells, 'document_pass': doc_pass, 'parse_score': parse_score,
        'valid': valid, 'recall': recall, 'kept': kept, 'rejected': rejected, 'accuracy': accuracy,
        'points': points, 'total': sum(points.values()) if accuracy is not None else None,
        'retrieval_seconds': retrieval_seconds, 'seconds': time.perf_counter()-started,
        'generator_usage': usage, 'api_errors': errors,
        'missed_retrieval_ids': [r['id'] for r in answerable if not r['retrieval_hit']],
        'wrong_answer_ids': [r['id'] for r in rows if generate_answers and not r['correct']],
        'document_problems': doc_problems}
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f"{variant}: recall={recall:.1%} kept={kept:.1%} rejected={rejected:.1%} accuracy={accuracy} total={metrics['total']} API errors={len(errors)}", flush=True)
    if errors:
        raise RuntimeError(f'{variant}: API errors occurred; see metrics.json before reporting these scores')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-folder', default='public_test')
    parser.add_argument('--out-dir', default='experiments')
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--retrieval-only', action='store_true', help='Skip model calls; answer accuracy is not measured')
    args = parser.parse_args()
    folder, output = Path(args.data_folder), Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    corpus = hw4.load_corpus(folder/'corpus')
    questions = hw4.load_json(folder/'questions.json')
    gold = hw4.load_json(folder/'ground_truth.json')
    docs_meta = hw4.load_json(folder/'docs.json')
    tables = hw4.load_json(folder/'tables.json')
    for v in args.variants:
        evaluate(v, corpus, questions, gold, docs_meta, tables, output, not args.retrieval_only)
    metrics = [json.loads(p.read_text()) for v in VARIANTS if (p := output/v/'metrics.json').exists()]
    with (output/'summary.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['variant','change','chunks','table_score','document_pass','parse_score','valid','recall','kept','rejected','accuracy','total','seconds']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(metrics)

if __name__ == '__main__':
    main()
