"""All LLM prompts for the RAG Zero pipeline."""

from __future__ import annotations

ROUTER_PROMPT = """You are a query router. Read the user's question and classify it.

Respond with exactly one of:
- RETRIEVAL: factual question that can be answered from a corpus
- DIRECT: greeting, meta-question, or simple request that needs no retrieval

{memory}Question: {question}
Label:"""


DECOMPOSER_PROMPT = """You are a question rewriter. Given a possibly complex user question, rewrite it as a clear, self-contained search query that preserves all named entities and constraints.

{memory}Original question: {question}
Search query:"""


FINAL_QUERY_PROMPT = """Refine the search query based on previous evidence and the remaining gaps. Output only the refined search query, no explanation.

Original question: {question}
Previous query: {query}
Previous evidence passages:
{evidence}
Refined search query:"""


GRADER_PROMPT = """You are a retrieval grader. Given a question and the retrieved evidence, rate how likely the evidence contains the answer.

Respond with a single float from 0.0 to 1.0 where:
- 1.0 means the evidence clearly answers the question
- 0.0 means the evidence is completely irrelevant or wrong

Question: {question}
Evidence:
{evidence}

Score (0.0-1.0):"""


GENERATOR_PROMPT = """You are a verifiable research assistant. Answer the user's question using ONLY the provided evidence.

Rules:
1. Every factual sentence must end with a citation in the format [chunk_id].
2. If the evidence is insufficient, respond exactly with: INSUFFICIENT_EVIDENCE
3. Do not use outside knowledge.
4. Be concise.

Question: {question}
Evidence:
{evidence}

Answer:"""


CONTEXUALIZER_PROMPT = """Read the passage below, then write a single sentence that describes the general context this chunk belongs to (e.g., what document or topic it is from). Do not answer any question; only provide context.

Passage:
{text}

One-line context prefix:"""


JUDGE_PROMPT = """You are a strict fact-checker. Given a claim and a set of evidence passages, decide if the claim is supported.

Evidence:
{evidence}

Claim: {claim}

Respond with a single float from 0.0 to 1.0 representing your confidence that the claim is fully supported by the evidence. 1.0 = fully supported, 0.0 = unsupported/contradicted."""


COVE_PROMPT = """You are a claim verifier. Given a draft answer, break it into atomic claims and verify each claim against the evidence.

For each claim, output a line in the format:
claim: <claim> | supported: <yes/no> | confidence: <0.0-1.0>

Draft answer: {answer}
Evidence:
{evidence}

Claims:"""


ABSTAIN_PROMPT = """You are a conservative assistant. Decide whether to answer the question or abstain.

If the evidence is insufficient or the question is based on a false premise, respond exactly with: ABSTAIN
Otherwise, answer the question briefly using only the evidence and include citations.

Question: {question}
Evidence:
{evidence}

Response:"""


RERANKER_PROMPT = """Does the document answer the query? Respond with Yes or No only.

Query: {query}
Document: {document}

Answer:"""
