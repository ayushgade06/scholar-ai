from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
import json
import re

app = FastAPI()

llm = Ollama(model="llama3")


class PageRequest(BaseModel):
    content: str


class QuestionRequest(BaseModel):
    content: str
    question: str


class ExplainRequest(BaseModel):
    text: str


def split_text(text, chunk_size=4000):
    chunks = []
    start = 0

    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size

    return chunks


@app.post("/summarize")
def summarize(page: PageRequest):

    chunks = split_text(page.content)

    partial_summaries = []

    for chunk in chunks:

        prompt = f"""
You are a research assistant.

Summarize the following part of an article.

Return a concise summary.

Text:
{chunk}
"""

        result = llm.invoke(prompt)

        partial_summaries.append(result)

    combined = "\n".join(partial_summaries)

    final_prompt = f"""
You are a research assistant.

Combine the following summaries into a structured summary.

Return in this format:

TLDR:
(2-3 sentence summary)

Key Concepts:
- concept 1
- concept 2
- concept 3

Key Findings:
- finding 1
- finding 2

Summaries:
{combined}
"""

    final_result = llm.invoke(final_prompt)

    return {"summary": final_result}


@app.post("/ask")
def ask_question(req: QuestionRequest):

    text = req.content[:8000]

    prompt = f"""
You are an expert research assistant.

Answer the question using ONLY the article.

If the answer is not in the article say:
"The article does not mention this."

Article:
{text}

Question:
{req.question}

Answer:
"""

    result = llm.invoke(prompt)

    return {"answer": result}


@app.post("/insights")
def extract_insights(page: PageRequest):

    text = page.content[:8000]

    prompt = f"""
You are an expert research analyst.

Analyze the following article and extract structured research insights.

Return in the following format:

Problem:
(What problem does the article address?)

Methodology:
(How does the article attempt to solve it?)

Findings:
(What are the main results?)

Evidence:
(What experiments, data, or arguments support the findings?)

Limitations:
(What are weaknesses or constraints?)

Article:
{text}
"""

    result = llm.invoke(prompt)

    return {
        "insights": result
    }


@app.post("/graph")
def generate_graph(page: PageRequest):

    text = page.content[:8000]

    prompt = f"""
You extract relationships from articles.

Return ONLY valid JSON.

Format:

[
  {{"source":"Entity1","relation":"relationship","target":"Entity2"}},
  {{"source":"EntityA","relation":"relationship","target":"EntityB"}}
]

Do NOT include explanations.
Do NOT include markdown.
Do NOT include ```json.

Article:
{text}
"""

    result = llm.invoke(prompt)

    try:
        cleaned = re.sub(r"```json|```", "", result).strip()
        graph = json.loads(cleaned)
    except:
        graph = []

    return {
        "graph": graph
    }


@app.post("/bias")
def detect_bias(page: PageRequest):

    text = page.content[:8000]

    prompt = f"""
You are an expert research reviewer.

Analyze the following article and identify critical evaluation points.

Return in the following format:

Possible Bias:
(Does the article show bias or favor a particular viewpoint?)

Weak Arguments:
(Identify arguments that lack strong support)

Missing Evidence:
(Claims made without sufficient evidence)

Limitations:
(Constraints or weaknesses in the study or article)

Article:
{text}
"""

    result = llm.invoke(prompt)

    return {
        "bias": result
    }


@app.post("/explain")
def explain_text(req: ExplainRequest):

    text = req.text[:1000]

    prompt = f"""
You are an expert teacher.

Explain the following concept in a very simple way.

Return in this format:

Concept:
(rewrite the concept briefly)

Explanation:
(simple explanation)

Analogy:
(give an everyday analogy)

Text:
{text}
"""

    result = llm.invoke(prompt)

    return {
        "explanation": result
    }


@app.post("/reading-time")
def reading_time(page: PageRequest):

    text = page.content

    words = len(text.split())

    reading_time = words / 200

    summary_time = 2

    efficiency = (1 - summary_time / reading_time) * 100 if reading_time > 0 else 0

    return {
        "word_count": words,
        "reading_time": round(reading_time, 2),
        "summary_time": summary_time,
        "efficiency_gain": round(efficiency, 2)
    }