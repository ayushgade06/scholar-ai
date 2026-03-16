from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import hashlib
import time
import asyncio
from contextlib import asynccontextmanager


# ─────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────

_executor = ThreadPoolExecutor(max_workers=4)

_summary_cache = {}
CACHE_TTL = 300


def cleanup_cache():
    now = time.time()
    expired = [k for k, (ts, _) in _summary_cache.items() if now - ts > CACHE_TTL]
    for k in expired:
        del _summary_cache[k]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    print("[*] Server shutting down")
    _executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────
# LLM CONFIG
# ─────────────────────────────────────────────────────

llm = OllamaLLM(
    model="llama3",
    num_predict=400,
    num_ctx=4096,
    temperature=0.2,
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=6000,
    chunk_overlap=400,
)


# ─────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────

class PageRequest(BaseModel):
    content: str


class QuestionRequest(BaseModel):
    content: str
    question: str


class ExplainRequest(BaseModel):
    text: str


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

def _sse(event: str, data: dict):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _cache_key(content: str):
    return hashlib.md5(content[:2000].encode()).hexdigest()


def _safe_invoke_llm(prompt: str, label="LLM"):
    for attempt in range(2):
        try:
            print(f"[*] Calling LLM [{label}]...")
            start = time.time()
            res = llm.invoke(prompt)
            print(f"[+] LLM [{label}] responded in {time.time() - start:.2f}s")
            return res
        except Exception as e:
            print(f"[!] {label} attempt {attempt+1} failed:", e)
            if attempt == 1:
                return f"Error: {str(e)}"
            time.sleep(2)


# ─────────────────────────────────────────────────────
# Health Routes
# ─────────────────────────────────────────────────────

@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "model": "llama3"}


# ─────────────────────────────────────────────────────
# Summarization
# ─────────────────────────────────────────────────────

@app.post("/summarize")
async def summarize(page: PageRequest):

    async def generate():

        try:

            cleanup_cache()

            content = page.content[:80000]

            key = _cache_key(content)

            if key in _summary_cache:

                ts, data = _summary_cache[key]

                if time.time() - ts < CACHE_TTL:

                    yield _sse("start", {"total_chunks": 1})
                    yield _sse("chunk", {"done": 1, "total": 1, "pct": 80})
                    yield _sse("done", data)
                    return

            chunks = text_splitter.split_text(content)
            total = len(chunks)

            if total == 0:
                yield _sse("error", {"message": "Empty content"})
                return

            yield _sse("start", {"total_chunks": total})

            partial = [""] * total

            prompt_template = (
                "Summarize this research section concisely.\n\n"
                "Return only the summary.\n\n"
                "Text:\n{chunk}"
            )

            futures = {
                _executor.submit(
                    _safe_invoke_llm,
                    prompt_template.format(chunk=c),
                    f"chunk {i+1}",
                ): i
                for i, c in enumerate(chunks)
            }

            done = 0

            for f in as_completed(futures):

                idx = futures[f]

                partial[idx] = await asyncio.to_thread(f.result)

                done += 1

                pct = round((done / total) * 80)

                yield _sse("chunk", {"done": done, "total": total, "pct": pct})

            yield _sse("reduce", {"message": "Composing summary"})

            combined = "\n\n".join(partial)

            final_prompt = (
                "Create structured research summary.\n\n"
                "TLDR:\n\n"
                "Key Concepts:\n\n"
                "Key Findings:\n\n"
                f"Input:\n{combined}"
            )

            result = await asyncio.to_thread(_safe_invoke_llm, final_prompt, "reduce")

            data = {"summary": result, "chunks_processed": total}

            _summary_cache[key] = (time.time(), data)

            yield _sse("done", data)

        except Exception as e:

            print("Summarization error:", e)

            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ─────────────────────────────────────────────────────
# Ask
# ─────────────────────────────────────────────────────

@app.post("/ask")
def ask(req: QuestionRequest):

    try:

        context = req.content[:15000]

        prompt = (
            "Answer the question using only the article.\n\n"
            f"{context}\n\n"
            f"Question: {req.question}\nAnswer:"
        )

        ans = _safe_invoke_llm(prompt, "ask")

        return {"answer": ans}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────
# Insights
# ─────────────────────────────────────────────────────

@app.post("/insights")
def insights(page: PageRequest):

    try:
        context = page.content[:15000]

        prompt = (
            "Identify the 3 most important scientific or research insights from this text.\n\n"
            "Format as a list of points.\n\n"
            f"Content:\n{context}"
        )

        res = _safe_invoke_llm(prompt, "insights")

        return {"insights": res}

    except Exception as e:
        print("[!] Insights error:", e)
        return {"insights": "Could not extract insights at this time."}


# ─────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────

@app.post("/graph")
def graph(page: PageRequest):

    try:

        context = page.content[:10000]

        prompt = (
            "Extract entity relationships.\n"
            "Return JSON array: "
            '[{"source":"","relation":"","target":""}]\n\n'
            f"{context}"
        )

        raw = _safe_invoke_llm(prompt)

        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

        start = cleaned.find("[")
        end = cleaned.rfind("]")

        if start != -1 and end != -1:

            edges = json.loads(cleaned[start:end + 1])

            return {"graph": edges[:40]}

        return {"graph": []}

    except Exception:

        return {"graph": []}


# ─────────────────────────────────────────────────────
# Bias
# ─────────────────────────────────────────────────────

@app.post("/bias")
def bias(page: PageRequest):

    context = page.content[:15000]

    prompt = (
        "Critique research methodology.\n\n"
        "Possible Bias:\n"
        "Weak Arguments:\n"
        "Missing Evidence:\n"
        "Logical Flaws:\n\n"
        f"{context}"
    )

    res = _safe_invoke_llm(prompt)

    return {"bias": res}


# ─────────────────────────────────────────────────────
# Reading Time
# ─────────────────────────────────────────────────────

@app.post("/reading-time")
def reading_time(page: PageRequest):

    words = len(page.content.split())

    read_time = words / 200

    sum_time = 2

    eff = (1 - (sum_time / read_time)) * 100 if read_time else 0

    return {
        "word_count": words,
        "reading_time": round(read_time, 2),
        "summary_time": sum_time,
        "efficiency_gain": round(eff, 2),
    }