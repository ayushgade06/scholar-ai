from fastapi import FastAPI
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

# ─── Global State ─────────────────────────────────────────────────────────────

_executor = ThreadPoolExecutor(max_workers=4)
_summary_cache: dict = {}
CACHE_TTL = 300


def cleanup_cache():
    now = time.time()
    expired = [k for k, (ts, _) in _summary_cache.items() if now - ts > CACHE_TTL]
    for k in expired:
        del _summary_cache[k]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    print("[*] Shutting down executor")
    _executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── LLM Config ───────────────────────────────────────────────────────────────

llm = OllamaLLM(
    model="llama3",
    num_predict=400,
    num_ctx=4096,
    temperature=0.2,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=6000,
    chunk_overlap=200,
)

# ─── Models ───────────────────────────────────────────────────────────────────

class PageRequest(BaseModel):
    content: str

class QuestionRequest(BaseModel):
    content: str
    question: str

class ExplainRequest(BaseModel):
    text: str

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _cache_key(content: str) -> str:
    return hashlib.md5(content[:2000].encode()).hexdigest()


def _safe_llm(prompt: str, label: str = "LLM") -> str:
    for attempt in range(2):
        try:
            print(f"[*] LLM [{label}] calling...")
            t0 = time.time()
            res = llm.invoke(prompt)
            print(f"[+] LLM [{label}] done in {time.time()-t0:.1f}s")
            return res
        except Exception as e:
            print(f"[!] LLM [{label}] attempt {attempt+1} failed: {e}")
            if attempt == 1:
                return f"Error: {e}"
            time.sleep(1)
    return "Error: LLM unavailable"

# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "model": "llama3"}

# ─── Summarize (SSE streaming) ────────────────────────────────────────────────

@app.post("/summarize")
async def summarize(page: PageRequest):

    async def generate():
        try:
            cleanup_cache()
            content = page.content[:80000]
            key = _cache_key(content)

            # Return from cache if fresh
            if key in _summary_cache:
                ts, cached = _summary_cache[key]
                if time.time() - ts < CACHE_TTL:
                    print("[*] Returning cached summary")
                    yield _sse("start",  {"total_chunks": 1})
                    yield _sse("chunk",  {"done": 1, "total": 1, "pct": 80})
                    yield _sse("done",   cached)
                    return

            chunks = text_splitter.split_text(content)
            total  = len(chunks)

            if total == 0:
                yield _sse("error", {"message": "No content to analyze"})
                return

            print(f"[+] /summarize — {total} chunks, processing in parallel")
            yield _sse("start", {"total_chunks": total})

            partial: list[str] = [""] * total

            futures = {
                _executor.submit(
                    _safe_llm,
                    f"Summarize this research section concisely. Return only the summary.\n\nText:\n{chunk}",
                    f"chunk-{i+1}"
                ): i
                for i, chunk in enumerate(chunks)
            }

            done_count = 0
            for future in as_completed(futures):
                i = futures[future]
                try:
                    partial[i] = future.result()
                except Exception as e:
                    partial[i] = f"[Section {i+1} unavailable: {e}]"

                done_count += 1
                pct = round((done_count / total) * 80)
                yield _sse("chunk", {"done": done_count, "total": total, "pct": pct})

            yield _sse("reduce", {"message": "Composing final summary..."})

            combined = "\n\n".join(partial)
            if len(partial) == 1:
                combined = partial[0]

            final_prompt = (
                "You are an expert research assistant.\n"
                "Combine the following partial summaries into one final structured summary.\n\n"
                "Return EXACTLY in this format:\n\n"
                "TLDR:\n(2-3 sentence overview)\n\n"
                "Key Concepts:\n- concept 1\n- concept 2\n\n"
                "Key Findings:\n- finding 1\n- finding 2\n\n"
                f"Input:\n{combined}"
            )

            result = await asyncio.to_thread(_safe_llm, final_prompt, "reduce")

            data = {"summary": result, "chunks_processed": total}
            _summary_cache[key] = (time.time(), data)

            yield _sse("done", data)

        except Exception as e:
            print(f"[!] Summarize stream error: {e}")
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

# ─── Ask ──────────────────────────────────────────────────────────────────────

@app.post("/ask")
def ask(req: QuestionRequest):
    print(f"\n[+] /ask — {req.question[:80]}")
    context = req.content[:15000]
    prompt = (
        "Answer the question using only the article content.\n"
        'If not mentioned, say: "The article does not cover this."\n\n'
        f"Article:\n{context}\n\n"
        f"Question: {req.question}\nAnswer:"
    )
    result = _safe_llm(prompt, "ask")
    return {"answer": result}

# ─── Insights ─────────────────────────────────────────────────────────────────

@app.post("/insights")
def insights(page: PageRequest):
    print(f"\n[+] /insights")
    context = page.content[:15000]
    prompt = (
        "Identify the most important research insights from this text.\n"
        "Return bullet points under these headers:\n\n"
        "Problem:\nMethodology:\nFindings:\nEvidence:\nLimitations:\n\n"
        f"Content:\n{context}"
    )
    result = _safe_llm(prompt, "insights")
    return {"insights": result}

# ─── Graph ────────────────────────────────────────────────────────────────────

@app.post("/graph")
def graph(page: PageRequest):
    print(f"\n[+] /graph")
    context = page.content[:10000]
    prompt = (
        "Extract entity relationships from this text.\n"
        "Return ONLY a valid JSON array, no explanations, no markdown fences.\n\n"
        'Format: [{"source":"Entity1","relation":"relationship","target":"Entity2"}]\n\n'
        f"Content:\n{context}"
    )
    raw = _safe_llm(prompt, "graph")
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        start = cleaned.find("[")
        end   = cleaned.rfind("]")
        if start != -1 and end != -1:
            edges = json.loads(cleaned[start:end + 1])
            return {"graph": edges[:40]}
    except Exception as e:
        print(f"[!] Graph JSON parse error: {e}")
    return {"graph": []}

# ─── Bias ─────────────────────────────────────────────────────────────────────

@app.post("/bias")
def bias(page: PageRequest):
    print(f"\n[+] /bias")
    context = page.content[:15000]
    prompt = (
        "Analyze this text for bias and logical weaknesses.\n"
        "Return bullet points under these headers:\n\n"
        "Possible Bias:\nWeak Arguments:\nMissing Evidence:\nLimitations:\n\n"
        f"Content:\n{context}"
    )
    result = _safe_llm(prompt, "bias")
    return {"bias": result}

# ─── Reading Time ─────────────────────────────────────────────────────────────

@app.post("/reading-time")
def reading_time(page: PageRequest):
    words     = len(page.content.split())
    read_time = words / 200
    sum_time  = 2.0
    eff       = (1 - sum_time / read_time) * 100 if read_time > 0 else 0
    return {
        "word_count":     words,
        "reading_time":   round(read_time, 2),
        "summary_time":   sum_time,
        "efficiency_gain": round(eff, 2),
    }