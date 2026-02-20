# ─────────────────────────────────────────────────────────────────────────────
# Phi-3 ML Q&A — FastAPI Server
# Run with:  uvicorn app:app --host 0.0.0.0 --port 8000
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Configuration — update MODEL_PATH to where you unzipped fine_tuned_phi3 ──
MODEL_PATH     = r"C:\fine_tuned_ph3\fine_tuned_phi3"   # <-- change this if you put it elsewhere
MAX_NEW_TOKENS = 200    # detailed ML answers; lower to 80 for faster/shorter replies
TEMPERATURE    = 0.7
TOP_P          = 0.9
# ─────────────────────────────────────────────────────────────────────────────

# Global model state — loaded once at startup, shared across all requests
_tokenizer = None
_model     = None
_device    = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once when the server starts; clean up when it stops."""
    global _tokenizer, _model, _device

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"\n\n❌ Model folder not found: {MODEL_PATH}\n"
            "   Please update MODEL_PATH in app.py to point to your fine_tuned_phi3 folder.\n"
        )

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Phi-3 from: {MODEL_PATH}")
    print(f"Device : {_device.upper()}  |  Quantization: 4-bit (fits in 4GB VRAM)")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)

    if _device == "cuda":
        # 4-bit quantization — shrinks 7GB fp16 model to ~2GB, fits entirely in RTX 3050's 4GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,   # fast fp16 matrix math
            bnb_4bit_use_double_quant=True,          # nested quant saves ~0.4 GB extra
            bnb_4bit_quant_type="nf4",              # NormalFloat4 — best quality for LLMs
        )
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,
        )
    else:
        # CPU fallback — no quantization
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=False,
        )
    _model.eval()   # disables dropout — required for inference

    # ── Warm-up pass ────────────────────────────────────────────────────────
    # The very first generation call compiles CUDA kernels and is slow (~5-10s).
    # Running one dummy pass here means the FIRST real API request is already fast.
    print("Warming up CUDA kernels (one dummy pass)...")
    _dummy = _tokenizer("Hello", return_tensors="pt").to(_device)
    with torch.no_grad():
        _model.generate(**_dummy, max_new_tokens=5, do_sample=False)
    del _dummy

    print("✅ Model ready!  Server live at http://localhost:8000")
    print("   Docs UI:       http://localhost:8000/docs\n")

    yield   # server is now running and serving requests

    # ── Cleanup on shutdown ──────────────────────────────────────────────────
    del _model, _tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model unloaded.")


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Phi-3 ML Q&A API",
    description="Fine-tuned Phi-3-mini-4k-instruct for Machine Learning questions",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ───────────────────────────────────────────────
class AskRequest(BaseModel):
    question:   str
    max_tokens: int = MAX_NEW_TOKENS   # caller can override per request


class AskResponse(BaseModel):
    question:     str
    answer:       str
    time_seconds: float
    tokens_generated: int


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def chat_ui():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ML Q&A — Phi-3</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; display: flex; flex-direction: column; height: 100vh; }
    header { background: #1a1d27; padding: 16px 24px; border-bottom: 1px solid #2a2d3e; }
    header h1 { font-size: 1.2rem; color: #7c9ef8; }
    header p  { font-size: 0.8rem; color: #666; margin-top: 2px; }
    #chat { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
    .bubble { max-width: 75%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; font-size: 0.95rem; }
    .user-wrap { display: flex; justify-content: flex-end; }
    .user   { background: #2e4a9e; border-radius: 12px 12px 2px 12px; }
    .bot-wrap  { display: flex; justify-content: flex-start; }
    .bot    { background: #1e2130; border: 1px solid #2a2d3e; border-radius: 12px 12px 12px 2px; }
    .meta   { font-size: 0.72rem; color: #555; margin-top: 6px; }
    .thinking { color: #555; font-style: italic; }
    footer  { background: #1a1d27; border-top: 1px solid #2a2d3e; padding: 16px 24px; display: flex; gap: 10px; }
    #input  { flex: 1; background: #0f1117; border: 1px solid #2a2d3e; border-radius: 8px; padding: 12px 16px; color: #e0e0e0; font-size: 0.95rem; resize: none; height: 48px; outline: none; }
    #input:focus { border-color: #7c9ef8; }
    #send   { background: #2e4a9e; color: #fff; border: none; border-radius: 8px; padding: 0 24px; font-size: 0.95rem; cursor: pointer; }
    #send:hover { background: #3d5bbf; }
    #send:disabled { background: #333; cursor: not-allowed; }
  </style>
</head>
<body>
  <header>
    <h1>ML Q&amp;A — Phi-3</h1>
    <p>Fine-tuned on Machine Learning &amp; Model-Making questions</p>
  </header>
  <div id="chat"></div>
  <footer>
    <textarea id="input" placeholder="Ask a machine learning question... (Enter to send)" rows="1"></textarea>
    <button id="send">Send</button>
  </footer>

  <script>
    const chat  = document.getElementById('chat');
    const input = document.getElementById('input');
    const send  = document.getElementById('send');

    function addBubble(text, role, meta) {
      const wrap = document.createElement('div');
      wrap.className = role === 'user' ? 'user-wrap' : 'bot-wrap';
      const bubble = document.createElement('div');
      bubble.className = 'bubble ' + role;
      bubble.textContent = text;
      if (meta) {
        const m = document.createElement('div');
        m.className = 'meta';
        m.textContent = meta;
        bubble.appendChild(m);
      }
      wrap.appendChild(bubble);
      chat.appendChild(wrap);
      chat.scrollTop = chat.scrollHeight;
      return bubble;
    }

    async function askQuestion() {
      const q = input.value.trim();
      if (!q) return;
      input.value = '';
      send.disabled = true;

      addBubble(q, 'user');
      const thinkingBubble = addBubble('Thinking...', 'bot thinking');

      try {
        const res = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q })
        });
        const data = await res.json();
        thinkingBubble.textContent = data.answer;
        const m = document.createElement('div');
        m.className = 'meta';
        m.textContent = data.time_seconds + 's · ' + data.tokens_generated + ' tokens';
        thinkingBubble.appendChild(m);
      } catch(e) {
        thinkingBubble.textContent = 'Error: could not reach the server.';
      }

      send.disabled = false;
      input.focus();
    }

    send.addEventListener('click', askQuestion);
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); askQuestion(); }
    });
  </script>
</body>
</html>
""")


@app.get("/health", summary="Check server + model status")
def health():
    return {
        "status":          "ok" if _model is not None else "model_not_loaded",
        "model_loaded":    _model is not None,
        "device":          _device,
        "cuda_available":  torch.cuda.is_available(),
        "gpu_name":        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_free_gb":    round(torch.cuda.mem_get_info()[0] / 1e9, 2) if torch.cuda.is_available() else None,
    }


@app.post("/ask", response_model=AskResponse, summary="Ask an ML question")
def ask(req: AskRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    # ── Build Phi-3 instruct prompt ──────────────────────────────────────────
    prompt = (
        "<|user|>\n"
        "Answer the following machine learning question clearly and concisely.\n\n"
        f"### Input:\n{req.question.strip()}<|end|>\n"
        "<|assistant|>\n"
    )

    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
    prompt_len = inputs["input_ids"].shape[1]

    # ── Generate ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            use_cache=True,          # KV-cache: critical for generation speed
            pad_token_id=_tokenizer.eos_token_id,
            eos_token_id=_tokenizer.convert_tokens_to_ids("<|end|>"),
        )
    elapsed = time.perf_counter() - t0

    # ── Decode only the newly generated tokens (skip echoing the prompt) ─────
    generated_ids = output_ids[0][prompt_len:]
    answer = _tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    tokens_out = len(generated_ids)

    return AskResponse(
        question=req.question,
        answer=answer,
        time_seconds=round(elapsed, 3),
        tokens_generated=tokens_out,
    )
