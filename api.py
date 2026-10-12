"""Minimal HTTP API. Run: uvicorn api:app --port 8000
Open http://localhost:8000 for a tiny chat page, or /docs for the Swagger UI.
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from rag import RAG

app = FastAPI(title="SDAIA RAG Assistant")
rag = RAG()


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Question):
    """Full answer plus the retrieved source chunks."""
    return rag.ask(q.question)


@app.post("/ask/stream")
def ask_stream(q: Question):
    """Answer streamed as plain text, token by token."""
    return StreamingResponse(rag.stream(q.question), media_type="text/plain; charset=utf-8")


PAGE = """<!doctype html><meta charset="utf-8"><title>SDAIA RAG Assistant</title>
<body style="font-family:sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem">
<h3>SDAIA AI Assistant</h3>
<textarea id="q" rows="3" style="width:100%" dir="auto" placeholder="Ask in Arabic or English"></textarea>
<button onclick="go()">Ask</button>
<pre id="a" dir="auto" style="white-space:pre-wrap"></pre>
<script>
async function go(){
  const a=document.getElementById('a'); a.textContent='';
  const r=await fetch('/ask/stream',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:document.getElementById('q').value})});
  const rd=r.body.getReader(), dec=new TextDecoder();
  for(;;){const {done,value}=await rd.read(); if(done)break; a.textContent+=dec.decode(value,{stream:true});}
}
</script>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
