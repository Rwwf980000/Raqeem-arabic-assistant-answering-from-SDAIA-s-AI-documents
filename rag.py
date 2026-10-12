"""The RAG chain: question -> retrieve chunks -> prompt -> Ollama LLM (streamed)."""
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

import config
from ingest import get_embeddings

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant that answers questions about Saudi Arabia's AI policies, "
     "using only the SDAIA documents provided below.\n"
     "Rules:\n"
     "- Answer in the same language as the question (Arabic or English), even if the "
     "context is in another language.\n"
     "- Use only the context. If it does not contain the answer, say you could not find "
     "it in the SDAIA documents.\n"
     "- Cite the sources you used as [file, p.N].\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


class RAG:
    def __init__(self, embeddings=None, llm=None):
        store = Chroma(
            collection_name=config.COLLECTION,
            embedding_function=embeddings or get_embeddings(),
            persist_directory=str(config.DB_DIR),
        )
        # MMR picks relevant chunks that are also different from each other,
        # so the answer is not built from five near-identical passages.
        self.retriever = store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config.TOP_K, "fetch_k": config.TOP_K * 4},
        )
        self.llm = llm or ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_URL,
            temperature=config.TEMPERATURE,
        )
        self.chain = PROMPT | self.llm | StrOutputParser()

    def retrieve(self, question: str):
        docs = self.retriever.invoke(question)
        context = "\n\n".join(
            f"[{d.metadata['source']}, p.{d.metadata['page']}]\n{d.page_content}" for d in docs
        )
        sources = [
            {"source": d.metadata["source"], "page": d.metadata["page"], "text": d.page_content[:300]}
            for d in docs
        ]
        return context, sources

    def ask(self, question: str) -> dict:
        context, sources = self.retrieve(question)
        answer = self.chain.invoke({"context": context, "question": question})
        return {"answer": answer, "sources": sources}

    def stream(self, question: str):
        """Yield the answer token by token, as the model generates it."""
        context, _ = self.retrieve(question)
        yield from self.chain.stream({"context": context, "question": question})


if __name__ == "__main__":
    # Quick terminal chat, like the tutorial: python rag.py
    rag = RAG()
    while (q := input("\nYou: ").strip()) not in ("exit", "quit"):
        print("Assistant: ", end="", flush=True)
        for token in rag.stream(q):
            print(token, end="", flush=True)
        print()
