"""Medical knowledge retrieval modules.

The package keeps keyword, FAISS, and Qdrant retrievers behind the same minimal
``retrieve(query, top_k)`` interface so RAG experiments can isolate retrieval
infrastructure from prompts, model weights, and evaluation code.
"""
