# 📚 Semantic Book Recommender System

A semantic search-based book recommender that suggests books based on the **meaning** of your query—not just keywords. Powered by **Hugging Face embeddings**, **LangChain**, and **ChromaDB**, with a simple **Gradio UI** for interaction.

## 🚀 Features
- 🔍 Semantic Search using sentence embeddings
- ⚡ Fast Retrieval via Chroma vector store
- 🧠 Natural Language Queries (e.g., "books about loneliness in space")
- 🪄 Interactive Gradio Interface
- ✂️ Automatic Text Chunking with LangChain splitters

## 🛠️ Tech Stack
- Python 3.12+
- Hugging Face Transformers
- LangChain
- ChromaDB
- Gradio
- dotenv, pandas, numpy

## ⚙️ Setup Instructions

```bash
git clone https://github.com/yourusername/semantic-book-recommender.git
cd semantic-book-recommender
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file like this:
```env
CHROMA_DB_DIR=chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## ▶️ Run the App
```bash
python main.py
```

## 📜 License
MIT