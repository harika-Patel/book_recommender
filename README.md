# Semantic Book Recommender System

This project is a semantic search-based book recommender that suggests books based on the **meaning** of a user’s query rather than just keywords. It uses **Hugging Face embeddings**, **LangChain**, and **ChromaDB**, and features an interactive interface built with **Gradio**.

## Features

- Semantic search using sentence-transformer embeddings
- Fast and scalable vector search via ChromaDB
- Natural language query support (e.g., "books about loneliness in space")
- Interactive web UI built with Gradio
- Book dataset with metadata and descriptions

## Technologies Used

- Python 3.12+
- Hugging Face Transformers
- LangChain
- ChromaDB
- Gradio
- Pandas, NumPy, Matplotlib, Seaborn
- dotenv (for environment variables)


## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/semantic-book-recommender.git
cd semantic-book-recommender
2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt

```

## Run the App
```bash
python main.py
```
# Screenshots
<img width="3832" height="2112" alt="Screenshot 2025-07-18 165643" src="https://github.com/user-attachments/assets/6d48bd34-e522-4b20-99a1-d634369d79f3" />
<img width="3809" height="2094" alt="Screenshot 2025-07-18 165357" src="https://github.com/user-attachments/assets/a07de688-aea5-41f6-9561-ab09bd9b6632" />


# How It Works
1.Loads and preprocesses book text or metadata using Pandas and LangChain
2.Splits long text into chunks with CharacterTextSplitter
3.Embeds the chunks using Hugging Face sentence transformers
4.Stores embeddings in ChromaDB
5.Accepts user queries, embeds them, and retrieves the top similar chunks
6.Displays matching book descriptions or titles via the Gradio UI

