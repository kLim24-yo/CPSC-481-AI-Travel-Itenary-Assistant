# AI Travel Itinerary Assistant

A travel assistant that uses RAG to answer questions from a travel guide. It is built with LangChain, Pinecone, and Streamlit.

## Project Structure

```text
ai-travel-assistant/
├── frontend/          # Streamlit app
│   └── app.py
├── backend/           # RAG and data processing logic
│   ├── ingest.py      # Reads the PDF, creates embeddings, stores them in Pinecone
│   └── rag.py         # Handles user questions and retrieval
├── data/              # Travel guide PDF
│   └── paris_guide.pdf
├── scripts/
│   └── setup_check.py # Checks that your environment is set up correctly
├── .env.example       # Example environment file
├── requirements.txt
└── README.md
```
## Usage
1. Unzip the project
bashcd ai-travel-assistant

2. Install dependencies  
```pip install -r requirements.txt```
3. Set up your .env  
```cp .env.example .env```  
Then open .env and fill in:  
```text
PINECONE_API_KEY=<Your_PineCone_API_Key>   ← your full key  
PINECONE_INDEX_NAME=travel-guide  
HUGGINGFACE_API_TOKEN=<hf_API_TOKEN>
```
5. Check everything is ready  
```python scripts/setup_check.py```
6. Load the PDF into Pinecone (run once)  
```python backend/ingest.py```
7. Launch the app  
```streamlit run frontend/app.py```