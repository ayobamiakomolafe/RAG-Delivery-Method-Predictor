# RAG-Delivery-Method-Predictor

# AI Delivery Method Recommender

An intelligent recommendation system that helps businesses determine the best AI delivery approach for their needs. The application compares user business profiles against a knowledge base of 12 different AI delivery methods using **Retrieval-Augmented Generation (RAG)** and traditional LLM approaches.

## Overview

This project demonstrates the difference between two AI recommendation approaches:
- **With RAG**: Uses a vector database to retrieve relevant delivery methods based on stored knowledge
- **Without RAG**: Uses a direct LLM comparison against all 12 delivery options

Users answer 7 guided questions about their industry, goals, data size, and deployment preferences to receive targeted AI delivery recommendations.

## Features

- 🎯 **Guided Business Profile Assessment** - Seven structured questions to understand business needs
- 🔍 **Dual Recommendation Engines**:
  - RAG-based retrieval from vector database (Chroma)
  - Direct LLM comparison without external knowledge base
- 📊 **12 AI Delivery Method Options** - From simple SaaS to custom models
- 🤖 **Powered by Cohere** - Uses Cohere embeddings and chat models
- 🎨 **Interactive UI** - Built with Streamlit for ease of use
- 📚 **Vector Database** - Chroma for efficient similarity search

## Architecture

### Components

1. **Frontend**: Streamlit application (`app.py`)
   - Interactive questionnaire
   - Tab-based interface for RAG vs non-RAG comparison
   
2. **RAG System** (`set_up.py` - `delivery_method_predictor_1`):
   - Loads business profile
   - Queries Chroma vector database
   - Retrieves relevant delivery methods using Cohere embeddings
   - Uses few-shot prompting for accurate recommendations
   
3. **Direct LLM System** (`set_up.py` - `delivery_method_predictor_2`):
   - Takes business profile
   - Compares against all 12 delivery methods embedded in prompt
   - Returns recommendation based on LLM reasoning
   
4. **Vector Database Setup** (`embedding.py`):
   - Loads delivery method data from CSV
   - Generates embeddings using Cohere
   - Stores in Chroma for persistent retrieval

### Data Flow

```
User Profile Input 
    ↓
    ├─→ RAG Route: Vector DB Retrieval → LLM Reasoning → Recommendation
    └─→ Direct Route: Prompt + All Options → LLM Reasoning → Recommendation
```

## Recommendation Criteria

The system evaluates delivery methods based on:
- **Industry & Goal Fit** - Does it match the business vertical and objectives?
- **Data & Complexity Alignment** - Is the method appropriate for data volume and team capability?
- **Deployment Control** - Does it match hosting and infrastructure preferences?
- **Team Experience** - Is the complexity level suitable for the team?

## Installation

### Prerequisites
- Python 3.8+
- Cohere API key (free tier available at [cohere.com](https://cohere.com))

### Setup

1. **Clone/navigate to project directory**
   ```bash
   cd "C:\Users\USER\Desktop\RAG Test"
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   Create a `.env` file in the project root:
   ```
   COHERE_API_KEY=your_cohere_api_key_here
   ```

## Usage

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### First-Time Setup (Initialize Vector Database)

If the `chroma_db/` directory doesn't exist, initialize it:

```bash
python embedding.py
```

This will:
- Load delivery methods from `Book-1.csv`
- Generate embeddings using Cohere
- Create persistent vector database in `chroma_db/`

### Testing Components

To test individual components:

```bash
python test.py
```

This script loads and displays all documents from the CSV.

## Project Structure

```
RAG Test/
├── app.py                 # Streamlit UI and main application
├── set_up.py             # RAG and non-RAG predictor functions
├── embedding.py          # Vector database initialization
├── test.py               # Testing and validation scripts
├── requirements.txt      # Python dependencies
├── Book-1.csv           # Knowledge base of 12 delivery methods
├── chroma_db/           # Vector database (persistent storage)
│   ├── chroma.sqlite3
│   └── [chroma indices]
└── __pycache__/         # Python cache
```

## Key Files

### `app.py`
- Streamlit UI with page configuration
- Questionnaire with 7 questions covering:
  - Industry selection
  - Primary business goal
  - Desired AI outcome
  - Team AI experience level
  - Available data size
  - Frontend deployment preference
  - Backend deployment preference
- Tab interface to compare RAG vs non-RAG approaches

### `set_up.py`
- **`delivery_method_predictor_1(profile)`**: RAG-based recommendation
  - Loads Chroma vector database
  - Structures business profile from user input
  - Retrieves top-2 relevant methods
  - Uses LLM with system prompt for reasoning
  
- **`delivery_method_predictor_2(profile)`**: Direct LLM recommendation
  - Embeds all 12 delivery methods in prompt
  - Compares profile against each option
  - Uses temperature=0.1 for stable outputs

### `embedding.py`
- One-time setup script
- Loads CSV data using LangChain CSVLoader
- Creates Cohere embeddings
- Persists vector database to `chroma_db/`

### `Book-1.csv`
Knowledge base containing 12 AI delivery methods with:
- Delivery Method name
- Simple Description
- Typical Best For use cases
- Hosting Control level
- Complexity rating

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web UI framework |
| langchain | LLM orchestration |
| langchain-community | Document loaders and embeddings |
| langchain-cohere | Cohere integration |
| langchain-chroma | Vector database integration |
| langchain_core | Core primitives |
| python-dotenv | Environment variable management |
| numpy | Numerical operations |
| pandas | Data manipulation |

See `requirements.txt` for complete list.

## Configuration

### LLM Settings
- **Model**: ChatCohere
- **Temperature**: 0.1 (for stable, deterministic outputs)
- **Top-K Retrieval**: 2 documents (RAG mode)

### Cohere API
- Uses `CohereEmbeddings` with user_agent="hhh"
- Supports free tier (limited requests)
- Requires API key in `.env` file

## Troubleshooting

### Vector Database Not Found
```bash
python embedding.py
```
Reinitialize the vector database if `chroma_db/` is missing or corrupted.

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### API Key Issues
- Verify `.env` file exists in project root
- Check `COHERE_API_KEY` is set correctly
- Test API key at [cohere.com/dashboard](https://cohere.com/dashboard)

### Streamlit Port Already in Use
```bash
streamlit run app.py --logger.level=debug --server.port 8502
```

## Future Enhancements

- [ ] Support for additional LLM providers (OpenAI, HuggingFace)
- [ ] User feedback loop to improve recommendations
- [ ] Persistent history of recommendations
- [ ] Export recommendations as PDF reports
- [ ] Admin panel to update delivery methods
- [ ] Multi-language support
- [ ] Advanced filtering options

## License

This project is provided as-is for educational and evaluation purposes.

## Support

For issues or questions, refer to:
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [Cohere Documentation](https://docs.cohere.com/)
