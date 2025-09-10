# Voice-Enabled RAG System

A comprehensive AI assistant system that combines speech recognition, retrieval-augmented generation (RAG), and text-to-speech capabilities using locally hosted LLMs through Ollama.

## 🚀 Features

- **Voice Interaction**: Speech-to-text using OpenAI Whisper and text-to-speech using pyttsx3/gTTS
- **RAG (Retrieval Augmented Generation)**: Upload and query documents using ChromaDB vector database
- **Local LLM Integration**: Uses Ollama for running models locally (Llama 2, Mistral, etc.)
- **Web Search Integration**: Real-time web search and content extraction
- **Document Processing**: Support for PDF, DOCX, TXT, CSV, and Excel files
- **Modern Web Interface**: Interactive Streamlit frontend with real-time audio recording
- **RESTful API**: Comprehensive FastAPI backend with streaming support

## 🏗️ Architecture

```
voice-rag-project/
├── backend/                    # FastAPI backend services
│   ├── main.py                # API endpoints and server
│   └── services/              # Core service modules
│       ├── speech_service.py  # Whisper STT integration
│       ├── tts_service.py     # Text-to-speech services
│       ├── rag_service.py     # ChromaDB and document processing
│       ├── llm_service.py     # Ollama LLM integration  
│       └── api_service.py     # External API services
├── frontend/                  # Streamlit web interface
│   └── app.py                # Main UI application
├── data/                      # Data storage
│   ├── documents/            # Uploaded documents
│   └── vector_store/         # ChromaDB database
├── requirements.txt          # Python dependencies
├── .env                      # Environment configuration
└── README.md                # This file
```

## 🛠️ Installation

### Prerequisites

1. **Python 3.8+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai)
3. **Audio dependencies** (OS-specific):
   - **Linux**: `sudo apt-get install portaudio19-dev python3-pyaudio`
   - **macOS**: `brew install portaudio`
   - **Windows**: Usually works out of the box

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd voice-rag-project
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and start Ollama**:
   ```bash
   # Install Ollama (follow instructions at ollama.ai)
   
   # Pull a model (e.g., Llama 2 7B)
   ollama pull llama2:7b
   
   # Start Ollama server (usually runs automatically)
   ollama serve
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env file with your preferences
   ```

5. **Create data directories**:
   ```bash
   mkdir -p data/documents data/vector_store temp
   ```

## 🚀 Usage

### Starting the System

1. **Start the FastAPI backend**:
   ```bash
   cd backend
   python main.py
   ```
   The API will be available at `http://localhost:8000`

2. **Start the Streamlit frontend**:
   ```bash
   cd frontend
   streamlit run app.py
   ```
   The web interface will open at `http://localhost:8501`

### Using the Interface

1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, TXT, or CSV files
2. **Text Chat**: Type messages in the chat input
3. **Voice Chat**: Use the voice input tab to record or upload audio
4. **Configure Settings**: Enable/disable RAG, web search, and TTS in the sidebar

### API Endpoints

The FastAPI backend provides comprehensive REST endpoints:

- `POST /api/chat` - Text-based chat
- `POST /api/voice-chat` - Complete voice pipeline (STT → Chat → TTS)
- `POST /api/speech/transcribe` - Speech-to-text
- `POST /api/tts/speak` - Text-to-speech
- `POST /api/documents/upload` - Upload and index documents
- `GET /api/documents/list` - List uploaded documents
- `DELETE /api/documents/{filename}` - Delete documents
- `POST /api/rag/search` - Search document database
- `POST /api/search` - External web search
- `GET /api/weather/{location}` - Weather information
- `GET /api/health` - System health check

Full API documentation available at `http://localhost:8000/docs` when running.

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2:7b

# ChromaDB Configuration  
CHROMA_DB_PATH=./data/vector_store

# Audio Configuration
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1

# TTS Configuration
TTS_ENGINE=pyttsx3  # Options: pyttsx3, gtts
TTS_VOICE_RATE=200
TTS_VOICE_VOLUME=1.0

# RAG Configuration
MAX_CONTEXT_LENGTH=4000
MAX_RETRIEVAL_DOCS=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# External APIs (Optional)
DUCKDUCKGO_ENABLED=true
SERPAPI_KEY=your_key_here
OPENWEATHER_API_KEY=your_key_here

# Logging
LOG_LEVEL=INFO
```

### Supported File Types

- **Documents**: PDF, DOCX, TXT
- **Data**: CSV, XLSX, XLS
- **Audio**: WAV, MP3, M4A, OGG

### Available LLM Models

Any Ollama-compatible model can be used:

```bash
# Popular options
ollama pull llama2:7b        # Llama 2 7B (recommended)
ollama pull llama2:13b       # Llama 2 13B (more capable)
ollama pull mistral:7b       # Mistral 7B (fast)
ollama pull codellama:7b     # Code Llama (for coding)
ollama pull vicuna:7b        # Vicuna 7B
```

## 🎯 Core Features

### Voice Processing

- **Speech-to-Text**: OpenAI Whisper with noise reduction and VAD
- **Text-to-Speech**: Multiple engines (pyttsx3, gTTS, Coqui TTS)
- **Real-time Audio**: WebRTC integration for live recording
- **Audio Preprocessing**: Noise reduction, normalization, format conversion

### RAG System

- **Vector Database**: ChromaDB with cosine similarity search
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Document Processing**: Intelligent text extraction and chunking
- **Metadata Tracking**: Source attribution and document management

### LLM Integration

- **Local Models**: Full privacy with Ollama-hosted models
- **Streaming Responses**: Real-time response generation
- **Context Management**: Dynamic prompt building with RAG context
- **Model Switching**: Runtime model selection and management

### External APIs

- **Web Search**: DuckDuckGo integration with content extraction
- **News Search**: Real-time news article retrieval
- **Weather**: Location-based weather information
- **Academic Search**: arXiv paper search integration

## 🔧 Development

### Running in Development Mode

1. **Backend with auto-reload**:
   ```bash
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Frontend with auto-refresh**:
   ```bash
   cd frontend
   streamlit run app.py --server.runOnSave true
   ```

### Testing

```bash
# Test backend endpoints
curl http://localhost:8000/health

# Test document upload
curl -X POST -F "file=@test.pdf" http://localhost:8000/api/documents/upload

# Test chat
curl -X POST -H "Content-Type: application/json" \
     -d '{"message":"Hello!","use_rag":true}' \
     http://localhost:8000/api/chat
```

### Project Structure Details

```
backend/services/
├── speech_service.py     # Whisper integration, audio processing
├── tts_service.py       # Multi-engine TTS with voice controls
├── rag_service.py       # ChromaDB operations, document indexing
├── llm_service.py       # Ollama API client, prompt engineering
└── api_service.py       # External API integrations

frontend/
└── app.py              # Streamlit UI with WebRTC audio recording
```

## 🚨 Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
   ```bash
   # Check if Ollama is running
   ollama list
   
   # Start Ollama service
   ollama serve
   ```

2. **Audio Recording Issues**:
   - **Linux**: Install portaudio: `sudo apt-get install portaudio19-dev`
   - **macOS**: Install portaudio: `brew install portaudio`
   - **Windows**: Ensure microphone permissions are granted

3. **Document Upload Failures**:
   - Check file permissions in `data/documents/`
   - Ensure supported file format
   - Verify file is not corrupted

4. **ChromaDB Issues**:
   ```bash
   # Clear and reinitialize database
   rm -rf data/vector_store
   mkdir data/vector_store
   ```

5. **Memory Issues with Large Models**:
   - Use smaller models like `llama2:7b` instead of `llama2:13b`
   - Increase system RAM or use model quantization
   - Adjust `MAX_CONTEXT_LENGTH` in .env

### Performance Optimization

1. **For better RAG performance**:
   - Use smaller chunk sizes (300-500 tokens)
   - Increase chunk overlap (50-100 tokens)
   - Use better embedding models

2. **For faster inference**:
   - Use quantized models (Q4, Q8)
   - Reduce `max_tokens` in requests
   - Use GPU acceleration with Ollama

3. **For better audio quality**:
   - Use higher sample rates for recording
   - Enable noise reduction
   - Use better TTS engines (Coqui vs pyttsx3)

## 📚 Dependencies

### Core Libraries

- **FastAPI**: Modern async web framework
- **Streamlit**: Interactive web applications
- **ChromaDB**: Vector database for embeddings
- **LangChain**: RAG pipeline orchestration
- **Sentence Transformers**: Text embeddings
- **OpenAI Whisper**: Speech recognition
- **Ollama**: Local LLM serving

### Audio Processing

- **PyAudio**: Audio I/O
- **librosa**: Audio analysis
- **webrtcvad**: Voice activity detection
- **noisereduce**: Audio denoising

### Document Processing

- **PyPDF2 & pdfplumber**: PDF text extraction
- **python-docx**: Word document processing
- **pandas**: Data file processing

### External APIs

- **httpx**: Async HTTP client
- **BeautifulSoup4**: Web scraping
- **newspaper3k**: Article extraction
- **duckduckgo-search**: Web search

## Sample Screenshot

<img width="1920" height="1128" alt="Screenshot 2025-09-10 at 13 48 48" src="https://github.com/user-attachments/assets/7eabf83f-e444-408f-87b7-bc0f929eae89" />

---

**Built with ❤️ for the AI community**
