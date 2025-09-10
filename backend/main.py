from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn
import os
import logging
from typing import List, Optional, Dict, Any
import asyncio
from pydantic import BaseModel
import tempfile
import shutil

# Import services
from services.speech_service import SpeechService
from services.tts_service import TTSService
from services.rag_service import RAGService
from services.llm_service import LLMService
from services.api_service import APIService

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Voice-Enabled RAG System",
    description="AI Assistant with Speech, RAG, and External API Integration",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
speech_service = SpeechService()
tts_service = TTSService(os.getenv('TTS_ENGINE', 'pyttsx3'))
rag_service = RAGService()  # Fixed - no arguments
llm_service = LLMService()
api_service = APIService()

# Pydantic models for requests
class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True
    use_web_search: bool = False
    temperature: float = 0.7
    max_tokens: int = 2000

class TTSRequest(BaseModel):
    text: str
    save_file: bool = False

class SearchRequest(BaseModel):
    query: str
    max_results: int = 5
    search_type: str = "web"  # web, news, academic

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Initializing services...")
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/documents", exist_ok=True)
    os.makedirs("data/vector_store", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    # Initialize RAG service
    await rag_service.initialize()

    # Initialize LLM service
    if not await llm_service.check_model_availability():
        logger.warning("LLM model not available, attempting to pull...")
        if await llm_service.pull_model():
            logger.info("Successfully pulled LLM model")
        else:
            logger.error("Failed to pull LLM model")
    
    logger.info("Services initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup services on shutdown"""
    await llm_service.close()
    await api_service.close()
    logger.info("Voice-Enabled RAG System shutdown complete.")

# Health check endpoint
@app.get("/health")
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check LLM availability
        llm_available = await llm_service.check_model_availability()
        
        # Get RAG stats
        rag_stats = rag_service.get_collection_stats()
        
        return {
            "status": "healthy",
            "services": {
                "llm": llm_available,
                "rag": rag_stats,
                "speech": "available",
                "tts": "available"
            },
            "model_info": llm_service.get_model_info()
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Speech-to-Text endpoints
@app.post("/api/speech/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Transcribe uploaded audio file."""
    try:
        # Save uploaded file temporarily
        temp_path = f"./temp/{audio_file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        # Transcribe
        transcription = await speech_service.transcribe_file(temp_path)

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {"transcription": transcription}

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/speech/transcribe-bytes")
async def transcribe_audio_bytes(audio_data: bytes = Form(...)):
    """Transcribe audio from raw bytes."""
    try:
        transcription = await speech_service.transcribe_audio(audio_data)
        return {"transcription": transcription}
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Text-to-Speech endpoints
@app.post("/api/tts/speak")
async def text_to_speech_endpoint(request: TTSRequest):
    """Convert text to speech."""
    try:
        if request.save_file:
            result = await tts_service.speak_text(request.text, save_file=True)
            if result:
                return {"audio_file": result, "message": "Audio file created successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to create audio file")
        else:
            await tts_service.speak_text(request.text, save_file=False)
            return {"message": "Speech synthesis completed"}

    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tts/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve generated audio files."""
    # Check both wav and mp3 extensions
    for ext in ['.wav', '.mp3']:
        base_name = filename.replace('.wav', '').replace('.mp3', '')
        file_path = f"./temp/{base_name}{ext}"
        
        if os.path.exists(file_path):
            media_type = "audio/wav" if ext == '.wav' else "audio/mpeg"
            return FileResponse(
                file_path, 
                media_type=media_type,
                filename=f"{base_name}{ext}"
            )
    
    # Try the filename as-is
    file_path = f"./temp/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    
    logger.error(f"Audio file not found: {filename}")
    raise HTTPException(status_code=404, detail="Audio file not found")

# Document management endpoints
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document for RAG."""
    try:
        # Save uploaded file
        file_path = f"./data/documents/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Add to RAG system
        success = await rag_service.add_document(file_path)

        if success:
            return {"message": f"Document {file.filename} uploaded and indexed successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to process document")

    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/list")
async def list_documents():
    """List all uploaded documents."""
    try:
        documents = rag_service.list_documents()
        stats = rag_service.get_collection_stats()
        return {
            "documents": documents,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document from RAG system."""
    try:
        success = await rag_service.delete_document(filename)
        if success:
            # Also delete physical file
            file_path = f"./data/documents/{filename}"
            if os.path.exists(file_path):
                os.remove(file_path)
            return {"message": f"Document {filename} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# RAG search endpoint
@app.post("/api/rag/search")
async def search_documents(query: str, max_results: int = 5):
    """Search documents using semantic similarity."""
    try:
        results = await rag_service.search_documents(query, max_results)
        return {"results": results}
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# External API endpoints
@app.post("/api/search")
async def search_external(request: SearchRequest):
    """Search external sources."""
    try:
        if request.search_type == "web":
            results = await api_service.search_web(request.query, request.max_results)
        elif request.search_type == "news":
            results = await api_service.get_news(request.query, request.max_results)
        elif request.search_type == "academic":
            results = await api_service.search_academic(request.query, request.max_results)
        else:
            raise HTTPException(status_code=400, detail="Invalid search type")

        return {"results": results}

    except Exception as e:
        logger.error(f"External search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/weather/{location}")
async def get_weather(location: str):
    """Get weather information."""
    try:
        weather_data = await api_service.get_weather(location)
        return weather_data
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main chat endpoint
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint with RAG and web search integration."""
    try:
        context_docs = []

        # Get relevant documents if RAG is enabled
        if request.use_rag:
            rag_results = await rag_service.search_documents(
                request.message,
                int(os.getenv('MAX_RETRIEVAL_DOCS', 5))
            )
            context_docs.extend(rag_results)

        # Get web search results if enabled
        if request.use_web_search:
            web_results = await api_service.search_and_extract(request.message, 3)
            # Convert web results to context format
            for result in web_results:
                context_docs.append({
                    'content': result.get('extracted_content', result.get('snippet', '')),
                    'metadata': {
                        'filename': result.get('title', 'Web Result'),
                        'source': result.get('url', ''),
                        'type': 'web_search'
                    }
                })

        # Generate response using LLM
        response = await llm_service.generate_response(
            request.message,
            context_docs=context_docs,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        return {
            "response": response,
            "context_used": len(context_docs),
            "sources": [doc.get('metadata', {}).get('filename', 'Unknown') for doc in context_docs]
        }

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Streaming chat endpoint
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint."""
    try:
        context_docs = []

        # Get context (same as regular chat)
        if request.use_rag:
            rag_results = await rag_service.search_documents(request.message, 5)
            context_docs.extend(rag_results)

        if request.use_web_search:
            web_results = await api_service.search_and_extract(request.message, 3)
            for result in web_results:
                context_docs.append({
                    'content': result.get('extracted_content', result.get('snippet', '')),
                    'metadata': {
                        'filename': result.get('title', 'Web Result'),
                        'source': result.get('url', ''),
                        'type': 'web_search'
                    }
                })

        # Stream response
        async def generate_stream():
            async for chunk in llm_service.stream_response(
                request.message,
                context_docs=context_docs,
                temperature=request.temperature
            ):
                yield f"data: {chunk}\n\n"

        return StreamingResponse(generate_stream(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice-chat")
async def voice_chat(
    audio_file: UploadFile = File(...),
    use_rag: bool = Form(True),
    use_web_search: bool = Form(False),
    return_audio: bool = Form(True)
):
    """Complete voice chat pipeline: STT -> Chat -> TTS."""
    try:
        # Step 1: Speech-to-Text
        temp_audio_path = f"./temp/voice_input_{audio_file.filename}"
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        logger.info(f"Processing audio file: {temp_audio_path}")
        transcription = await speech_service.transcribe_file(temp_audio_path)
        
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        logger.info(f"Transcription result: '{transcription}'")
        
        if not transcription or not transcription.strip():
            logger.warning("Empty transcription received")
            return {
                "transcription": "",
                "response": "I couldn't understand the audio. Please try speaking more clearly.",
                "audio_file": None,
                "context_used": 0,
                "error": "Empty transcription"
            }
        
        # Step 2: Generate chat response
        context_docs = []
        
        if use_rag:
            try:
                rag_results = await rag_service.search_documents(transcription, 5)
                context_docs.extend(rag_results)
                logger.info(f"Retrieved {len(rag_results)} RAG documents")
            except Exception as e:
                logger.error(f"RAG search failed: {e}")
        
        if use_web_search:
            try:
                web_results = await api_service.search_and_extract(transcription, 3)
                for result in web_results:
                    context_docs.append({
                        'content': result.get('extracted_content', result.get('snippet', '')),
                        'metadata': {
                            'filename': result.get('title', 'Web Result'),
                            'source': result.get('url', ''),
                            'type': 'web_search'
                        }
                    })
                logger.info(f"Retrieved {len(web_results)} web search results")
            except Exception as e:
                logger.error(f"Web search failed: {e}")
        
        logger.info(f"Sending to LLM: '{transcription}' with {len(context_docs)} context docs")
        
        # Step 3: Generate LLM response
        try:
            response_text = await llm_service.generate_response(
                transcription,
                context_docs=context_docs,
                temperature=0.7,
                max_tokens=2000
            )
            logger.info(f"LLM response received: '{response_text[:100]}{'...' if len(response_text) > 100 else ''}'")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            response_text = f"I encountered an error processing your request: {str(e)}"
        
        # Step 4: Text-to-Speech (if requested)
        audio_file_path = None
        if return_audio and response_text:
            try:
                audio_file_path = await tts_service.speak_text(response_text, save_file=True)
                logger.info(f"TTS audio generated: {audio_file_path}")
            except Exception as e:
                logger.error(f"TTS generation failed: {e}")
        
        return {
            "transcription": transcription,
            "response": response_text,
            "audio_file": audio_file_path,
            "context_used": len(context_docs)
        }
        
    except Exception as e:
        logger.error(f"Voice chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# LLM management endpoints
@app.get("/api/llm/models")
async def list_models():
    """List available LLM models."""
    try:
        models = await llm_service.list_available_models()
        current_model = llm_service.get_model_info()
        return {
            "available_models": models,
            "current_model": current_model
        }
    except Exception as e:
        logger.error(f"List models error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/llm/model/{model_name}")
async def set_model(model_name: str):
    """Set the active LLM model."""
    try:
        # Check if model is available
        available_models = await llm_service.list_available_models()
        if model_name not in available_models:
            # Try to pull the model
            llm_service.set_model(model_name)  # Set it first, then try to pull
            success = await llm_service.pull_model()
            if not success:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found and pull failed")

        llm_service.set_model(model_name)
        return {"message": f"Model set to {model_name}"}

    except Exception as e:
        logger.error(f"Set model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration endpoints
@app.get("/api/config")
async def get_config():
    """Get current system configuration."""
    return {
        "llm_model": llm_service.model_name,
        "llm_host": llm_service.host,
        "tts_engine": tts_service.engine_type,
        "rag_stats": rag_service.get_collection_stats(),
        "max_context_length": os.getenv('MAX_CONTEXT_LENGTH', 4000),
        "chunk_size": os.getenv('CHUNK_SIZE', 500)
    }

@app.get("/api/tts/voices")
async def get_tts_voices():
    """Get available TTS voices."""
    try:
        voices = tts_service.get_available_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Get voices error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@app.post("/api/summarize")
async def summarize_text(text: str = Form(...), max_length: int = Form(200)):
    """Summarize provided text."""
    try:
        summary = await llm_service.summarize_text(text, max_length)
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/rag/clear")
async def clear_rag_database():
    """Clear all documents from RAG database."""
    try:
        rag_service.clear_collection()
        return {"message": "RAG database cleared successfully"}
    except Exception as e:
        logger.error(f"Clear RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test-components")
async def test_components(request: dict):
    """Test individual components for debugging"""
    results = {}
    
    # Test LLM with sample text
    if "test_text" in request:
        try:
            response = await llm_service.generate_response(request["test_text"])
            results["llm_test"] = response
        except Exception as e:
            results["llm_error"] = str(e)
    
    # Test TTS
    if "tts_text" in request:
        try:
            audio_file = await tts_service.speak_text(request["tts_text"], save_file=True)
            results["tts_test"] = audio_file
        except Exception as e:
            results["tts_error"] = str(e)
    
    return results

# Simple test endpoint for Ollama connection
@app.get("/api/test-ollama")
async def test_ollama():
    """Test Ollama connection and response"""
    try:
        test_response = await llm_service.generate_response("Hello, can you hear me?")
        return {
            "status": "success",
            "response": test_response,
            "model": llm_service.model_name,
            "host": llm_service.host
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "model": llm_service.model_name,
            "host": llm_service.host
        }

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
