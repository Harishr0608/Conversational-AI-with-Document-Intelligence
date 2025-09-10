import streamlit as st
import requests
import json
import os
import io
from typing import List, Dict, Any
import time
import base64
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np
import asyncio
import httpx

# Page configuration
st.set_page_config(
    page_title="Voice-Enabled RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BACKEND_URL = "http://localhost:8000"
AUDIO_SAMPLE_RATE = 16000

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = True
if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = True
if "recording" not in st.session_state:
    st.session_state.recording = False

# Utility functions
def make_api_request(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None) -> Dict:
    """Make API request to backend."""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, data=data)
            else:
                response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return {"error": str(e)}

def upload_document(file) -> bool:
    """Upload document to backend."""
    files = {"file": (file.name, file.getvalue(), file.type)}
    result = make_api_request("/api/documents/upload", "POST", files=files)
    return "error" not in result

def get_documents() -> List[str]:
    """Get list of uploaded documents."""
    result = make_api_request("/api/documents/list")
    if "error" not in result:
        return result.get("documents", [])
    return []

def delete_document(filename: str) -> bool:
    """Delete a document."""
    result = make_api_request(f"/api/documents/{filename}", "DELETE")
    return "error" not in result

def send_chat_message(message: str, use_rag: bool, use_web_search: bool) -> Dict:
    """Send chat message to backend."""
    data = {
        "message": message,
        "use_rag": use_rag,
        "use_web_search": use_web_search,
        "temperature": st.session_state.get("temperature", 0.7),
        "max_tokens": st.session_state.get("max_tokens", 2000)
    }
    return make_api_request("/api/chat", "POST", data)

def text_to_speech(text: str) -> str:
    """Convert text to speech and return audio file path."""
    data = {"text": text, "save_file": True}
    result = make_api_request("/api/tts/speak", "POST", data)
    if "error" not in result:
        return result.get("audio_file")
    return None

def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio bytes to text."""
    files = {"audio_file": ("audio.wav", audio_bytes, "audio/wav")}
    result = make_api_request("/api/speech/transcribe", "POST", files=files)
    if "error" not in result:
        return result.get("transcription", "")
    return ""

# Audio recording function
def audio_recorder():
    """Audio recording component."""
    def audio_frame_callback(frame):
        audio_array = frame.to_ndarray()
        return av.AudioFrame.from_ndarray(audio_array, format="s16", layout="mono")
    
    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_frame_callback=audio_frame_callback,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        async_processing=True,
    )
    
    return webrtc_ctx

# Sidebar
def render_sidebar():
    """Render sidebar with settings and document management."""
    st.sidebar.title("⚙️ Settings")
    
    # Chat settings
    st.sidebar.subheader("Chat Settings")
    st.session_state.rag_enabled = st.sidebar.checkbox("Enable RAG", value=st.session_state.rag_enabled)
    st.session_state.web_search_enabled = st.sidebar.checkbox("Enable Web Search", value=st.session_state.web_search_enabled)
    st.session_state.tts_enabled = st.sidebar.checkbox("Enable Text-to-Speech", value=st.session_state.tts_enabled)
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        st.session_state.max_tokens = st.slider("Max Tokens", 100, 4000, 2000, 100)
    
    # Document management
    st.sidebar.subheader("📚 Document Management")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Document",
        type=["pdf", "txt", "docx", "csv", "xlsx"],
        help="Upload documents for RAG context"
    )
    
    if uploaded_file is not None:
        if st.sidebar.button("Upload Document"):
            with st.spinner("Uploading and processing..."):
                if upload_document(uploaded_file):
                    st.sidebar.success(f"✅ {uploaded_file.name} uploaded successfully!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Upload failed")
    
    # Document list
    documents = get_documents()
    if documents:
        st.sidebar.subheader("📄 Uploaded Documents")
        for doc in documents:
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(f"📄 {doc}")
            if col2.button("🗑️", key=f"del_{doc}", help="Delete document"):
                if delete_document(doc):
                    st.sidebar.success(f"Deleted {doc}")
                    st.rerun()
                else:
                    st.sidebar.error("Delete failed")
    
    # Clear all documents
    if documents and st.sidebar.button("🗑️ Clear All Documents", type="secondary"):
        if st.sidebar.confirm("Are you sure you want to clear all documents?"):
            make_api_request("/api/rag/clear", "DELETE")
            st.sidebar.success("All documents cleared!")
            st.rerun()
    
    # System status
    st.sidebar.subheader("📊 System Status")
    try:
        health = make_api_request("/api/health")
        if "error" not in health:
            st.sidebar.success("🟢 Backend Online")
            if health.get("services", {}).get("llm"):
                st.sidebar.success("🟢 LLM Available")
            else:
                st.sidebar.warning("🟡 LLM Loading...")
            
            rag_stats = health.get("services", {}).get("rag", {})
            doc_count = rag_stats.get("document_count", 0)
            st.sidebar.info(f"📚 Documents: {doc_count}")
        else:
            st.sidebar.error("🔴 Backend Offline")
    except:
        st.sidebar.error("🔴 Backend Offline")

# Main chat interface
def render_chat_interface():
    """Render main chat interface."""
    st.title("🤖 Voice-Enabled RAG Assistant")
    st.markdown("Ask questions using text or voice input. I can search your documents and the web!")
    
    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                sources = message["sources"]
                if sources:
                    with st.expander("📚 Sources Used"):
                        for source in sources:
                            st.write(f"• {source}")
            
            # Play audio if available
            if message["role"] == "assistant" and "audio_file" in message and message["audio_file"]:
                try:
                    audio_url = f"{BACKEND_URL}/api/tts/audio/{os.path.basename(message['audio_file'])}"
                    st.audio(audio_url)
                except:
                    pass
    
    # Chat input methods
    tab1, tab2 = st.tabs(["💬 Text Input", "🎤 Voice Input"])
    
    with tab1:
        # Text input
        if prompt := st.chat_input("Type your message here..."):
            process_message(prompt)
    
    with tab2:
        # Voice input section
        st.subheader("🎤 Voice Input")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # File upload for audio
            audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a", "ogg"])
            if audio_file and st.button("🔄 Transcribe Audio"):
                with st.spinner("Transcribing audio..."):
                    transcription = transcribe_audio(audio_file.getvalue())
                    if transcription:
                        st.success(f"Transcription: {transcription}")
                        process_message(transcription)
                    else:
                        st.error("Could not transcribe audio")
        
        with col2:
            # WebRTC audio recorder
            st.write("Real-time Recording:")
            webrtc_ctx = audio_recorder()
            
            if webrtc_ctx.state.playing:
                st.info("🔴 Recording... Click stop when done")
            
        with col3:
            # Manual recording controls
            st.write("Manual Controls:")
            if st.button("🎤 Start Recording"):
                st.session_state.recording = True
                st.info("Recording started... (This is a demo - implement actual recording)")
            
            if st.button("⏹️ Stop & Transcribe") and st.session_state.recording:
                st.session_state.recording = False
                st.info("Recording stopped. In a full implementation, this would transcribe the audio.")

def process_message(message: str):
    """Process user message and get response."""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": message})
    
    with st.chat_message("user"):
        st.markdown(message)
    
    # Get response from backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_data = send_chat_message(
                message,
                st.session_state.rag_enabled,
                st.session_state.web_search_enabled
            )
        
        if "error" not in response_data:
            response_text = response_data.get("response", "Sorry, I couldn't generate a response.")
            sources = response_data.get("sources", [])
            context_used = response_data.get("context_used", 0)
            
            # Display response
            st.markdown(response_text)
            
            # Show context info
            if context_used > 0:
                st.info(f"📊 Used {context_used} context sources")
            
            # Show sources
            if sources:
                with st.expander("📚 Sources Used"):
                    for source in sources:
                        st.write(f"• {source}")
            
            # Generate TTS if enabled
            audio_file = None
            if st.session_state.tts_enabled and response_text:
                with st.spinner("Generating speech..."):
                    audio_file = text_to_speech(response_text)
                    if audio_file:
                        try:
                            audio_url = f"{BACKEND_URL}/api/tts/audio/{os.path.basename(audio_file)}"
                            st.audio(audio_url)
                        except:
                            pass
            
            # Add assistant message to history
            assistant_message = {
                "role": "assistant",
                "content": response_text,
                "sources": sources,
                "context_used": context_used
            }
            if audio_file:
                assistant_message["audio_file"] = audio_file
            
            st.session_state.messages.append(assistant_message)
            
        else:
            error_msg = "Sorry, I encountered an error. Please try again."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Additional features
def render_additional_features():
    """Render additional features in expandable sections."""
    
    with st.expander("🔍 Quick Search"):
        search_query = st.text_input("Search your documents:")
        if st.button("Search") and search_query:
            with st.spinner("Searching..."):
                search_data = {"query": search_query, "max_results": 5}
                results = make_api_request("/api/rag/search", "POST", search_data)
                
                if "error" not in results and results.get("results"):
                    st.subheader("Search Results:")
                    for i, result in enumerate(results["results"]):
                        st.write(f"**{i+1}.** {result.get('content', '')[:200]}...")
                        st.write(f"*Source: {result.get('metadata', {}).get('filename', 'Unknown')}*")
                        st.divider()
                else:
                    st.info("No results found.")
    
    with st.expander("🌐 Web Search"):
        web_query = st.text_input("Search the web:")
        search_type = st.selectbox("Search Type:", ["web", "news", "academic"])
        
        if st.button("Search Web") and web_query:
            with st.spinner("Searching web..."):
                search_data = {
                    "query": web_query,
                    "max_results": 5,
                    "search_type": search_type
                }
                results = make_api_request("/api/search", "POST", search_data)
                
                if "error" not in results and results.get("results"):
                    st.subheader("Web Search Results:")
                    for i, result in enumerate(results["results"]):
                        st.write(f"**{i+1}. {result.get('title', 'No Title')}**")
                        st.write(result.get('snippet', result.get('summary', ''))[:300] + "...")
                        if result.get('url'):
                            st.write(f"🔗 [Link]({result['url']})")
                        st.divider()
                else:
                    st.info("No web results found.")
    
    with st.expander("📝 Text Summarization"):
        text_to_summarize = st.text_area("Text to Summarize:", height=150)
        max_length = st.slider("Summary Length:", 50, 500, 200)
        
        if st.button("Summarize") and text_to_summarize:
            with st.spinner("Summarizing..."):
                summary_data = {"text": text_to_summarize, "max_length": max_length}
                # Note: This would need to be implemented as a proper API call with form data
                # For now, we'll simulate it
                st.info("Summary feature would be implemented here with proper form handling.")

# Main app
def main():
    """Main application function."""
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat interface
        render_chat_interface()
    
    with col2:
        # Additional features
        render_additional_features()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.8em;'>
            Voice-Enabled RAG Assistant v1.0<br>
            Powered by Ollama, ChromaDB, and Whisper
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()