import os
import requests
import time
import google.generativeai as genai
import streamlit as st
import re
import pymongo
import gridfs
from datetime import datetime

# -------------------------------
# LOCAL MONGODB CONFIGURATION
# -------------------------------
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "audiobot"

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)  # for storing audio files
records_collection = db["audio_records"]  # for storing metadata (summary, trendy, key moments)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------
# ASSEMBLYAI CONFIGURATION
# -------------------------------
ASSEMBLYAI_API_KEY = "b968b0d0ad9d4c88a87316567c6ca1db"
ASSEMBLYAI_API_URL = "https://api.assemblyai.com/v2"
headers = {
    "authorization": ASSEMBLYAI_API_KEY,
    "content-type": "application/json"
}

def save_to_mongo(file_path, file_name, summary_text):
    """Extract sections and save metadata + audio file to MongoDB."""
    # Extract Trendy Content
    trendy_match = re.search(r'##\s*Trendy Content\s*(.*?)(##\s*Key Moments|$)', summary_text, re.DOTALL)
    trendy_content = trendy_match.group(1).strip() if trendy_match else ""

    # Extract Key Moments
    key_moments_match = re.search(r'##\s*Key Moments\s*(.*)', summary_text, re.DOTALL)
    key_moments = key_moments_match.group(1).strip() if key_moments_match else ""

    # Save file to GridFS
    with open(file_path, "rb") as f:
        file_id = fs.put(f, filename=file_name)

    # Save metadata
    record = {
        "filename": file_name,
        "filepath": file_path,
        "file_id": file_id,  # link to GridFS file
        "summary": summary_text,
        "trendy_content": trendy_content,
        "key_moments": key_moments,
        "timestamp": datetime.now()
    }

    records_collection.insert_one(record)
    return record

def upload_audio(audio_data, filename):
    """Upload audio bytes to AssemblyAI and get public URL."""
    response = requests.post(
        f"{ASSEMBLYAI_API_URL}/upload",
        headers={"authorization": ASSEMBLYAI_API_KEY},
        data=audio_data
    )
    response.raise_for_status()
    return response.json()["upload_url"]

def transcribe_audio_assemblyai(audio_url):
    """Transcribe audio with automatic language detection using AssemblyAI API."""
    payload = {
        "audio_url": audio_url,
        "language_detection": True,
    }
    response = requests.post(
        f"{ASSEMBLYAI_API_URL}/transcript",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    return response.json()["id"]

def get_transcription_result(transcript_id):
    """Retrieve transcription result from AssemblyAI."""
    while True:
        response = requests.get(
            f"{ASSEMBLYAI_API_URL}/transcript/{transcript_id}",
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        if data["status"] == "completed":
            return data
        elif data["status"] == "failed":
            raise Exception("Transcription failed.")
        time.sleep(5)

def diarize_audio_assemblyai(audio_url):
    """Perform speaker diarization using AssemblyAI API."""
    payload = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "language_detection": True
    }
    response = requests.post(
        f"{ASSEMBLYAI_API_URL}/transcript",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    return response.json()["id"]

def get_diarization_result(transcript_id):
    """Retrieve speaker-wise diarization result from AssemblyAI."""
    while True:
        response = requests.get(
            f"{ASSEMBLYAI_API_URL}/transcript/{transcript_id}",
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        if data["status"] == "completed":
            return data.get("utterances", [])
        elif data["status"] == "failed":
            raise Exception("Diarization failed.")
        time.sleep(5)

# NEW HELPER FUNCTION FOR TIMESTAMP CONVERSION
def seconds_to_mmss(seconds):
    """Convert seconds to mm:ss format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

# -------------------------------
# GEMINI CONFIGURATION
# -------------------------------

def summarize_text_gemini(text, podcast_type="General"):
    """Summarize text using Gemini, tailored to podcast type."""
    genai.configure(api_key="AIzaSyCL2IqaNvldvJ-960RM-BrIQEr5npq8dkA")
    model = genai.GenerativeModel("gemini-1.5-flash")

    style_prompts = {
        "General": "Summarize in a clear and concise way for any type of podcast.",
        "News": "Focus on headlines, breaking updates, and key facts. Highlight what’s new and relevant.",
        "Sports": "Emphasize match highlights, scores, player performances, controversies, and crowd reactions.",
        "Comedy": "Extract jokes, punchlines, funny exchanges, and humorous context.",
        "Technology": "Highlight product launches, trends, innovations, technical discussions, and future outlook.",
        "Business": "Focus on financial insights, market trends, strategies, and key announcements.",
        "Education": "Highlight learning points, key concepts, and examples useful for learners.",
        "True Crime": "For Summary: Recap of case details, suspects, and investigations. "
                      "For Trendy Content: Shocking revelations or twists. "
                      "For Key Moments: Evidence discussions or timeline events."
    }

    style = style_prompts.get(podcast_type, style_prompts["General"])

    prompt = (
        f"You are analyzing a {podcast_type} podcast transcript.\n\n"
        f"{style}\n\n"
        "Organize the output into three sections:\n"
        "1. ## Summary: Concise overview.\n"
        "2. ## Trendy Content: Engaging/viral/interesting clips with [mm:ss] timestamps.\n"
        "   - Include exact quoted text from the transcript for each point. "
        "   - Use the exact timestamps provided in the transcript (e.g., if transcript says [00:12 - 00:56], use that directly; do not calculate or change them).\n"
        "3. ## Key Moments: Major highlights with [mm:ss] timestamps.\n\n"
        "   - Include exact quoted text from the transcript for each point. "
        "   - Use the exact timestamps provided in the transcript (e.g., if transcript says [00:12 - 00:56], use that directly; do not calculate or change them).\n\n"
        f"Transcript:\n{text}"
    )

    response = model.generate_content(prompt)
    return response.text

def chat_with_gemini(question, context, podcast_type="General"):
    """Answer questions in style tailored to podcast type."""
    genai.configure(api_key="AIzaSyCL2IqaNvldvJ-960RM-BrIQEr5npq8dkA")
    model = genai.GenerativeModel("gemini-1.5-flash")

    style_prompts = {
        "General": "Answer in a neutral, informative way.",
        "News": "Answer factually, highlighting headlines and recent developments.",
        "Sports": "Answer like a sports commentator, with excitement where relevant.",
        "Comedy": "Answer with humor, light tone, and highlight punchlines.",
        "Technology": "Answer like a tech analyst, focusing on innovations and trends.",
        "Business": "Answer like a market analyst, focusing on strategy and finance.",
        "Education": "Answer like a teacher, breaking down complex ideas clearly."
    }

    style = style_prompts.get(podcast_type, style_prompts["General"])

    prompt = (
        f"You are a helpful chatbot for a {podcast_type} podcast.\n"
        f"{style}\n\n"
        "Always cite timestamps in [mm:ss - mm:ss] format when referencing events.\n"
        "- Use the exact timestamps from the transcript (e.g., if transcript says [00:12 - 00:56], use that directly; do not calculate or change them).\n"
        "- If the question asks 'when' something happened, answer with the event + exact timestamp range.\n"
        "- If the event does not exist, say: 'This was not mentioned in the audio.'\n\n"
        f"**Transcription**:\n{context['transcription']}\n\n"
        f"**Summary**:\n{context['summary']}\n\n"
        f"**User Question**: {question}"
    )

    response = model.generate_content(prompt)
    return response.text

# -------------------------------
# STREAMLIT APP - ENHANCED UI
# -------------------------------
st.set_page_config(
    page_title="AURA VOX - Audio AI",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Updated CSS with new color scheme
st.markdown(
    """
    <style>
    :root {
        --bg: #f8fafc;
        --card: #ffffff;
        --muted: #64748b;
        --accent: #3b82f6;
        --accent-2: #8b5cf6;
        --glass: rgba(255,255,255,0.7);
        --shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .stApp { 
        background: linear-gradient(120deg, #f8fafc, #f1f5f9);
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }

    /* HERO SECTION */
    .hero-box {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: white;
        text-align: center;
        padding: 40px 30px;
        border-radius: 16px;
        box-shadow: var(--shadow);
        margin-bottom: 24px;
        animation: fadeIn 1s ease-in-out;
    }
    .hero-title {
        font-size: 36px;
        font-weight: 800;
        letter-spacing: 1px;
        text-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    .hero-sub {
        font-size: 16px;
        margin-top: 8px;
        opacity: 0.9;
        font-weight: 500;
    }

    /* CARDS */
    .card {
        background: var(--glass);
        border-radius: 14px;
        padding: 20px;
        box-shadow: var(--shadow);
        border: 1px solid rgba(15,23,42,0.05);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }

    /* BUTTONS */
    .stButton>button {
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
        color: white !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        filter: brightness(1.15);
        transform: scale(1.03);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--glass);
        border-radius: 10px;
        padding: 10px 16px;
        font-weight: 600;
        color: var(--muted);
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59,130,246,0.1);
        color: var(--accent);
    }

    /* DIALOGUE */
    .dialogue-box { 
        background: linear-gradient(180deg, #ffffff, #f8fafc); 
        border-radius: 10px; 
        padding: 12px; 
        margin-bottom: 12px; 
        border: 1px solid rgba(15,23,42,0.05); 
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        animation: slideUp 0.4s ease-in-out;
    }
    .speaker-label { 
        font-weight: 600; 
        color: var(--accent); 
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        border-right: none;
        background: linear-gradient(180deg, rgba(255,255,255,0.8), rgba(241,245,249,0.6));
        backdrop-filter: blur(12px);
    }

    /* FILE UPLOADER */
    .stFileUploader>div {
        border-radius: 10px;
        border: 2px dashed var(--accent);
        background: rgba(59,130,246,0.05);
    }

    /* CHAT STYLING */
    .stChatMessage {
        background: var(--glass);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
        border: 1px solid rgba(15,23,42,0.05);
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        animation: slideUp 0.4s ease-in-out;
    }
    .stChatMessage.user {
        background: linear-gradient(180deg, #dbeafe, #bfdbfe);
    }
    .stChatMessage.assistant {
        background: linear-gradient(180deg, #f3e8ff, #ddd6fe);
    }
    .stChatInput>input {
        border-radius: 10px !important;
        border: 2px solid var(--accent) !important;
        padding: 10px !important;
    }

    /* ANIMATIONS */
    @keyframes fadeIn {
        from {opacity:0; transform: translateY(-8px);}
        to {opacity:1; transform: translateY(0);}
    }
    @keyframes slideUp {
        from {opacity:0; transform: translateY(15px);}
        to {opacity:1; transform: translateY(0);}
    }

    /* MOBILE OPTIMIZATION */
    @media (max-width: 768px) {
        .hero-title { font-size: 24px; }
        .hero-sub { font-size: 14px; }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px;
            font-size: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Top header (hero)
with st.container():
    header_col1, header_col2 = st.columns([9,1])
    hero_html = """
    <div class="hero-box">
      <div class="hero-title">✨ AURA VOX</div>
      <div class="hero-sub">🎙️ Audio Transcription, Diarization & AI Insights</div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)
    with header_col2:
        st.markdown("")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("Upload an audio file (mp3, wav, m4a, ogg) and click **Process Audio**.")
    st.divider()
    st.caption("⚠️ Demo keys are embedded in the script. Use Streamlit secrets for production.")
    st.divider()
    st.markdown("### Status")
    status_placeholder = st.empty()
    st.divider()
    st.markdown("### About")
    st.write("Built with AssemblyAI for ASR + Gemini for summarization and chat.")

# Main app area
left_col, right_col = st.columns([1,2], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    podcast_type = st.selectbox(
        "Select Podcast Type",
        ["General", "News", "Sports", "Comedy", "Technology", "Business", "Tech", "Education", "True Crime"]
    )
    uploaded_file = st.file_uploader("Choose an audio file to upload", type=["mp3", "wav", "m4a", "ogg"])
    if uploaded_file is not None:
        st.success(f"File ready: {uploaded_file.name}")
        audio_data = uploaded_file.read()
    else:
        audio_data = None

    if uploaded_file is not None:
        if st.button("Process Audio", key="process_btn"):
            status_placeholder.info("Uploading audio to AssemblyAI...")
            try:
                audio_url = upload_audio(audio_data, uploaded_file.name)
                status_placeholder.success("Audio uploaded successfully.")
            except Exception as e:
                status_placeholder.error(f"Upload failed: {e}")
                st.stop()

            status_placeholder.info("Transcribing audio...")
            try:
                transcript_id = transcribe_audio_assemblyai(audio_url)
                transcript_data = get_transcription_result(transcript_id)
                status_placeholder.success("Transcription completed.")
            except Exception as e:
                status_placeholder.error(f"Transcription failed: {e}")
                st.stop()

            status_placeholder.info("Performing speaker diarization...")
            try:
                diarization_id = diarize_audio_assemblyai(audio_url)
                diarization_data = get_diarization_result(diarization_id)
                status_placeholder.success("Diarization completed.")
            except Exception as e:
                status_placeholder.error(f"Diarization failed: {e}")
                st.stop()

            aligned_dialogue = []
            for segment in diarization_data:
                speaker = segment.get("speaker", "Unknown")
                start_time = segment.get("start", 0) / 1000  # seconds
                end_time = segment.get("end", 0) / 1000      # seconds
                text = segment.get("text", "")
                start_mmss = seconds_to_mmss(start_time)
                end_mmss = seconds_to_mmss(end_time)
                aligned_dialogue.append(f"[{speaker} {start_mmss} - {end_mmss}] {text}")

            status_placeholder.info("Generating summary with Gemini...")
            try:
                dialogue_text = "\n".join(aligned_dialogue)
                summary = summarize_text_gemini(dialogue_text, podcast_type)
                status_placeholder.success("Summary generated.")
            except Exception as e:
                status_placeholder.error(f"Summary generation failed: {e}")
                st.stop()

            st.session_state['aligned_dialogue'] = aligned_dialogue
            st.session_state['summary'] = summary
            # Save uploaded file locally
            local_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(local_path, "wb") as f:
                f.write(audio_data)

            # Save to MongoDB
            try:
                save_to_mongo(local_path, uploaded_file.name, summary)
                status_placeholder.success("✅ Audio + summary saved to MongoDB successfully!")
            except Exception as e:
                status_placeholder.error(f"MongoDB save failed: {e}")
                
    st.markdown('</div>', unsafe_allow_html=True)

# Right column with menu bar
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Menu bar using tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dialogue", "Summary", "Trendy Content", "Key Moments", "Chat"])
    
    # Dialogue tab
    with tab1:
        st.header("🗣️ Speaker-Labeled Dialogue")
        if 'aligned_dialogue' in st.session_state:
            for line in st.session_state['aligned_dialogue']:
                parts = line.split("] ", 1)
                if len(parts) == 2:
                    timestamp_label = parts[0][1:]
                    text = parts[1]
                    st.markdown(f"<div class=\"dialogue-box\"><div class=\"speaker-label\">{timestamp_label}</div><div style=\"margin-top:8px\">{text}</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class=\"dialogue-box\">{line}</div>", unsafe_allow_html=True)
            st.download_button(
                label="Download Dialogue",
                data="\n".join(st.session_state['aligned_dialogue']),
                file_name="aligned_dialogue.txt",
                mime="text/plain"
            )
        else:
            st.info("Process an audio file to view dialogue.")

    # Summary tab
    with tab2:
        st.header("📝 Summary")
        if 'summary' in st.session_state:
            # Extract Summary section
            summary_text = st.session_state['summary']
            summary_section = re.split(r'##\s*Trendy Content', summary_text, 1)[0].strip()
            st.markdown(summary_section)
            st.download_button(
                label="Download Summary",
                data=summary_text,
                file_name="summary.txt",
                mime="text/plain"
            )
        else:
            st.info("Process an audio file to view summary.")

    # Trendy Content tab
    with tab3:
        st.header("🔥 Trendy Content")
        if 'summary' in st.session_state:
            summary_text = st.session_state['summary']
            trendy_match = re.search(r'##\s*Trendy Content\s*(.*?)(##\s*Key Moments|$)', summary_text, re.DOTALL)
            if trendy_match:
                trendy_content = trendy_match.group(1).strip()
                st.markdown(trendy_content)
            else:
                st.info("No trendy content identified.")
        else:
            st.info("Process an audio file to view trendy content.")

    # Key Moments tab
    with tab4:
        st.header("⭐ Key Moments")
        if 'summary' in st.session_state:
            summary_text = st.session_state['summary']
            key_moments_match = re.search(r'##\s*Key Moments\s*(.*)', summary_text, re.DOTALL)
            if key_moments_match:
                key_moments = key_moments_match.group(1).strip()
                st.markdown(key_moments)
            else:
                st.info("No key moments identified.")
        else:
            st.info("Process an audio file to view key moments.")

    # Chat tab
    with tab5:
        st.header("💬 Chat About the Audio")
        if 'aligned_dialogue' in st.session_state and 'summary' in st.session_state:
            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []

            # Chat input at the top
            user_input = st.chat_input("Ask a question about the audio content...")
            if user_input:
                # Prepare context for Gemini
                context = {
                    "transcription": "\n".join(st.session_state['aligned_dialogue']),
                    "summary": st.session_state['summary']
                }
                
                # Get response from Gemini
                try:
                    status_placeholder.info("Generating chat response...")
                    response = chat_with_gemini(user_input, context, podcast_type)
                    # Store question and answer as a pair
                    st.session_state['chat_history'].append({
                        "question": {"role": "user", "content": user_input},
                        "answer": {"role": "assistant", "content": response}
                    })
                    status_placeholder.success("Chat response generated.")
                except Exception as e:
                    status_placeholder.error(f"Chat response failed: {e}")

                # Rerun to update chat display
                st.rerun()

            # Clear chat history button
            if st.button("Clear Chat History", key="clear_chat_btn"):
                st.session_state['chat_history'] = []
                st.rerun()

            # Display chat history in reverse order (newest first), question before answer
            for pair in reversed(st.session_state['chat_history']):
                # Display question
                with st.chat_message(pair["question"]["role"]):
                    st.markdown(pair["question"]["content"])
                # Display answer
                with st.chat_message(pair["answer"]["role"]):
                    st.markdown(pair["answer"]["content"])
        else:
            st.info("Process an audio file to enable the chat feature.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("Made with ❤️ — AssemblyAI + Gemini. Keep API keys private in production.")