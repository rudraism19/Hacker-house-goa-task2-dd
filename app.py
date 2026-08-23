"""
Streamlit Web Application for Voice-Enabled RAG System
Pixel-Perfect UI matching Hacker House Goa 2026 Reference Design (Task #2: Voice-Enabled RAG Model).
Design System: Emerald Green Beach Backdrop, Dark Forest Cards, Sunset Gold (#FDB827) Highlights, Sub-200ms SLA.
"""

import streamlit as st
import time
import os
import base64
import urllib.parse
import pandas as pd
from dataset_loader import MSMARCOXIBackendLoader
from chunking_engine import MultiStrategyChunkingEngine
from vector_store import VectorStore
from stt_engine import SpeechToTextEngine
from model_harness import ModelHarnessOrchestrator, VoiceRAGRequest
from latency_analytics import LatencyAnalyticsEngine
import config

DEFAULT_STT_LANG = getattr(config, "DEFAULT_STT_LANG", "en-IN")

# -------------------------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="HACKER गोवा HOUSE // Voice-Enabled RAG",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State for queries
if "active_query" not in st.session_state:
    st.session_state.active_query = ""
if "voice_mode" not in st.session_state:
    st.session_state.voice_mode = "Sentence-Aware (Semantic)"
if "lang_selection" not in st.session_state:
    st.session_state.lang_selection = "English (en-IN)"
if "show_recorder" not in st.session_state:
    st.session_state.show_recorder = False

# Load background image as base64 if available
bg_image_path = os.path.join(os.path.dirname(__file__), "assets", "hh_goa_bg.png")
bg_base64_css = ""
if os.path.exists(bg_image_path):
    with open(bg_image_path, "rb") as img_file:
        encoded_bg = base64.b64encode(img_file.read()).decode()
        bg_base64_css = f"""
        .stApp {{
            background: linear-gradient(rgba(6, 44, 25, 0.88), rgba(4, 28, 16, 0.94)),
                        url("data:image/png;base64,{encoded_bg}") no-repeat center top fixed;
            background-size: cover;
        }}
        """

# -------------------------------------------------------------------
# Hacker House Goa 2026 Reference CSS Styling
# -------------------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=JetBrains+Mono:ital,wght@0,300;0,400;0,600;0,700;1,400&family=Space+Grotesk:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    :root {{
        --hh-emerald: #083c22;
        --hh-dark-card: #0b1f1a;
        --hh-card-surface: #0e241f;
        --hh-gold: #FDB827;
        --hh-gold-hover: #FFAA00;
        --hh-cyan: #00E5FF;
        --hh-green: #00FF88;
        --hh-text-light: #F8F9FA;
        --hh-text-muted: #9CA3AF;
        --hh-border: rgba(255, 255, 255, 0.08);
    }}
    
    html, body, [class*="css"] {{
        font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
    }}
    
    code, pre, .mono-text {{
        font-family: 'JetBrains Mono', monospace !important;
    }}
    
    [data-testid="stSidebar"] {{ display: none; }}
    .stAppHeader {{ background-color: transparent; }}
    
    {bg_base64_css if bg_base64_css else """
    .stApp {
        background-color: #062b19;
        background: radial-gradient(circle at 50% 0%, #0d4e2d 0%, #062616 70%);
        color: #F8F9FA;
    }
    """}
    
    /* Top Header Bar */
    .hh-nav-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 25px;
        margin-bottom: 25px;
    }}
    
    .hh-logo {{
        font-family: 'Cinzel', serif;
        font-size: 1.85rem;
        font-weight: 900;
        color: #FDB827;
        letter-spacing: 2px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }}
    
    .hh-devanagari-badge {{
        background: #E53E3E;
        color: #FFFFFF;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.62rem;
        font-weight: 800;
        padding: 2px 6px;
        border-radius: 4px;
        letter-spacing: 0px;
        vertical-align: middle;
    }}
    
    .hh-voice-heard-btn {{
        background: #FDB827;
        color: #000000;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-size: 0.82rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 10px 22px;
        border-radius: 6px;
        border: 1px solid #E5A51F;
        box-shadow: 0 4px 15px rgba(253, 184, 39, 0.3);
        display: inline-flex;
        align-items: center;
        text-decoration: none;
        background-image: repeating-linear-gradient(-45deg, transparent, transparent 4px, rgba(0,0,0,0.08) 4px, rgba(0,0,0,0.08) 8px);
        transition: all 0.2s ease;
    }}
    .hh-voice-heard-btn:hover {{
        background-color: #FFAA00;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(253, 184, 39, 0.5);
        color: #000;
    }}
    
    /* Dual Hero Cards */
    .hh-hero-card {{
        background: #0b1f1a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 38px 34px;
        height: 100%;
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.45);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    
    .hh-pill-tag {{
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #FDB827;
        border: 1px solid rgba(253, 184, 39, 0.4);
        padding: 5px 14px;
        border-radius: 20px;
        background: rgba(253, 184, 39, 0.06);
        margin-bottom: 24px;
    }}
    
    .hero-main-title {{
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
        color: #FFFFFF;
        margin-bottom: 20px;
        letter-spacing: -0.5px;
    }}
    
    .highlight-gold {{
        color: #FDB827;
    }}
    
    .hero-body-text {{
        color: #9CA3AF;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 30px;
    }}
    
    .hero-features-row {{
        display: flex;
        justify-content: space-between;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 22px;
        margin-top: auto;
    }}
    
    .feature-title {{
        font-weight: 700;
        font-size: 0.95rem;
        color: #FFFFFF;
        margin-bottom: 3px;
    }}
    
    .feature-sub {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #6B7280;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    
    /* Live Studio Right Card */
    .hh-studio-card {{
        background: #0b1f1a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 34px;
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.45);
    }}
    
    .studio-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 25px;
    }}
    
    .studio-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        color: #9CA3AF;
        letter-spacing: 2px;
    }}
    
    .query-ready-badge {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: #00E5FF;
        background: rgba(0, 229, 255, 0.08);
        border: 1px solid rgba(0, 229, 255, 0.35);
        padding: 4px 12px;
        border-radius: 15px;
        letter-spacing: 1px;
    }}
    
    /* Divider OR TYPE */
    .or-type-divider {{
        display: flex;
        align-items: center;
        text-align: center;
        color: #6B7280;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 2px;
        margin: 22px 0;
    }}
    .or-type-divider::before, .or-type-divider::after {{
        content: '';
        flex: 1;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .or-type-divider:not(:empty)::before {{
        margin-right: 15px;
    }}
    .or-type-divider:not(:empty)::after {{
        margin-left: 15px;
    }}
    
    /* Telemetry Metric Chip */
    .hh-metric-chip {{
        background: rgba(9, 26, 21, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
        transition: all 0.3s ease;
    }}
    .hh-metric-chip:hover {{
        border-color: rgba(253, 184, 39, 0.4);
        box-shadow: 0 4px 20px rgba(253, 184, 39, 0.15);
    }}
    .hh-metric-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.35rem;
        font-weight: 700;
        color: #FDB827;
    }}
    .hh-metric-lbl {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }}
    
    /* Chat Results */
    .chat-bubble-user {{
        background: rgba(253, 184, 39, 0.1);
        border: 1px solid rgba(253, 184, 39, 0.35);
        border-radius: 14px 14px 2px 14px;
        padding: 16px 20px;
        margin-bottom: 15px;
        color: #FFF3EB;
        font-size: 1rem;
    }}
    
    .chat-bubble-ai {{
        background: #0e2620;
        border: 1px solid rgba(0, 229, 255, 0.35);
        border-radius: 14px 14px 14px 2px;
        padding: 20px 24px;
        margin-bottom: 15px;
        color: #F8F9FA;
        box-shadow: 0 6px 30px rgba(0, 0, 0, 0.4);
    }}
    
    /* Footer */
    .hh-footer {{
        text-align: center;
        padding: 30px 10px 10px 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin-top: 40px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #6B7280;
    }}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Global Pipeline Initialization
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_rag_system():
    loader_en = MSMARCOXIBackendLoader(lang="en", max_samples=300)
    loader_hi = MSMARCOXIBackendLoader(lang="hi", max_samples=300)
    dataset = loader_en.load_dataset() + loader_hi.load_dataset()

    chunk_engine = MultiStrategyChunkingEngine(strategy_name="semantic_boundary")
    chunks = chunk_engine.chunk_documents(dataset)

    vector_store = VectorStore()
    vector_store.build_index(chunks)

    stt_engine = SpeechToTextEngine(provider="groq")
    orchestrator = ModelHarnessOrchestrator(stt_engine, vector_store, chunk_engine)
    
    return dataset, chunk_engine, vector_store, stt_engine, orchestrator

dataset, chunk_engine, vector_store, stt_engine, orchestrator = init_rag_system()

# -------------------------------------------------------------------
# Top Brand Bar
# -------------------------------------------------------------------
st.markdown("""
<div class="hh-nav-container">
    <div class="hh-logo">
        HACKER <span class="hh-devanagari-badge">गोवा</span> HOUSE
    </div>
    <div>
        <a href="#live-studio" class="hh-voice-heard-btn">GET YOUR VOICE HEARD</a>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Two-Column Hero Grid
# -------------------------------------------------------------------
col_hero_left, col_hero_right = st.columns([1.1, 1.3], gap="large")

# LEFT CARD: "Ask in your voice"
with col_hero_left:
    st.markdown("""
    <div class="hh-hero-card">
        <div>
            <span class="hh-pill-tag">VOICE-ENABLED RAG</span>
            <div class="hero-main-title">
                Ask in your<br/>
                voice. Get<br/>
                <span class="highlight-gold">grounded</span><br/>
                answers.
            </div>
            <div class="hero-body-text">
                Real-time voice capture, multi-strategy document chunking, SIMD float32 vector retrieval, and ultra-low latency grounded LLM synthesis.
            </div>
        </div>
        <div class="hero-features-row">
            <div>
                <div class="feature-title">Voice-first</div>
                <div class="feature-sub">HANDS-FREE INPUT</div>
            </div>
            <div>
                <div class="feature-title">Grounded</div>
                <div class="feature-sub">EVIDENCE-BACKED</div>
            </div>
            <div>
                <div class="feature-title">Fast</div>
                <div class="feature-sub">&lt;200MS SLA</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# RIGHT CARD: "LIVE STUDIO"
with col_hero_right:
    st.markdown('<div class="hh-studio-card" id="live-studio">', unsafe_allow_html=True)
    st.markdown("""
    <div class="studio-header">
        <span class="studio-label">LIVE STUDIO</span>
        <span class="query-ready-badge">QUERY READY</span>
    </div>
    """, unsafe_allow_html=True)

    # Top row controls
    col_c1, col_c2 = st.columns([1, 1.2])
    with col_c1:
        st.write("")
        if st.button("🎙️ SPEAK NOW", use_container_width=True):
            st.session_state.show_recorder = not st.session_state.show_recorder

    with col_c2:
        voice_mode = st.selectbox(
            "VOICE MODE",
            [
                "Sentence-Aware (Semantic)",
                "Fixed-Size Overlap",
                "Hierarchical (Parent-Child)",
                "Metadata-Aware Window"
            ],
            index=0,
            key="voice_mode_select"
        )
        strategy_map = {
            "Sentence-Aware (Semantic)": "semantic_boundary",
            "Fixed-Size Overlap": "fixed_overlap",
            "Hierarchical (Parent-Child)": "hierarchical",
            "Metadata-Aware Window": "metadata_aware"
        }
        active_strat = strategy_map.get(voice_mode, "semantic_boundary")
        chunk_engine.set_strategy(active_strat)

    # Audio recorder trigger
    recorded_audio = None
    if st.session_state.show_recorder or hasattr(st, "audio_input"):
        st.markdown("<br/>", unsafe_allow_html=True)
        if hasattr(st, "audio_input"):
            recorded_audio = st.audio_input("Record voice question:")
        else:
            recorded_audio = st.file_uploader("Upload voice audio (.wav / .mp3)", type=["wav", "mp3"])

    # Divider OR TYPE
    st.markdown('<div class="or-type-divider">OR TYPE</div>', unsafe_allow_html=True)

    # Text Input + Send Button
    col_inp, col_snd = st.columns([3.5, 1])
    with col_inp:
        typed_query = st.text_input(
            "Query Input",
            value=st.session_state.active_query,
            placeholder="Type query here...",
            label_visibility="collapsed"
        )
    with col_snd:
        send_clicked = st.button("Send", use_container_width=True)

    # Preset Prompt Chips
    st.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
    chip_cols = st.columns(4)
    presets = [
        "What is a corporation?",
        "कॉर्पोरेशन क्या है?",
        "कैश फ्लो स्टेटमेंट क्या है?",
        "What are CSE subjects?"
    ]
    for i, p in enumerate(presets):
        with chip_cols[i % 4]:
            if st.button(p, key=f"chip_{i}", use_container_width=True):
                st.session_state.active_query = p
                typed_query = p
                send_clicked = True
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# Pipeline Execution & Conversational Response
# -------------------------------------------------------------------
query_to_run = typed_query.strip() if send_clicked and typed_query else None
has_audio = recorded_audio is not None

if query_to_run or has_audio:
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="hh-studio-card">', unsafe_allow_html=True)
    
    with st.spinner("⚡ Executing Voice RAG Pipeline (<200ms Target)..."):
        if has_audio:
            audio_bytes = recorded_audio.read()
            filename = getattr(recorded_audio, "name", "voice_input.wav")
            req = VoiceRAGRequest(
                audio_bytes=audio_bytes,
                audio_filename=filename,
                language_code="en-IN",
                chunking_strategy=active_strat,
                stt_provider="local",
                synthesizer_mode="auto"
            )
        else:
            is_hindi = any('\u0900' <= char <= '\u097F' for char in query_to_run)
            lang_code = "hi-IN" if is_hindi else "en-IN"
            req = VoiceRAGRequest(
                prompt_text=query_to_run,
                language_code=lang_code,
                chunking_strategy=active_strat,
                stt_provider="local",
                synthesizer_mode="auto"
            )

        start_t = time.time()
        response = orchestrator.run_pipeline(req)
        total_time_ms = (time.time() - start_t) * 1000

    # User Query Bubble
    st.markdown(f"""
    <div class="chat-bubble-user">
        <span class="hh-pill-tag" style="margin-bottom:6px; padding: 2px 10px; font-size: 0.7rem;">👤 INPUT QUERY</span><br/>
        <strong>"{response.transcript}"</strong>
    </div>
    """, unsafe_allow_html=True)

    # AI Response Bubble
    if response.is_refused:
        st.markdown(f"""
        <div class="chat-bubble-ai" style="border-color: rgba(239, 68, 68, 0.6); background: rgba(35, 15, 15, 0.95);">
            <span class="hh-pill-tag" style="background: rgba(239, 68, 68, 0.2); color: #EF4444; border-color: rgba(239, 68, 68, 0.4); margin-bottom: 6px; padding: 2px 10px;">🛡️ GUARDRAIL SAFE REFUSAL</span><br/>
            <strong style="color: #F87171;">Reason:</strong> {response.refusal_reason}<br/><br/>
            {response.answer}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-bubble-ai">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: #00FF88; background: rgba(0, 255, 136, 0.1); padding: 3px 10px; border-radius: 6px; border: 1px solid rgba(0, 255, 136, 0.3);">🤖 GROUNDED ANSWER</span><br/>
            <div style="font-size: 1.05rem; line-height: 1.6; margin-top: 10px; color: #F8F9FA;">{response.answer}</div>
        </div>
        """, unsafe_allow_html=True)

        if response.citations:
            st.markdown("**📌 Evidence & Document Citations:**")
            for c in response.citations:
                st.markdown(f"- 📄 `{c}`")

        # Auto-speech narration
        clean_speech = response.answer.replace('"', '\\"').replace('\n', ' ').replace("'", "\\'")
        st.components.v1.html(f"""
        <div style="margin-top: 5px;">
            <button onclick="
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel();
                    function getLang(t) {{
                        if (/[\u0980-\u09FF]/.test(t)) return 'bn-IN';
                        if (/[\u0B80-\u0BFF]/.test(t)) return 'ta-IN';
                        if (/[\u0C00-\u0C7F]/.test(t)) return 'te-IN';
                        if (/[\u0C80-\u0CFF]/.test(t)) return 'kn-IN';
                        if (/[\u0D00-\u0D7F]/.test(t)) return 'ml-IN';
                        if (/[\u0A80-\u0AFF]/.test(t)) return 'gu-IN';
                        if (/[\u0A00-\u0A7F]/.test(t)) return 'pa-IN';
                        if (/[\u0B00-\u0B7F]/.test(t)) return 'or-IN';
                        if (/[\u0600-\u06FF]/.test(t)) return 'ur-IN';
                        if (/[\u0900-\u097F]/.test(t)) return 'hi-IN';
                        return 'en-IN';
                    }}
                    const u = new SpeechSynthesisUtterance('{clean_speech}');
                    u.lang = getLang('{clean_speech}');
                    window.speechSynthesis.speak(u);
                }}
            " style="background: rgba(253, 184, 39, 0.15); border: 1px solid rgba(253, 184, 39, 0.4); color: #FDB827; padding: 4px 14px; border-radius: 20px; font-family: 'sans-serif'; font-weight: 700; font-size: 0.78rem; cursor: pointer;">
                🔊 Replay Voice Narration
            </button>
        </div>
        <script>
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                function getLang(t) {{
                    if (/[\u0980-\u09FF]/.test(t)) return 'bn-IN';
                    if (/[\u0B80-\u0BFF]/.test(t)) return 'ta-IN';
                    if (/[\u0C00-\u0C7F]/.test(t)) return 'te-IN';
                    if (/[\u0C80-\u0CFF]/.test(t)) return 'kn-IN';
                    if (/[\u0D00-\u0D7F]/.test(t)) return 'ml-IN';
                    if (/[\u0A80-\u0AFF]/.test(t)) return 'gu-IN';
                    if (/[\u0A00-\u0A7F]/.test(t)) return 'pa-IN';
                    if (/[\u0B00-\u0B7F]/.test(t)) return 'or-IN';
                    if (/[\u0600-\u06FF]/.test(t)) return 'ur-IN';
                    if (/[\u0900-\u097F]/.test(t)) return 'hi-IN';
                    return 'en-IN';
                }}
                const utter = new SpeechSynthesisUtterance('{clean_speech}');
                utter.lang = getLang('{clean_speech}');
                window.speechSynthesis.speak(utter);
            }}
        </script>
        """, height=40)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 20px 0;'>", unsafe_allow_html=True)

    # Telemetry Grid
    st.markdown("#### ⚡ Real-Time Pipeline Telemetry")
    m1, m2, m3, m4 = st.columns(4)

    sla_status = "⚡ PASSED (<200ms)" if response.met_sla_200ms else "⚠️ EXCEEDED (>200ms)"
    with m1:
        st.markdown(f"""
        <div class="hh-metric-chip">
            <div class="hh-metric-val">{response.total_latency_ms} ms</div>
            <div class="hh-metric-lbl">Total Latency ({sla_status})</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="hh-metric-chip">
            <div class="hh-metric-val">{int(response.grounding_score * 100)}%</div>
            <div class="hh-metric-lbl">Grounding Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="hh-metric-chip">
            <div class="hh-metric-val">{int(response.hallucination_risk * 100)}%</div>
            <div class="hh-metric-lbl">Hallucination Risk</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        synth_label = response.synthesizer
        st.markdown(f"""
        <div class="hh-metric-chip">
            <div class="hh-metric-val" style="font-size: 0.95rem; color: #00FF88;">{synth_label}</div>
            <div class="hh-metric-lbl">Active Synthesizer</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("##### ⏱️ Stage Breakdown (ms)")
    df_stages = pd.DataFrame([
        {"Pipeline Stage": k, "Latency (ms)": f"{v:.2f} ms"} for k, v in response.stage_latencies_ms.items()
    ])
    st.dataframe(df_stages, use_container_width=True, hide_index=True)

    # Social Share Card
    st.markdown("<br/>", unsafe_allow_html=True)
    share_text = urllib.parse.quote(
        f"Just benchmarked our sub-200ms Voice-Enabled RAG pipeline for @hhgoa 2026 (Task #2)! "
        f"⚡ Latency: {response.total_latency_ms}ms | Grounding: {int(response.grounding_score * 100)}% #RAGInGoa #HHGoa2026"
    )
    x_intent_url = f"https://twitter.com/intent/tweet?text={share_text}"
    st.markdown(f"""
    <div style="background: rgba(253, 184, 39, 0.08); border: 1px dashed rgba(253, 184, 39, 0.3); border-radius: 12px; padding: 15px; text-align: center;">
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #FDB827;">
            ✦ <strong>Task #2 Verification:</strong> Ready to post on X with <strong>#RAGInGoa</strong>
        </span><br/><br/>
        <a href="{x_intent_url}" target="_blank" style="display: inline-block; background: #000; color: #FFF; border: 1px solid #FDB827; padding: 8px 18px; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; text-decoration: none; font-weight: 600;">
            🐦 Share Result on X (#RAGInGoa) ↗
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# Benchmark, Chunking, Telemetry & Guardrail Tabs
# -------------------------------------------------------------------
st.markdown("<br/><br/>", unsafe_allow_html=True)
tabs = st.tabs([
    "[01 // ✂️ CHUNKING STRATEGIES LAB]",
    "[02 // 📊 P50/P70/P100 LATENCY BENCHMARK]",
    "[03 // 🛡️ HARNESS & GUARDRAILS]"
])

# TAB 1: Chunking Strategies Evaluation
with tabs[0]:
    st.markdown('<div class="hh-studio-card">', unsafe_allow_html=True)
    st.markdown("### ✂️ Multi-Strategy Engineered Chunking Benchmark")
    st.markdown("""
    Compare performance, chunk granularity, and execution throughput across all 4 chunking strategies evaluated on the **AI4Bharat MSMARCO-XI** corpus:
    """)

    c_s1, c_s2, c_s3, c_s4 = st.columns(4)
    with c_s1:
        st.markdown("""
        <div class="hh-metric-chip">
            <div style="color: #FDB827; font-weight:700;">1. Fixed Overlap</div>
            <div style="font-size:0.75rem; color:#9CA3AF; margin-top:4px;">Window sliding with boundary context preservation.</div>
        </div>
        """, unsafe_allow_html=True)
    with c_s2:
        st.markdown("""
        <div class="hh-metric-chip">
            <div style="color: #00FF88; font-weight:700;">2. Semantic Boundary</div>
            <div style="font-size:0.75rem; color:#9CA3AF; margin-top:4px;">Sentence and punctuation aware splitting.</div>
        </div>
        """, unsafe_allow_html=True)
    with c_s3:
        st.markdown("""
        <div class="hh-metric-chip">
            <div style="color: #00E5FF; font-weight:700;">3. Hierarchical</div>
            <div style="font-size:0.75rem; color:#9CA3AF; margin-top:4px;">Parent-child context hierarchy indexing.</div>
        </div>
        """, unsafe_allow_html=True)
    with c_s4:
        st.markdown("""
        <div class="hh-metric-chip">
            <div style="color: #FFAA00; font-weight:700;">4. Metadata-Aware</div>
            <div style="font-size:0.75rem; color:#9CA3AF; margin-top:4px;">Language & passage position payload embedding.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("⚡ EXECUTE CHUNKING COMPARISON BENCHMARK"):
        with st.spinner("Benchmarking 4 chunking algorithms across corpus..."):
            comparison_results = chunk_engine.compare_strategies(dataset[:40])
            df_comp = pd.DataFrame.from_dict(comparison_results, orient="index")
            df_comp.index.name = "Strategy"
            st.table(df_comp)
            st.success("✅ Multi-strategy chunking benchmark completed.")

    st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: Latency Percentile Analytics Suite
with tabs[1]:
    st.markdown('<div class="hh-studio-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Latency Percentiles Telemetry (P50 / P70 / P100)")
    st.markdown("""
    Execute the automated micro-benchmark harness across real queries to evaluate statistical tail latencies and SLA compliance.
    """)

    num_samples = st.slider("Select Benchmark Query Sample Size:", min_value=20, max_value=100, value=50, step=10)

    if st.button("🚀 RUN QUERY LATENCY BENCHMARK SUITE"):
        with st.spinner(f"Executing {num_samples} real queries through full RAG harness..."):
            analytics_data = run_benchmark_suite(num_samples=num_samples, strategy="semantic_boundary")

        overall = analytics_data.get("overall_latency", {})
        sla_pass = analytics_data.get("sla_pass_rate_percent", 0.0)

        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.markdown(f"""
            <div class="hh-metric-chip">
                <div class="hh-metric-val">{overall.get('p50', 0)} ms</div>
                <div class="hh-metric-lbl">P50 Latency (Median)</div>
            </div>
            """, unsafe_allow_html=True)
        with m_c2:
            st.markdown(f"""
            <div class="hh-metric-chip">
                <div class="hh-metric-val">{overall.get('p70', 0)} ms</div>
                <div class="hh-metric-lbl">P70 Latency (70th %tile)</div>
            </div>
            """, unsafe_allow_html=True)
        with m_c3:
            st.markdown(f"""
            <div class="hh-metric-chip">
                <div class="hh-metric-val">{overall.get('p100', 0)} ms</div>
                <div class="hh-metric-lbl">P100 Latency (Max Tail)</div>
            </div>
            """, unsafe_allow_html=True)
        with m_c4:
            st.markdown(f"""
            <div class="hh-metric-chip">
                <div class="hh-metric-val" style="color: #00FF88;">{sla_pass}%</div>
                <div class="hh-metric-lbl">Sub-200ms SLA Pass Rate</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("#### ⏱️ Stage-by-Stage Percentile Breakdown (ms)")
        breakdown_dict = analytics_data.get("stage_breakdown", {})
        df_breakdown = pd.DataFrame(breakdown_dict).T
        st.dataframe(df_breakdown, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: Model Harness & Safety Guardrails
with tabs[2]:
    st.markdown('<div class="hh-studio-card">', unsafe_allow_html=True)
    st.markdown("### 🛡️ Production Model Harness & Safety Guardrails")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("""
        <div style="background: rgba(9, 26, 21, 0.85); border: 1px solid rgba(253, 184, 39, 0.3); border-radius: 12px; padding: 20px;">
            <span class="hh-pill-tag" style="margin-bottom: 8px;">⚙️ ORCHESTRATION & HARNESS</span>
            <ul style="color: #D1D5DB; font-size: 0.9rem; line-height: 1.7; margin-top: 10px;">
                <li><strong>Structured Pydantic Schemas</strong>: Strict <code>VoiceRAGRequest</code> and <code>VoiceRAGResponse</code> validation.</li>
                <li><strong>Tool Calling Engine</strong>:
                    <ul>
                        <li><code>refine_query_tool</code>: Normalizes speech transcript entities.</li>
                        <li><code>metadata_filter_tool</code>: Language & passage constraints.</li>
                        <li><code>synthesize_answer_tool</code>: Groq & Gemini synthesis with citations.</li>
                    </ul>
                </li>
                <li><strong>Fault Tolerance & Retries</strong>: Exponential backoff on transient vector / API timeouts.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_h2:
        st.markdown("""
        <div style="background: rgba(9, 26, 21, 0.85); border: 1px solid rgba(0, 229, 255, 0.3); border-radius: 12px; padding: 20px;">
            <span class="hh-pill-tag" style="background: rgba(0, 229, 255, 0.1); color: #00E5FF; border-color: rgba(0, 229, 255, 0.4); margin-bottom: 8px;">🛡️ MULTI-TIER GUARDRAILS</span>
            <ul style="color: #D1D5DB; font-size: 0.9rem; line-height: 1.7; margin-top: 10px;">
                <li><strong>Input Guardrail</strong>: Detects empty audio transcripts, off-topic requests, and adversarial prompt injections.</li>
                <li><strong>Grounding & Hallucination Guardrail</strong>: Evaluates word overlap and semantic similarity against retrieved passages.</li>
                <li><strong>Safe Refusal Handler</strong>: Gracefully refuses with explanation when context is insufficient or ungrounded.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# Branded Footer
# -------------------------------------------------------------------
st.markdown("""
<div class="hh-footer">
    <p>
        <strong>HACKER <span style="background:#E53E3E;color:#FFF;padding:1px 5px;border-radius:3px;font-size:0.75em;">गोवा</span> HOUSE 2026</strong> · Task #2: Voice-Enabled RAG System · 
        <span style="color: #FDB827;">⚡ Sub-200ms Latency SLA</span> · 
        <strong>#RAGInGoa</strong>
    </p>
</div>
""", unsafe_allow_html=True)
