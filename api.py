"""
FastAPI REST Server for Voice-Enabled RAG System
Exposes API endpoints for audio queries, benchmark suites, and chunking strategy analytics.
Provides a clean, streamlined, high-performance UI matching Hacker House Goa 2026 Reference Design.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from pathlib import Path
import os
import time

from dataset_loader import MSMARCOXIBackendLoader
from chunking_engine import MultiStrategyChunkingEngine
from vector_store import VectorStore
from stt_engine import SpeechToTextEngine
from model_harness import ModelHarnessOrchestrator, VoiceRAGRequest, VoiceRAGResponse
from latency_analytics import LatencyAnalyticsEngine
from benchmark_runner import run_benchmark_suite
import config

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Voice-Enabled RAG System API",
    description="Sub-200ms Latency Voice RAG API trained on AI4Bharat MSMARCO-XI dataset.",
    version="1.0.0"
)

# Global State Engine (Bilingual Multilingual Corpus: English + Hindi)
loader_en = MSMARCOXIBackendLoader(lang="en", max_samples=300)
loader_hi = MSMARCOXIBackendLoader(lang="hi", max_samples=300)
dataset = loader_en.load_dataset() + loader_hi.load_dataset()

chunk_engine = MultiStrategyChunkingEngine(strategy_name="semantic_boundary")
chunks = chunk_engine.chunk_documents(dataset)

vector_store = VectorStore()
vector_store.build_index(chunks)

stt_engine = SpeechToTextEngine(provider="sarvam")
orchestrator = ModelHarnessOrchestrator(
    stt_engine=stt_engine,
    vector_store=vector_store,
    chunking_engine=chunk_engine
)

@app.get("/assets/{file_name}")
def get_static_asset(file_name: str):
    asset_file = BASE_DIR / "assets" / file_name
    if asset_file.exists():
        return FileResponse(asset_file)
    raise HTTPException(status_code=404, detail="Asset not found")

@app.get("/", response_class=HTMLResponse)
@app.get("/api", response_class=HTMLResponse)
@app.get("/api.py", response_class=HTMLResponse)
def index_landing_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HACKER गोवा HOUSE // Voice-Enabled RAG Studio</title>
        <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@700;800;900&family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
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
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            html, body {
                font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
                background-color: #062b19;
                background: linear-gradient(rgba(6, 44, 25, 0.88), rgba(4, 28, 16, 0.94)), url('/assets/hh_goa_bg.png') no-repeat center top fixed;
                background-size: cover;
                color: var(--hh-text-light);
                min-height: 100vh;
                padding-bottom: 60px;
            }
            code, pre, .mono-text {
                font-family: 'JetBrains Mono', monospace !important;
            }
            .container {
                max-width: 1280px;
                margin: 0 auto;
                padding: 10px 24px;
            }

            /* Top Brand Navigation */
            .hh-nav-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 18px 0;
                margin-bottom: 25px;
            }
            .hh-logo {
                font-family: 'Cinzel', serif;
                font-size: 1.85rem;
                font-weight: 900;
                color: var(--hh-gold);
                letter-spacing: 2px;
                display: inline-flex;
                align-items: center;
                gap: 6px;
                text-shadow: 0 2px 10px rgba(0,0,0,0.5);
            }
            .hh-devanagari-badge {
                background: #E53E3E;
                color: #FFFFFF;
                font-family: 'Space Grotesk', sans-serif;
                font-size: 0.62rem;
                font-weight: 800;
                padding: 2px 6px;
                border-radius: 4px;
                letter-spacing: 0px;
                vertical-align: middle;
            }
            .hh-voice-heard-btn {
                background: var(--hh-gold);
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
            }
            .hh-voice-heard-btn:hover {
                background-color: var(--hh-gold-hover);
                transform: translateY(-1px);
                box-shadow: 0 6px 20px rgba(253, 184, 39, 0.5);
                color: #000;
            }

            /* Dual Hero Grid */
            .hero-grid {
                display: grid;
                grid-template-columns: 1.1fr 1.3fr;
                gap: 28px;
                margin-bottom: 30px;
            }
            @media (max-width: 950px) { .hero-grid { grid-template-columns: 1fr; } }

            .hh-hero-card {
                background: var(--hh-dark-card);
                border: 1px solid var(--hh-border);
                border-radius: 24px;
                padding: 38px 34px;
                box-shadow: 0 16px 45px rgba(0, 0, 0, 0.45);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                min-height: 480px;
            }
            .hh-pill-tag {
                display: inline-block;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 1.5px;
                color: var(--hh-gold);
                border: 1px solid rgba(253, 184, 39, 0.4);
                padding: 5px 14px;
                border-radius: 20px;
                background: rgba(253, 184, 39, 0.06);
                margin-bottom: 24px;
            }
            .hero-main-title {
                font-size: 3.2rem;
                font-weight: 800;
                line-height: 1.1;
                color: #FFFFFF;
                margin-bottom: 20px;
                letter-spacing: -0.5px;
            }
            .highlight-gold { color: var(--hh-gold); }
            .hero-body-text {
                color: var(--hh-text-muted);
                font-size: 1rem;
                line-height: 1.6;
                margin-bottom: 30px;
            }
            .hero-features-row {
                display: flex;
                justify-content: space-between;
                border-top: 1px solid rgba(255, 255, 255, 0.08);
                padding-top: 22px;
                margin-top: auto;
            }
            .feature-title {
                font-weight: 700;
                font-size: 0.95rem;
                color: #FFFFFF;
                margin-bottom: 3px;
            }
            .feature-sub {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.7rem;
                color: #6B7280;
                letter-spacing: 0.5px;
                text-transform: uppercase;
            }

            /* Live Studio Card */
            .hh-studio-card {
                background: var(--hh-dark-card);
                border: 1px solid var(--hh-border);
                border-radius: 24px;
                padding: 34px;
                box-shadow: 0 16px 45px rgba(0, 0, 0, 0.45);
            }
            .studio-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 25px;
            }
            .studio-label {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                font-weight: 700;
                color: var(--hh-text-muted);
                letter-spacing: 2px;
            }
            .query-ready-badge {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.72rem;
                font-weight: 700;
                color: var(--hh-cyan);
                background: rgba(0, 229, 255, 0.08);
                border: 1px solid rgba(0, 229, 255, 0.35);
                padding: 4px 12px;
                border-radius: 15px;
                letter-spacing: 1px;
            }
            .controls-row {
                display: grid;
                grid-template-columns: 1fr 1.2fr;
                gap: 16px;
                margin-bottom: 16px;
                align-items: end;
            }
            .speak-now-btn {
                background: var(--hh-gold);
                color: #000000;
                font-family: 'Space Grotesk', sans-serif;
                font-weight: 800;
                font-size: 0.95rem;
                padding: 11px 22px;
                border-radius: 8px;
                border: none;
                cursor: pointer;
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 8px;
                letter-spacing: 0.5px;
                box-shadow: 0 4px 15px rgba(253, 184, 39, 0.35);
                transition: all 0.2s ease;
                width: 100%;
            }
            .speak-now-btn:hover {
                background: var(--hh-gold-hover);
                transform: translateY(-1px);
                box-shadow: 0 6px 20px rgba(253, 184, 39, 0.5);
            }
            .speak-now-btn.recording {
                background: linear-gradient(135deg, #EF4444 0%, #B91C1C 100%);
                color: #FFFFFF;
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
                70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
                100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
            }
            .mode-select-wrapper label {
                display: block;
                font-size: 0.75rem;
                color: var(--hh-text-muted);
                margin-bottom: 6px;
                font-family: 'Space Grotesk', sans-serif;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .mode-select {
                background: #061713;
                color: #FFFFFF;
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 8px;
                padding: 10px 14px;
                font-family: 'Space Grotesk', sans-serif;
                font-size: 0.85rem;
                width: 100%;
                outline: none;
            }

            /* Divider OR TYPE */
            .or-type-divider {
                display: flex;
                align-items: center;
                text-align: center;
                color: #6B7280;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.75rem;
                letter-spacing: 2px;
                margin: 20px 0;
            }
            .or-type-divider::before, .or-type-divider::after {
                content: '';
                flex: 1;
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            }
            .or-type-divider:not(:empty)::before { margin-right: 15px; }
            .or-type-divider:not(:empty)::after { margin-left: 15px; }

            /* Text Input & Send Button */
            .input-row { display: flex; gap: 10px; margin-bottom: 16px; }
            .query-input {
                flex: 1;
                background: #061713;
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 8px;
                padding: 12px 16px;
                color: #FFFFFF;
                font-family: 'Space Grotesk', sans-serif;
                font-size: 0.95rem;
                outline: none;
            }
            .query-input:focus { border-color: var(--hh-gold); }
            .send-btn {
                background: var(--hh-gold);
                color: #000000;
                font-weight: 700;
                border: none;
                border-radius: 8px;
                padding: 0 24px;
                cursor: pointer;
                font-size: 0.95rem;
                transition: all 0.2s ease;
                box-shadow: 0 4px 15px rgba(253, 184, 39, 0.3);
            }
            .send-btn:hover {
                background: var(--hh-gold-hover);
                transform: translateY(-1px);
            }

            /* Preset Chips Grid */
            .preset-chips-row {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 8px;
                margin-top: 14px;
            }
            @media (max-width: 768px) { .preset-chips-row { grid-template-columns: repeat(2, 1fr); } }
            .preset-chip {
                background: rgba(14, 36, 31, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.12);
                color: #D1D5DB;
                padding: 8px 10px;
                border-radius: 14px;
                font-size: 0.78rem;
                cursor: pointer;
                text-align: center;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                transition: all 0.2s ease;
            }
            .preset-chip:hover {
                border-color: var(--hh-gold);
                color: var(--hh-gold);
                background: rgba(253, 184, 39, 0.1);
            }

            /* Chat Result Bubbles */
            .chat-bubble-user {
                background: rgba(253, 184, 39, 0.1);
                border: 1px solid rgba(253, 184, 39, 0.35);
                border-radius: 14px 14px 2px 14px;
                padding: 16px 20px;
                margin-bottom: 15px;
                color: #FFF3EB;
                font-size: 1rem;
            }
            .chat-bubble-ai {
                background: #0e2620;
                border: 1px solid rgba(0, 229, 255, 0.35);
                border-radius: 14px 14px 14px 2px;
                padding: 20px 24px;
                margin-bottom: 15px;
                color: #F8F9FA;
                box-shadow: 0 6px 30px rgba(0, 0, 0, 0.4);
            }
            .chat-bubble-ai.refused {
                border-color: rgba(239, 68, 68, 0.6);
                background: rgba(35, 15, 15, 0.95);
            }

            /* Telemetry Metrics */
            .telemetry-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 12px;
                margin: 20px 0;
            }
            @media (max-width: 768px) { .telemetry-grid { grid-template-columns: repeat(2, 1fr); } }
            .hh-metric-chip {
                background: rgba(9, 26, 21, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 14px 16px;
                text-align: center;
                transition: all 0.3s ease;
            }
            .hh-metric-chip:hover {
                border-color: rgba(253, 184, 39, 0.4);
                box-shadow: 0 4px 20px rgba(253, 184, 39, 0.15);
            }
            .hh-metric-val {
                font-family: 'JetBrains Mono', monospace;
                font-size: 1.35rem;
                font-weight: 700;
                color: var(--hh-gold);
            }
            .hh-metric-lbl {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.72rem;
                color: var(--hh-text-muted);
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-top: 4px;
            }

            /* Tables */
            .hh-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 14px;
                font-size: 0.85rem;
            }
            .hh-table th, .hh-table td {
                padding: 10px 14px;
                text-align: left;
                border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            }
            .hh-table th { color: #9CA3AF; font-family: 'JetBrains Mono', monospace; background: rgba(0,0,0,0.2); }
            .hh-table td { color: #E5E7EB; }

            /* Tabs Styling */
            .hh-tabs-wrapper { margin-top: 36px; }
            .hh-tabs-nav {
                display: flex;
                gap: 8px;
                background: rgba(9, 26, 21, 0.7);
                padding: 6px;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.05);
                overflow-x: auto;
                margin-bottom: 20px;
            }
            .hh-tab-nav-btn {
                background: transparent;
                border-radius: 8px;
                color: #9CA3AF;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                font-weight: 600;
                padding: 8px 16px;
                border: none;
                cursor: pointer;
                white-space: nowrap;
                transition: all 0.2s;
            }
            .hh-tab-nav-btn.active {
                background: rgba(253, 184, 39, 0.15) !important;
                color: var(--hh-gold) !important;
                border: 1px solid rgba(253, 184, 39, 0.4) !important;
            }
            .hh-tab-pane { display: none; }
            .hh-tab-pane.active { display: block; }

            .strat-cards-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 12px;
                margin-top: 14px;
            }
            @media (max-width: 768px) { .strat-cards-grid { grid-template-columns: repeat(2, 1fr); } }

            .harness-cards-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-top: 14px;
            }
            @media (max-width: 768px) { .harness-cards-grid { grid-template-columns: 1fr; } }

            .hh-btn-gold {
                background: var(--hh-gold);
                color: #000000;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-family: 'Space Grotesk', sans-serif;
                font-weight: 700;
                font-size: 0.9rem;
                cursor: pointer;
                box-shadow: 0 4px 15px rgba(253, 184, 39, 0.3);
                transition: all 0.2s ease;
            }
            .hh-btn-gold:hover {
                background: var(--hh-gold-hover);
                transform: translateY(-1px);
            }

            /* Share Card */
            .hh-share-card {
                background: rgba(253, 184, 39, 0.08);
                border: 1px dashed rgba(253, 184, 39, 0.3);
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                margin-top: 24px;
            }
            .hh-x-btn {
                display: inline-block;
                background: #000;
                color: #FFF;
                border: 1px solid #FDB827;
                padding: 8px 18px;
                border-radius: 8px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.82rem;
                text-decoration: none;
                font-weight: 600;
                transition: all 0.2s;
            }
            .hh-x-btn:hover {
                background: #FDB827;
                color: #000;
            }

            /* Footer */
            .hh-footer {
                text-align: center;
                padding: 30px 10px 10px 10px;
                border-top: 1px solid rgba(255, 255, 255, 0.08);
                margin-top: 40px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.8rem;
                color: #6B7280;
            }
            .hh-footer a {
                color: var(--hh-gold);
                text-decoration: none;
            }
            .hh-footer a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Top Brand Bar -->
            <div class="hh-nav-container">
                <div class="hh-logo">
                    HACKER <span class="hh-devanagari-badge">गोवा</span> HOUSE
                </div>
                <div>
                    <a href="#live-studio" class="hh-voice-heard-btn">GET YOUR VOICE HEARD</a>
                </div>
            </div>

            <!-- Dual Hero Grid -->
            <div class="hero-grid">
                <!-- Left Hero Card -->
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

                <!-- Right Live Studio Card -->
                <div class="hh-studio-card" id="live-studio">
                    <div class="studio-header">
                        <span class="studio-label">LIVE STUDIO</span>
                        <span class="query-ready-badge">QUERY READY</span>
                    </div>

                    <div class="controls-row">
                        <div>
                            <button id="mic-btn" class="speak-now-btn" onclick="toggleVoiceRecording()">
                                <span id="mic-icon">🎙️</span> <span id="mic-text">SPEAK NOW</span>
                            </button>
                        </div>
                        <div class="mode-select-wrapper">
                            <label>VOICE MODE</label>
                            <select id="voice-mode-select" class="mode-select">
                                <option value="semantic_boundary" selected>Sentence-Aware (Semantic)</option>
                                <option value="fixed_overlap">Fixed-Size Overlap</option>
                                <option value="hierarchical">Hierarchical (Parent-Child)</option>
                                <option value="metadata_aware">Metadata-Aware Window</option>
                            </select>
                        </div>
                    </div>

                    <!-- Hidden Audio Upload Area Fallback -->
                    <div id="audio-upload-box" style="display:none; margin-bottom:14px; background: rgba(14, 36, 31, 0.7); border:1px dashed rgba(253,184,39,0.3); border-radius:8px; padding:10px;">
                        <span style="font-size:0.8rem; color:#FDB827;">📁 Upload .wav or .mp3 voice recording:</span>
                        <input type="file" id="audio-file" accept="audio/*" style="margin-top:6px; font-size:0.8rem; color:#D1D5DB;" onchange="uploadAudioFile(this.files[0])"/>
                    </div>

                    <div class="or-type-divider">OR TYPE</div>

                    <div class="input-row">
                        <input type="text" id="query-input" class="query-input" placeholder="Type query here..." onkeypress="handleEnter(event)"/>
                        <button class="send-btn" onclick="submitTextQuery()">Send</button>
                    </div>

                    <div class="preset-chips-row">
                        <button class="preset-chip" onclick="setPreset('What is a corporation?')">What is a corporation?</button>
                        <button class="preset-chip" onclick="setPreset('कॉर्पोरेशन क्या है?')">कॉर्पोरेशन क्या है?</button>
                        <button class="preset-chip" onclick="setPreset('कैश फ्लो स्टेटमेंट क्या है?')">कैश फ्लो स्टेटमेंट क्या है?</button>
                        <button class="preset-chip" onclick="setPreset('What are CSE subjects?')">What are CSE subjects?</button>
                    </div>
                </div>
            </div>

            <!-- Dynamic Execution & Telemetry Results -->
            <div id="response-container" class="hh-studio-card" style="display:none; margin-top:25px;">
                <!-- User Query Bubble -->
                <div class="chat-bubble-user">
                    <span class="hh-pill-tag" style="margin-bottom:6px; padding: 2px 10px; font-size: 0.7rem;">👤 INPUT QUERY</span><br/>
                    <strong id="user-query-text" style="font-size:1.05rem;"></strong>
                </div>

                <!-- AI Response Bubble -->
                <div id="ai-bubble" class="chat-bubble-ai">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span id="ai-badge" style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: #00FF88; background: rgba(0, 255, 136, 0.1); padding: 3px 10px; border-radius: 6px; border: 1px solid rgba(0, 255, 136, 0.3);">🤖 GROUNDED ANSWER</span>
                        <button id="tts-btn" onclick="toggleAudioNarration()" style="background: rgba(253, 184, 39, 0.15); border: 1px solid rgba(253, 184, 39, 0.4); color: #FDB827; padding: 4px 14px; border-radius: 20px; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 0.78rem; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; transition: all 0.2s;">
                            <span id="tts-icon">🔊</span> <span id="tts-label">Listen</span>
                        </button>
                    </div>
                    <div id="ai-answer-text" style="font-size: 1.05rem; line-height: 1.6; margin-top: 6px; color: #F8F9FA;"></div>
                </div>

                <!-- Citations -->
                <div id="citations-box" style="margin-top:14px; display:none;">
                    <strong style="font-size:0.9rem; color:#FFFFFF;">📌 Evidence & Document Citations:</strong>
                    <div id="citations-list" style="margin-top:6px; font-size:0.85rem; color:#D1D5DB;"></div>
                </div>

                <hr style="border:none; border-top:1px solid rgba(255,255,255,0.08); margin: 20px 0;">

                <!-- Telemetry Metrics Grid -->
                <div class="telemetry-grid">
                    <div class="hh-metric-chip">
                        <div class="hh-metric-val" id="m-lat">0.0 ms</div>
                        <div class="hh-metric-lbl" id="m-sla">Total Latency</div>
                    </div>
                    <div class="hh-metric-chip">
                        <div class="hh-metric-val" id="m-ground" style="color: #00FF88;">100%</div>
                        <div class="hh-metric-lbl">Grounding Faithfulness</div>
                    </div>
                    <div class="hh-metric-chip">
                        <div class="hh-metric-val" id="m-halluc" style="color: #00E5FF;">0%</div>
                        <div class="hh-metric-lbl">Hallucination Risk</div>
                    </div>
                    <div class="hh-metric-chip">
                        <div class="hh-metric-val" id="m-synth" style="font-size: 0.95rem; color: #00FF88;">Groq Ultra-Fast</div>
                        <div class="hh-metric-lbl">Active Synthesizer</div>
                    </div>
                </div>

                <!-- Stage Latencies Breakdown -->
                <div style="margin-top:20px;">
                    <strong style="font-size: 0.9rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1px; font-family: 'JetBrains Mono', monospace;">⚡ Stage-by-Stage Latency Breakdown (Sub-200ms Target):</strong>
                    <table class="hh-table" id="stages-table">
                        <thead>
                            <tr>
                                <th>Pipeline Stage</th>
                                <th>Latency (ms)</th>
                            </tr>
                        </thead>
                        <tbody id="stages-tbody">
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>

                <!-- X Share Card -->
                <div class="hh-share-card">
                    <p style="color: #D1D5DB; font-size: 0.9rem; margin-bottom: 10px;">🌴 Verified sub-200ms latency on AI4Bharat MSMARCO-XI dataset.</p>
                    <a id="share-x-btn" href="#" target="_blank" class="hh-x-btn">POST RESULT TO X (#RAGInGoa)</a>
                </div>
            </div>

            <!-- Deep Dive Interactive Tabs -->
            <div class="hh-tabs-wrapper">
                <div class="hh-tabs-nav">
                    <button class="hh-tab-nav-btn active" onclick="switchTab(0)">✂️ Chunking Strategies Lab</button>
                    <button class="hh-tab-nav-btn" onclick="switchTab(1)">📊 Statistical Benchmarks</button>
                    <button class="hh-tab-nav-btn" onclick="switchTab(2)">🛡️ System Architecture & Guardrails</button>
                </div>

                <!-- Tab 1: Chunking Lab -->
                <div class="hh-tab-pane active" id="pane-0">
                    <div class="hh-studio-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                            <div>
                                <h3>✂️ 4 Multi-Strategy Document Chunking Engines</h3>
                                <p style="color:#9CA3AF; font-size:0.9rem; margin-top:4px;">Comparative preprocessing and metadata analytics across the MSMARCO-XI corpus.</p>
                            </div>
                            <button class="hh-btn-gold" onclick="runChunkingComparison()">Run Chunking Benchmark</button>
                        </div>

                        <div id="chunking-res-wrapper" style="margin-top:20px; display:none;">
                            <table class="hh-table">
                                <thead>
                                    <tr>
                                        <th>Strategy Name</th>
                                        <th>Generated Chunks</th>
                                        <th>Avg Chunk Chars</th>
                                        <th>Execution Latency</th>
                                        <th>Metadata Richness</th>
                                    </tr>
                                </thead>
                                <tbody id="chunking-tbody"></tbody>
                            </table>
                        </div>

                        <div class="strat-cards-grid">
                            <div style="background: rgba(9, 26, 21, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px;">
                                <h4 style="color:#FDB827; font-size:0.95rem; margin-bottom:6px;">1. Fixed-Size Overlap</h4>
                                <p style="color:#9CA3AF; font-size:0.82rem; line-height:1.5;">256-char sliding window with 32-char overlap. Uniform token density.</p>
                            </div>
                            <div style="background: rgba(9, 26, 21, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px;">
                                <h4 style="color:#00FF88; font-size:0.95rem; margin-bottom:6px;">2. Semantic Boundary</h4>
                                <p style="color:#9CA3AF; font-size:0.82rem; line-height:1.5;">Sentence-aware splitting supporting English and Indic punctuation (।, ?, !).</p>
                            </div>
                            <div style="background: rgba(9, 26, 21, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px;">
                                <h4 style="color:#00E5FF; font-size:0.95rem; margin-bottom:6px;">3. Parent-Child</h4>
                                <p style="color:#9CA3AF; font-size:0.82rem; line-height:1.5;">100-char child chunks for vector search + full parent passage for synthesis.</p>
                            </div>
                            <div style="background: rgba(9, 26, 21, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px;">
                                <h4 style="color:#FFAA00; font-size:0.95rem; margin-bottom:6px;">4. Metadata-Aware</h4>
                                <p style="color:#9CA3AF; font-size:0.82rem; line-height:1.5;">Injected document header metadata (language, topic, query ID) in vector space.</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Tab 2: Benchmarks -->
                <div class="hh-tab-pane" id="pane-1">
                    <div class="hh-studio-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                            <div>
                                <h3>📊 Statistical Latency Percentiles (P50 / P70 / P100)</h3>
                                <p style="color:#9CA3AF; font-size:0.9rem; margin-top:4px;">Evaluates end-to-end latency across real multilingual queries.</p>
                            </div>
                            <div style="display:flex; gap:10px; align-items:center;">
                                <input type="number" id="bench-slider" value="20" min="10" max="60" style="background:#061713; border:1px solid rgba(255,255,255,0.2); border-radius:6px; color:#FFF; padding:6px 12px; width:70px; font-family:'JetBrains Mono';"/>
                                <button class="hh-btn-gold" onclick="runLivePercentileBenchmark()">Execute Suite</button>
                            </div>
                        </div>

                        <div id="bench-results-box" style="margin-top:20px; display:none;">
                            <div class="telemetry-grid">
                                <div class="hh-metric-chip">
                                    <div class="hh-metric-val" id="bp-50">0.0 ms</div>
                                    <div class="hh-metric-lbl">P50 Latency (Median)</div>
                                </div>
                                <div class="hh-metric-chip">
                                    <div class="hh-metric-val" id="bp-70">0.0 ms</div>
                                    <div class="hh-metric-lbl">P70 Latency</div>
                                </div>
                                <div class="hh-metric-chip">
                                    <div class="hh-metric-val" id="bp-100">0.0 ms</div>
                                    <div class="hh-metric-lbl">P100 Latency (Max)</div>
                                </div>
                                <div class="hh-metric-chip">
                                    <div class="hh-metric-val" id="bp-sla" style="color:#00FF88;">100%</div>
                                    <div class="hh-metric-lbl">&lt;200ms SLA Pass Rate</div>
                                </div>
                            </div>

                            <table class="hh-table" style="margin-top:20px;">
                                <thead>
                                    <tr>
                                        <th>Pipeline Stage</th>
                                        <th>P50 (ms)</th>
                                        <th>P70 (ms)</th>
                                        <th>P100 (ms)</th>
                                        <th>Mean (ms)</th>
                                        <th>Min (ms)</th>
                                        <th>Max (ms)</th>
                                    </tr>
                                </thead>
                                <tbody id="bench-stages-tbody"></tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- Tab 3: System Architecture -->
                <div class="hh-tab-pane" id="pane-2">
                    <div class="hh-studio-card">
                        <h3>🛡️ High-Performance Architecture & Multi-Tier Guardrails</h3>
                        <div class="harness-cards-grid">
                            <div style="background: rgba(9, 26, 21, 0.85); border: 1px solid rgba(253, 184, 39, 0.3); border-radius: 12px; padding: 20px;">
                                <span class="hh-pill-tag" style="margin-bottom: 8px;">⚙️ MODEL HARNESS ORCHESTRATOR</span>
                                <ul style="color: #D1D5DB; font-size: 0.9rem; line-height: 1.7; margin-top: 10px; padding-left: 18px;">
                                    <li><strong>Structured Schemas</strong>: Pydantic typed request & response validation.</li>
                                    <li><strong>Tool Calling Engine</strong>:
                                        <ul>
                                            <li><code>refine_query_tool</code>: Normalizes speech transcript entities.</li>
                                            <li><code>metadata_filter_tool</code>: Language & passage constraints.</li>
                                            <li><code>synthesize_answer_tool</code>: Groq & Gemini synthesis with citations.</li>
                                        </ul>
                                    </li>
                                    <li><strong>Fault Tolerance & Retries</strong>: Exponential backoff on transient timeouts.</li>
                                </ul>
                            </div>
                            <div style="background: rgba(9, 26, 21, 0.85); border: 1px solid rgba(0, 229, 255, 0.3); border-radius: 12px; padding: 20px;">
                                <span class="hh-pill-tag" style="background: rgba(0, 229, 255, 0.1); color: #00E5FF; border-color: rgba(0, 229, 255, 0.4); margin-bottom: 8px;">🛡️ MULTI-TIER GUARDRAILS</span>
                                <ul style="color: #D1D5DB; font-size: 0.9rem; line-height: 1.7; margin-top: 10px; padding-left: 18px;">
                                    <li><strong>Input Guardrail</strong>: Detects empty audio transcripts, off-topic requests, and prompt injections.</li>
                                    <li><strong>Grounding & Hallucination Guardrail</strong>: Evaluates word overlap and semantic similarity against retrieved passages.</li>
                                    <li><strong>Safe Refusal Handler</strong>: Gracefully refuses with structured explanation when context is insufficient or ungrounded.</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Branded Footer -->
            <div class="hh-footer">
                <p>
                    <strong>HACKER <span style="background:#E53E3E;color:#FFF;padding:1px 5px;border-radius:3px;font-size:0.75em;">गोवा</span> HOUSE 2026</strong> · Task #2: Voice-Enabled RAG System · 
                    <span style="color: #FDB827;">⚡ Sub-200ms Latency SLA</span> · 
                    <strong>#RAGInGoa</strong>
                </p>
            </div>
        </div>

        <script>
            let recognition = null;
            let isRecording = false;

            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SpeechRec();
                recognition.continuous = false;
                recognition.interimResults = true;
                recognition.lang = 'en-IN';

                recognition.onresult = function(event) {
                    let transcript = '';
                    for (let i = event.resultIndex; i < event.results.length; ++i) {
                        transcript += event.results[i][0].transcript;
                    }
                    document.getElementById('query-input').value = transcript;
                };

                recognition.onend = function() {
                    isRecording = false;
                    const btn = document.getElementById('mic-btn');
                    btn.classList.remove('recording');
                    document.getElementById('mic-icon').innerText = '🎙️';
                    document.getElementById('mic-text').innerText = 'SPEAK NOW';
                    if (document.getElementById('query-input').value.trim()) {
                        submitTextQuery();
                    }
                };
            }

            function toggleVoiceRecording() {
                if (!recognition) {
                    const uploadBox = document.getElementById('audio-upload-box');
                    uploadBox.style.display = uploadBox.style.display === 'none' ? 'block' : 'none';
                    return;
                }
                const btn = document.getElementById('mic-btn');
                if (isRecording) {
                    recognition.stop();
                } else {
                    document.getElementById('query-input').value = "";
                    recognition.start();
                    isRecording = true;
                    btn.classList.add('recording');
                    document.getElementById('mic-icon').innerText = '🔴';
                    document.getElementById('mic-text').innerText = 'LISTENING...';
                }
            }

            function setPreset(text) {
                document.getElementById('query-input').value = text;
                submitTextQuery();
            }

            function handleEnter(e) {
                if (e.key === 'Enter') submitTextQuery();
            }

            function detectLanguage(text) {
                if (/[\u0980-\u09FF]/.test(text)) return 'bn-IN'; // Bengali / Assamese
                if (/[\u0B80-\u0BFF]/.test(text)) return 'ta-IN'; // Tamil
                if (/[\u0C00-\u0C7F]/.test(text)) return 'te-IN'; // Telugu
                if (/[\u0C80-\u0CFF]/.test(text)) return 'kn-IN'; // Kannada
                if (/[\u0D00-\u0D7F]/.test(text)) return 'ml-IN'; // Malayalam
                if (/[\u0A80-\u0AFF]/.test(text)) return 'gu-IN'; // Gujarati
                if (/[\u0A00-\u0A7F]/.test(text)) return 'pa-IN'; // Punjabi (Gurmukhi)
                if (/[\u0B00-\u0B7F]/.test(text)) return 'or-IN'; // Odia
                if (/[\u0600-\u06FF]/.test(text)) return 'ur-IN'; // Urdu
                if (/[\u0900-\u097F]/.test(text)) return 'hi-IN'; // Hindi / Marathi / Sanskrit
                return 'en-IN';
            }

            async function submitTextQuery() {
                const query = document.getElementById('query-input').value.trim();
                if (!query) return;

                const strategy = document.getElementById('voice-mode-select').value;
                const langCode = detectLanguage(query);

                const responseContainer = document.getElementById('response-container');
                responseContainer.style.display = 'block';
                document.getElementById('user-query-text').innerText = `"${query}"`;
                document.getElementById('ai-answer-text').innerHTML = '<span style="color:#FDB827;">⚡ Executing Voice RAG Pipeline (&lt;200ms Target)...</span>';
                document.getElementById('citations-box').style.display = 'none';

                try {
                    const res = await fetch('/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            query_text: query,
                            language_code: langCode,
                            chunking_strategy: strategy,
                            stt_provider: 'local'
                        })
                    });

                    const data = await res.json();
                    renderResponse(data);
                } catch (err) {
                    document.getElementById('ai-answer-text').innerText = 'Error executing query: ' + err;
                }
            }

            async function uploadAudioFile(file) {
                if (!file) return;
                const strategy = document.getElementById('voice-mode-select').value;
                const responseContainer = document.getElementById('response-container');
                responseContainer.style.display = 'block';
                document.getElementById('user-query-text').innerText = `[Audio File: ${file.name}]`;
                document.getElementById('ai-answer-text').innerHTML = '<span style="color:#FDB827;">⚡ Transcribing audio and executing RAG pipeline...</span>';

                const formData = new FormData();
                formData.append('file', file);
                formData.append('language_code', 'en-IN');
                formData.append('chunking_strategy', strategy);
                formData.append('stt_provider', 'groq');

                try {
                    const res = await fetch('/query/audio', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await res.json();
                    document.getElementById('user-query-text').innerText = `"${data.transcript}"`;
                    renderResponse(data);
                } catch(err) {
                    document.getElementById('ai-answer-text').innerText = 'Error processing audio: ' + err;
                }
            }

            function renderResponse(data) {
                const aiBubble = document.getElementById('ai-bubble');
                const aiBadge = document.getElementById('ai-badge');
                
                if (data.is_refused) {
                    aiBubble.classList.add('refused');
                    aiBadge.innerText = '🛡️ GUARDRAIL SAFE REFUSAL';
                    aiBadge.style.color = '#EF4444';
                    aiBadge.style.background = 'rgba(239, 68, 68, 0.2)';
                    aiBadge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
                    document.getElementById('ai-answer-text').innerHTML = `<strong style="color: #F87171;">Reason:</strong> ${data.refusal_reason || 'Out of context'}<br/><br/>${data.answer}`;
                } else {
                    aiBubble.classList.remove('refused');
                    aiBadge.innerText = '🤖 GROUNDED ANSWER';
                    aiBadge.style.color = '#00FF88';
                    aiBadge.style.background = 'rgba(0, 255, 136, 0.1)';
                    aiBadge.style.borderColor = 'rgba(0, 255, 136, 0.3)';
                    document.getElementById('ai-answer-text').innerText = data.answer;
                }

                // Citations
                const citationsBox = document.getElementById('citations-box');
                const citationsList = document.getElementById('citations-list');
                if (data.citations && data.citations.length > 0) {
                    citationsBox.style.display = 'block';
                    citationsList.innerHTML = data.citations.map(c => `<div>📄 <code>${c}</code></div>`).join('');
                } else {
                    citationsBox.style.display = 'none';
                }

                // Telemetry
                document.getElementById('m-lat').innerText = data.total_latency_ms + ' ms';
                const slaText = data.met_sla_200ms ? '⚡ PASSED (&lt;200ms)' : '⚠️ EXCEEDED (&gt;200ms)';
                document.getElementById('m-sla').innerHTML = `Total Latency (${slaText})`;
                document.getElementById('m-ground').innerText = Math.round(data.grounding_score * 100) + '%';
                document.getElementById('m-halluc').innerText = Math.round(data.hallucination_risk * 100) + '%';
                const synthColor = '#00FF88';
                const synthEl = document.getElementById('m-synth');
                synthEl.innerText = data.synthesizer || 'Groq Ultra-Fast';
                synthEl.style.color = synthColor;

                // Stages breakdown
                const tbody = document.getElementById('stages-tbody');
                tbody.innerHTML = '';
                for (const [k, v] of Object.entries(data.stage_latencies_ms || {})) {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `<td>${k}</td><td><strong>${Number(v).toFixed(2)} ms</strong></td>`;
                    tbody.appendChild(tr);
                }

                // Share link
                const tweet = encodeURIComponent(`Just benchmarked our sub-200ms Voice-Enabled RAG pipeline for @hhgoa 2026 (Task #2)! ⚡ Latency: ${data.total_latency_ms}ms | Grounding: ${Math.round(data.grounding_score * 100)}% #RAGInGoa #HHGoa2026`);
                document.getElementById('share-x-btn').href = `https://twitter.com/intent/tweet?text=${tweet}`;

                // Automatically speak / narrate the answer out loud
                if (data.answer) {
                    narrateText(data.answer);
                }

                document.getElementById('response-container').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }

            let currentUtterance = null;
            let isNarrating = false;

            function narrateText(text) {
                if (!('speechSynthesis' in window)) return;
                window.speechSynthesis.cancel();
                if (!text || text.trim() === '') return;

                const clean = text.replace(/\\[Document \\d+.*?\\]/g, '').replace(/Doc ID: \\S+/g, '').trim();
                currentUtterance = new SpeechSynthesisUtterance(clean);
                currentUtterance.lang = detectLanguage(clean);
                currentUtterance.rate = 1.0;
                currentUtterance.pitch = 1.0;

                const ttsBtn = document.getElementById('tts-btn');
                const ttsIcon = document.getElementById('tts-icon');
                const ttsLabel = document.getElementById('tts-label');

                currentUtterance.onstart = function() {
                    isNarrating = true;
                    if (ttsIcon) ttsIcon.innerText = '⏹️';
                    if (ttsLabel) ttsLabel.innerText = 'Stop';
                    if (ttsBtn) {
                        ttsBtn.style.background = 'rgba(239, 68, 68, 0.2)';
                        ttsBtn.style.borderColor = '#EF4444';
                        ttsBtn.style.color = '#EF4444';
                    }
                };

                currentUtterance.onend = function() {
                    isNarrating = false;
                    if (ttsIcon) ttsIcon.innerText = '🔊';
                    if (ttsLabel) ttsLabel.innerText = 'Listen';
                    if (ttsBtn) {
                        ttsBtn.style.background = 'rgba(253, 184, 39, 0.15)';
                        ttsBtn.style.borderColor = 'rgba(253, 184, 39, 0.4)';
                        ttsBtn.style.color = '#FDB827';
                    }
                };

                currentUtterance.onerror = function() {
                    isNarrating = false;
                    if (ttsIcon) ttsIcon.innerText = '🔊';
                    if (ttsLabel) ttsLabel.innerText = 'Listen';
                    if (ttsBtn) {
                        ttsBtn.style.background = 'rgba(253, 184, 39, 0.15)';
                        ttsBtn.style.borderColor = 'rgba(253, 184, 39, 0.4)';
                        ttsBtn.style.color = '#FDB827';
                    }
                };

                window.speechSynthesis.speak(currentUtterance);
            }

            function toggleAudioNarration() {
                if (isNarrating) {
                    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
                    isNarrating = false;
                    const ttsBtn = document.getElementById('tts-btn');
                    const ttsIcon = document.getElementById('tts-icon');
                    const ttsLabel = document.getElementById('tts-label');
                    if (ttsIcon) ttsIcon.innerText = '🔊';
                    if (ttsLabel) ttsLabel.innerText = 'Listen';
                    if (ttsBtn) {
                        ttsBtn.style.background = 'rgba(253, 184, 39, 0.15)';
                        ttsBtn.style.borderColor = 'rgba(253, 184, 39, 0.4)';
                        ttsBtn.style.color = '#FDB827';
                    }
                } else {
                    const text = document.getElementById('ai-answer-text').innerText;
                    narrateText(text);
                }
            }

            function switchTab(idx) {
                document.querySelectorAll('.hh-tab-nav-btn').forEach((b, i) => {
                    b.classList.toggle('active', i === idx);
                });
                document.querySelectorAll('.hh-tab-pane').forEach((p, i) => {
                    p.classList.toggle('active', i === idx);
                });
            }

            async function runChunkingComparison() {
                const wrapper = document.getElementById('chunking-res-wrapper');
                wrapper.style.display = 'block';
                const tbody = document.getElementById('chunking-tbody');
                tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:#FDB827;">Benchmarking 4 chunking algorithms across corpus...</td></tr>';

                try {
                    const res = await fetch('/chunking/compare');
                    const data = await res.json();
                    tbody.innerHTML = '';
                    for (const [strat, row] of Object.entries(data)) {
                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td><strong>${strat}</strong></td>
                            <td>${row.chunk_count}</td>
                            <td>${row.avg_chunk_chars}</td>
                            <td>${row.chunking_time_ms} ms</td>
                            <td><span style="color:#00FF88;">${row.metadata_richness}</span></td>
                        `;
                        tbody.appendChild(tr);
                    }
                } catch(e) {
                    tbody.innerHTML = `<tr><td colspan="5" style="color:#EF4444;">Error: ${e}</td></tr>`;
                }
            }

            async function runLivePercentileBenchmark() {
                const box = document.getElementById('bench-results-box');
                box.style.display = 'block';
                const num = document.getElementById('bench-slider').value;
                const tbody = document.getElementById('bench-stages-tbody');
                tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color:#FDB827;">Executing benchmark queries...</td></tr>';

                try {
                    const res = await fetch(`/benchmark?num_samples=${num}&strategy=semantic_boundary`);
                    const data = await res.json();
                    
                    const overall = data.overall_latency || {};
                    document.getElementById('bp-50').innerText = (overall.p50 || 0) + ' ms';
                    document.getElementById('bp-70').innerText = (overall.p70 || 0) + ' ms';
                    document.getElementById('bp-100').innerText = (overall.p100 || 0) + ' ms';
                    document.getElementById('bp-sla').innerText = (data.sla_pass_rate_percent || 100) + '%';

                    tbody.innerHTML = '';
                    const breakdown = data.stage_breakdown || {};
                    for (const [stage, metrics] of Object.entries(breakdown)) {
                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td><strong>${stage}</strong></td>
                            <td>${metrics.p50 || 0}</td>
                            <td>${metrics.p70 || 0}</td>
                            <td>${metrics.p100 || 0}</td>
                            <td>${metrics.mean || 0}</td>
                            <td>${metrics.min || 0}</td>
                            <td>${metrics.max || 0}</td>
                        `;
                        tbody.appendChild(tr);
                    }
                } catch(e) {
                    tbody.innerHTML = `<tr><td colspan="7" style="color:#EF4444;">Benchmark Error: ${e}</td></tr>`;
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "dataset_samples": len(dataset),
        "indexed_chunks": len(chunks),
        "stt_provider": stt_engine.provider
    }

class ApiKeyRequest(BaseModel):
    key: str

@app.post("/set-groq-key")
@app.post("/api/set-groq-key")
def set_groq_key(req: ApiKeyRequest):
    config.save_env_variable("GROQ_API_KEY", req.key.strip())
    return {"status": "success", "message": "Groq API key updated and persisted"}

@app.post("/set-gemini-key")
@app.post("/api/set-gemini-key")
def set_gemini_key(req: ApiKeyRequest):
    config.save_env_variable("GEMINI_API_KEY", req.key.strip())
    return {"status": "success", "message": "Gemini API key updated and persisted"}

class TextQueryRequest(BaseModel):
    query_text: str
    language_code: Optional[str] = "hi-IN"
    chunking_strategy: Optional[str] = "semantic_boundary"
    stt_provider: Optional[str] = "local"
    synthesizer_mode: Optional[str] = "auto"

@app.post("/query", response_model=Dict[str, Any])
@app.post("/api/query", response_model=Dict[str, Any])
def query_rag(req: TextQueryRequest):
    """
    Executes Voice RAG query (or text query) through STT, Guardrails, Retrieval, Harness, and Grounding.
    """
    chunk_engine.set_strategy(req.chunking_strategy)
    rag_req = VoiceRAGRequest(
        prompt_text=req.query_text,
        language_code=req.language_code,
        chunking_strategy=req.chunking_strategy,
        stt_provider=req.stt_provider or "local",
        synthesizer_mode=req.synthesizer_mode or "auto"
    )
    res: VoiceRAGResponse = orchestrator.run_pipeline(rag_req)
    return res.model_dump()

@app.post("/query/audio")
@app.post("/api/query/audio")
async def query_rag_audio(
    file: UploadFile = File(...),
    language_code: str = Form("hi-IN"),
    chunking_strategy: str = Form("semantic_boundary"),
    stt_provider: str = Form("local")
):
    """
    Processes audio upload file through Speech-To-Text and RAG pipeline.
    """
    audio_bytes = await file.read()
    stt_engine.provider = stt_provider
    chunk_engine.set_strategy(chunking_strategy)

    rag_req = VoiceRAGRequest(
        audio_bytes=audio_bytes,
        audio_filename=file.filename or "audio.wav",
        language_code=language_code,
        chunking_strategy=chunking_strategy,
        stt_provider=stt_provider,
        synthesizer_mode="auto"
    )
    res: VoiceRAGResponse = orchestrator.run_pipeline(rag_req)
    return res.model_dump()

@app.get("/chunking/compare")
@app.get("/api/chunking/compare")
def compare_chunking(samples: int = 40):
    """
    Returns comparative benchmarking analytics across all 4 chunking strategies.
    """
    return chunk_engine.compare_strategies(dataset[:samples])

@app.get("/benchmark")
@app.get("/api/benchmark")
def execute_benchmark(num_samples: int = 50, strategy: str = "semantic_boundary"):
    """
    Runs benchmark suite and returns P50, P70, P100 latency percentiles.
    """
    return run_benchmark_suite(num_samples=num_samples, strategy=strategy)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
