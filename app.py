import streamlit as st
import streamlit.components.v1 as components
import json
import asyncio
import edge_tts
import time
import os
import sqlite3
import base64
from datetime import datetime
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List

st.set_page_config(page_title="Mente Digital", layout="wide", page_icon="🎙")

# --- DIRECTORIOS Y BASE DE DATOS LOCAL ---
DATA_DIR = "library_data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
AUDIO_DIR = os.path.join(DATA_DIR, "audios")
MUSIC_DIR = os.path.join(DATA_DIR, "music")

for d in [PDF_DIR, AUDIO_DIR, MUSIC_DIR]:
    os.makedirs(d, exist_ok=True)

def get_db():
    conn = sqlite3.connect(os.path.join(DATA_DIR, "library.db"))
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            filepath TEXT,
            upload_date TEXT,
            status TEXT DEFAULT 'pending'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            date TEXT,
            pdf_path TEXT,
            audio_path TEXT,
            transcript_json TEXT
        )
    """)
    conn.commit()
    return conn

def delete_uploaded_paper(paper_id: int, filepath: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM uploaded_papers WHERE id = ?", (paper_id,))
    conn.commit()
    conn.close()
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception:
            pass

def delete_episode(ep_id: int, pdf_path: str, audio_path: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM episodes WHERE id = ?", (ep_id,))
    conn.commit()
    conn.close()
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except Exception:
            pass
    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except Exception:
            pass

def delete_music(music_path: str):
    if os.path.exists(music_path):
        try:
            os.remove(music_path)
        except Exception:
            pass

# --- MODELOS DE DATOS ---
class DialogueTurn(BaseModel):
    speaker: str = Field(description="'ANA' o 'DANI'")
    text: str = Field(description="Intervención hablada de este turno")

class PodcastScript(BaseModel):
    title: str = Field(description="Título exacto del episodio")
    dialogue: List[DialogueTurn] = Field(description="Secuencia del podcast")

SYSTEM_PROMPT = """
Eres el guionista principal del podcast científico 'Mente Digital'. 
Tu objetivo es analizar minuciosamente artículos empíricos de investigación y transformarlos en un guion dialogado riguroso, pausado, sobrio y de alto nivel metodológico entre dos conductores:

- ANA: Conductora e inquisidora metodológica. Hace preguntas incisivas, pide la 'versión corta', señala las implicaciones prácticas, cuestiona los p-valores en muestras grandes y frena los titulares sensacionalistas.
- DANI: Analista metodológico. Desmenuza la letra pequeña del paper: cita pruebas concretas (MANCOVA, regresiones, mediaciones Process, análisis cualitativo temático reflexivo), extrae estadísticos exactos (d de Cohen, betas, correlaciones, tamaños de muestra, eta cuadrado parcial), nombra las escalas psicométricas utilizadas, identifica solapamientos de ítems y desmonta interpretaciones causales precipitadas.

REGLAS DE FORMATO Y ESTILO:
1. Diálogo orgánico sin fórmulas teatrales: Prohibidas frases clichés como "¡Qué fascinante!", "¡Cuéntanos Dani!", "¡Así es, amigos!". El diálogo debe sonar a dos expertos dialogando con rigor y naturalidad.
2. Profundidad explicativa: Ana realiza intervenciones concisas y afiladas; Dani tiene espacio para desarrollar explicaciones completas cuando la metodología lo exija.
3. Precisión de microdatos: Extrae filtros de calidad exactos (tiempos mínimos de respuesta, preguntas de atención), asimetrías sociodemográficas entre grupos, distribuciones de uso, instrumentos específicos de medida y modelos de mediación o temas cualitativos.
4. El guion debe estructurarse obligatoriamente siguiendo estos 8 bloques:
   - Apertura (Bienvenida, ficha del estudio y versión corta).
   - Por qué este estudio (Hueco en la literatura, contexto cultural y delimitación conceptual del constructo medido).
   - Cómo se hizo (Muestra, cribado de calidad, tasas de respuesta, diferencias sociodemográficas de partida y definición operativa de usuario/participante).
   - Comparación o hallazgos principales (Resultados estadísticos vs tamaños de efecto / temas cualitativos con recuentos descriptivos).
   - Desglose de motivos / Predictores / Subtemas analíticos.
   - Modelos estadísticos avanzados o interpretaciones profundas.
   - Límites (Costuras del estudio: medidas ultracortas, autoinforme, sesgos, transferibilidad) y fortalezas objetivas.
   - Qué podemos llevarnos (Titular a evitar en la prensa, aplicación práctica prudente y cierre ético de salud mental).
"""

def generate_script_from_pdf(pdf_bytes: bytes, api_key: str) -> PodcastScript:
    client = genai.Client(api_key=api_key.strip())
    
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Analiza el documento PDF adjunto. Extrae minuciosamente todos sus datos empíricos, tablas y detalles "
        "metodológicos reales, y redacta el diálogo completo entre ANA y DANI reproduciendo exactamente el mismo "
        "nivel de detalle y rigor metodológico.\n\n"
        "Devuelve únicamente el bloque JSON con las claves 'title' y 'dialogue' (con lista de turnos 'speaker' y 'text')."
    )

    candidate_models = [
        "gemini-3.0-pro",
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite"
    ]
    last_error = None

    for model_name in candidate_models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(
                            data=pdf_bytes,
                            mime_type="application/pdf"
                        ),
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.3,
                    ),
                )
                raw_text = response.text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]
                
                script_dict = json.loads(raw_text.strip())
                return PodcastScript(**script_dict)
            except Exception as e:
                last_error = e
                time.sleep(3)
                continue

    raise last_error

async def create_audio(dialogue: List[DialogueTurn], output_file: str):
    temp_files = []
    voice_map = {"ANA": "es-ES-ElviraNeural", "DANI": "es-ES-AlvaroNeural"}
    try:
        for idx, turn in enumerate(dialogue):
            voice = voice_map.get(turn.speaker.upper(), "es-ES-AlvaroNeural")
            temp_name = f"temp_{idx}_{int(time.time() * 1000)}.mp3"
            comm = edge_tts.Communicate(turn.text, voice)
            await comm.save(temp_name)
            temp_files.append(temp_name)

        with open(output_file, "wb") as out:
            for tf in temp_files:
                with open(tf, "rb") as inf:
                    out.write(inf.read())
    finally:
        for tf in temp_files:
            if os.path.exists(tf):
                os.remove(tf)

# --- RECUPERAR API KEY ---
api_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

# --- INTERFAZ PRINCIPAL ---
tab_papers, tab_library, tab_mixer, tab_music = st.tabs([
    "📥 Mis Papers (Subir y Procesar)", 
    "📻 Biblioteca de Episodios",
    "🎛️ Mezclador de Estudio (Móvil & PC)",
    "🎵 Gestor de Música"
])

# ==========================================
# PESTAÑA 1: GESTOR DE PAPERS (SUBIDA MÚLTIPLE)
# ==========================================
with tab_papers:
    st.header("Almacén de Artículos Científicos")
    
    if not api_key:
        st.error("No se detectó 'GEMINI_API_KEY' en los Secrets de Streamlit.")

    st.write("Selecciona uno o varios PDFs y haz clic en **Guardar en Almacén** para dejarlos listos y procesarlos cuando quieras.")
    
    uploaded_files = st.file_uploader(
        "Arrastra o selecciona tus archivos PDF", 
        type=["pdf"], 
        accept_multiple_files=True, 
        key="batch_pdf_uploader"
    )

    if uploaded_files:
        if st.button("💾 Guardar Archivos Seleccionados en Almacén", type="primary"):
            conn = get_db()
            c = conn.cursor()
            saved_count = 0
            for uploaded_file in uploaded_files:
                # Comprobar si ya existe por nombre
                existing = c.execute("SELECT id FROM uploaded_papers WHERE filename = ?", (uploaded_file.name,)).fetchone()
                if not existing:
                    timestamp = int(time.time() * 1000)
                    safe_name = f"{timestamp}_{uploaded_file.name}"
                    pdf_path = os.path.join(PDF_DIR, safe_name)
                    
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    c.execute("""
                        INSERT INTO uploaded_papers (filename, filepath, upload_date, status)
                        VALUES (?, ?, ?, ?)
                    """, (uploaded_file.name, pdf_path, datetime.now().strftime("%Y-%m-%d %H:%M"), "ready"))
                    saved_count += 1
            conn.commit()
            conn.close()
            if saved_count > 0:
                st.success(f"Se han guardado {saved_count} artículo(s) nuevo(s) en tu almacén.")
            else:
                st.info("Los archivos seleccionados ya se encontraban en el almacén.")
            st.rerun()

    st.divider()
    st.subheader("📚 Artículos Listos para Procesar")
    
    conn = get_db()
    papers = conn.execute("SELECT id, filename, filepath, upload_date FROM uploaded_papers ORDER BY id DESC").fetchall()
    conn.close()

    if not papers:
        st.info("No tienes ningún artículo guardado en el almacén todavía.")
    else:
        for p_id, p_name, p_path, p_date in papers:
            with st.container():
                col_txt, col_action, col_del = st.columns([3, 1.5, 0.8])
                with col_txt:
                    st.write(f"📄 **{p_name}**")
                    st.caption(f"Subido el: {p_date}")
                with col_action:
                    if st.button("🎙️ Generar Podcast", key=f"btn_gen_{p_id}", disabled=not api_key):
                        if not os.path.exists(p_path):
                            st.error
