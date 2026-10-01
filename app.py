import streamlit as st
import json
import asyncio
import edge_tts
import time
import os
import sqlite3
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

def delete_episode(ep_id: int, pdf_path: str, audio_path: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM episodes WHERE id = ?", (ep_id,))
    conn.commit()
    conn.close()
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)

def delete_music(music_path: str):
    if os.path.exists(music_path):
        os.remove(music_path)

# --- MODELOS DE DATOS ---
class DialogueTurn(BaseModel):
    speaker: str = Field(description="'ANA' o 'DANI'")
    text: str = Field(description="Intervención hablada de este turno")

class PodcastScript(BaseModel):
    title: str = Field(description="Título exacto del episodio")
    dialogue: List[DialogueTurn] = Field(description="Secuencia del podcast")

# --- PROMPT METODOLÓGICO CALIBRADO ---
SYSTEM_PROMPT = """
Eres el guionista principal del podcast científico 'Mente Digital'. 
Tu objetivo es analizar minuciosamente artículos empíricos de investigación y transformarlos en un guion dialogado riguroso, pausado, sobrio y de alto nivel metodológico entre dos conductores:

- ANA: Conductora e inquisidora metodológica. Hace preguntas incisivas, pide la 'versión corta', señala las implicaciones prácticas, cuestiona los p-valores en muestras grandes y frena los titulares sensacionalistas.
- DANI: Analista metodológico. Desmenuza la letra pequeña del paper: cita pruebas concretas (MANCOVA, regresiones, mediaciones Process), extrae estadísticos exactos (d de Cohen, betas, correlaciones, tamaños de muestra, eta cuadrado parcial), nombra las escalas psicométricas utilizadas, identifica solapamientos de ítems y desmonta interpretaciones causales precipitadas.

REGLAS DE FORMATO Y ESTILO:
1. Diálogo orgánico sin fórmulas teatrales: Prohibidas frases clichés como "¡Qué fascinante!", "¡Cuéntanos Dani!", "¡Así es, amigos!". El diálogo debe sonar a dos expertos dialogando con rigor y naturalidad.
2. Profundidad explicativa: Ana realiza intervenciones concisas y afiladas; Dani tiene espacio para desarrollar explicaciones completas cuando la estadística lo exija.
3. Precisión de microdatos: Extrae filtros de calidad exactos (tiempos mínimos de respuesta, preguntas de atención), asimetrías sociodemográficas entre grupos, distribuciones porcentuales de uso, instrumentos específicos de medida y modelos de mediación (distinguiendo mediación total, parcial o efectos de supresión).
4. El guion debe estructurarse obligatoriamente siguiendo estos 8 bloques:
   - Apertura (Bienvenida, ficha del estudio y versión corta).
   - Por qué este estudio (Hueco en la literatura, contexto cultural y delimitación conceptual del constructo medido).
   - Cómo se hizo (Muestra, cribado de calidad, tasas de respuesta, diferencias sociodemográficas de partida y definición operativa de usuario).
   - Comparación principal (Resultados estadísticos vs tamaños de efecto, muestra grande y correlación vs causalidad).
   - Desglose de motivos / Predictores (Conductas comunes vs minoritarias, análisis de regresión, pesos de predictores y análisis de solapamiento de ítems).
   - Modelos estadísticos avanzados (Mediaciones, si son parciales o totales, supresión estadística y el mecanismo teórico propuesto).
   - Límites (Costuras del estudio: medidas ultracortas, autoinforme, sesgos) y fortalezas objetivas.
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
            temp_name = f"temp_{idx}_{int(time.time())}.mp3"
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
tab_generator, tab_library, tab_player, tab_music = st.tabs([
    "🚀 Crear Episodio", 
    "📻 Biblioteca de Episodios",
    "🎧 Reproductor de Estudio (Podcast + Música)",
    "🎵 Gestor de Música"
])

# ==========================================
# PESTAÑA 1: CREAR EPISODIO
# ==========================================
with tab_generator:
    st.header("Generador de Podcasts Científicos")
    
    if not api_key:
        st.error("No se detectó 'GEMINI_API_KEY' en los Secrets de Streamlit.")
    
    uploaded_pdf = st.file_uploader("Sube el artículo académico (PDF)", type=["pdf"])

    if uploaded_pdf and api_key:
        if st.button("Generar y Guardar en Biblioteca"):
            timestamp = int(time.time())
            pdf_bytes = uploaded_pdf.read()
            pdf_filename = f"paper_{timestamp}.pdf"
            pdf_save_path = os.path.join(PDF_DIR, pdf_filename)
            
            with open(pdf_save_path, "wb") as f:
                f.write(pdf_bytes)

            try:
                with st.spinner("1/2: Analizando artículo completo y tablas estadísticas..."):
                    script_obj = generate_script_from_pdf(pdf_bytes, api_key)
                
                with st.spinner("2/2: Sintetizando voces de Ana y Dani..."):
                    audio_filename = f"podcast_{timestamp}.mp3"
                    audio_save_path = os.path.join(AUDIO_DIR, audio_filename)
                    asyncio.run(create_audio(script_obj.dialogue, audio_save_path))

                # Guardar en SQLite
                conn = get_db()
                c = conn.cursor()
                c.execute("""
                    INSERT INTO episodes (title, date, pdf_path, audio_path, transcript_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    script_obj.title,
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    pdf_save_path,
                    audio_save_path,
                    script_obj.model_dump_json()
                ))
                conn.commit()
                conn.close()

                st.success("¡Episodio generado y guardado en tu biblioteca!")
                st.subheader(script_obj.title)
                st.audio(audio_save_path, format="audio/mp3")
                
                with st.expander("Ver transcripción completa"):
                    for turn in script_obj.dialogue:
                        st.markdown(f"**{turn.speaker}:** {turn.text}")

            except Exception as e:
                st.error(f"Error durante el proceso: {e}")

# ==========================================
# PESTAÑA 2: BIBLIOTECA DE EPISODIOS
# ==========================================
with tab_library:
    st.header("Episodios Guardados")
    conn = get_db()
    episodes = conn.execute("SELECT id, title, date, pdf_path, audio_path, transcript_json FROM episodes ORDER BY id DESC").fetchall()
    conn.close()

    if not episodes:
        st.info("Aún no tienes episodios guardados en la biblioteca.")
    else:
        for ep_id, ep_title, ep_date, ep_pdf, ep_audio, ep_json in episodes:
            with st.container():
                col_head, col_del = st.columns([5, 1])
                with col_head:
                    st.subheader(ep_title)
                    st.caption(f"Generado el: {ep_date}")
                with col_del:
                    if st.button("🗑️ Eliminar", key=f"del_btn_{ep_id}"):
                        delete_episode(ep_id, ep_pdf, ep_audio)
                        st.warning(f"Episodio eliminado.")
                        st.rerun()

                if os.path.exists(ep_audio):
                    st.audio(ep_audio, format="audio/mp3")
                    with open(ep_audio, "rb") as af:
                        st.download_button(
                            label="⬇️ Descargar Episodio en MP3",
                            data=af.read(),
                            file_name=os.path.basename(ep_audio),
                            mime="audio/mpeg",
                            key=f"dl_audio_{ep_id}"
                        )

                with st.expander("📄 Ver Transcripción Completa"):
                    dialogue_data = json.loads(ep_json)
                    for turn in dialogue_data.get("dialogue", []):
                        st.markdown(f"**{turn['speaker']}:** {turn['text']}")

                if os.path.exists(ep_pdf):
                    with open(ep_pdf, "rb") as pf:
                        st.download_button(
                            label="📑 Descargar Paper Original (PDF)",
                            data=pf.read(),
                            file_name=os.path.basename(ep_pdf),
                            mime="application/pdf",
                            key=f"dl_pdf_{ep_id}"
                        )
                st.divider()

# ==========================================
# PESTAÑA 3: REPRODUCTOR COMBINADO
# ==========================================
with tab_player:
    st.header("🎧 Estudio de Reproducción Combinada")
    st.write("Escucha cualquier episodio de tu biblioteca junto a una pista de fondo musical.")

    conn = get_db()
    episodes = conn.execute("SELECT id, title, audio_path FROM episodes ORDER BY id DESC").fetchall()
    conn.close()

    saved_tracks = [f for f in os.listdir(MUSIC_DIR) if f.lower().endswith(".mp3")]

    if not episodes:
        st.warning("No hay episodios disponibles. Genera uno en la primera pestaña.")
    elif not saved_tracks:
        st.warning("No hay pistas de música subidas. Sube una en el 'Gestor de Música'.")
    else:
        ep_dict = {f"[{ep[0]}] {ep[1]}": ep[2] for ep in episodes}
        selected_ep_label = st.selectbox("Selecciona el Episodio:", list(ep_dict.keys()))
        selected_track = st.selectbox("Selecciona la Música de Fondo:", saved_tracks)

        st.divider()

        col_pod, col_bgm = st.columns(2)
        with col_pod:
            st.markdown("### 🎙️ Voz del Podcast")
            st.write(f"**Episodio:** {selected_ep_label}")
            ep_audio_file = ep_dict[selected_ep_label]
            if os.path.exists(ep_audio_file):
                st.audio(ep_audio_file, format="audio/mp3")

        with col_bgm:
            st.markdown("### 🎵 Música de Fondo")
            st.write(f"**Pista:** {selected_track}")
            track_audio_file = os.path.join(MUSIC_DIR, selected_track)
            if os.path.exists(track_audio_file):
                st.audio(track_audio_file, format="audio/mp3")

# ==========================================
# PESTAÑA 4: GESTOR DE MÚSICA
# ==========================================
with tab_music:
    st.header("Pistas de Música de Fondo")
    uploaded_music = st.file_uploader("Selecciona un archivo MP3 para añadir a la colección", type=["mp3"], key="uploader_music")
    if uploaded_music is not None:
        target_path = os.path.join(MUSIC_DIR, uploaded_music.name)
        if not os.path.exists(target_path):
            with open(target_path, "wb") as f:
                f.write(uploaded_music.getbuffer())
            st.success(f"Pista guardada: {uploaded_music.name}")
            st.rerun()

    st.subheader("Tu Colección Musical")
    saved_tracks = [f for f in os.listdir(MUSIC_DIR) if f.lower().endswith(".mp3")]
    
    if saved_tracks:
        for track in saved_tracks:
            track_path = os.path.join(MUSIC_DIR, track)
            col_info, col_player, col_del = st.columns([2, 3, 1])
            with col_info:
                st.write(f"🎵 **{track}**")
            with col_player:
                st.audio(track_path, format="audio/mp3")
            with col_del:
                if st.button("🗑️", key=f"del_track_{track}"):
                    delete_music(track_path)
                    st.rerun()
            st.divider()
    else:
        st.info("No hay pistas de música subidas.")
