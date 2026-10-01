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

# --- MODELOS DE DATOS ---
class DialogueTurn(BaseModel):
    speaker: str = Field(description="'ANA' o 'DANI'")
    text: str = Field(description="Intervención hablada de este turno")

class PodcastScript(BaseModel):
    title: str = Field(description="Título exacto del episodio")
    dialogue: List[DialogueTurn] = Field(description="Secuencia del podcast")

# --- PROMPT CALIBRADO CON EL ESTILO DEL EJEMPLO 2 ---
SYSTEM_PROMPT = """
Eres el guionista principal del podcast científico 'Mente Digital'. 
Tu objetivo es analizar un artículo científico empírico y convertirlo en un diálogo divulgativo, riguroso, pausado y profundamente analítico entre:
- ANA: Conduce el programa. Hace preguntas inteligentes, pide la 'versión corta', señala las implicaciones prácticas y resume los puntos clave con sentido común.
- DANI: Analista metodológico. Lee la letra pequeña, desglosa la muestra, contextualiza los datos (explica qué significa un tamaño de efecto pequeño, por qué una muestra grande influye en la significación, advierte de solapamientos entre preguntas y desmitifica los titulares sensacionalistas).

REGLAS DE FORMATO Y ESTILO:
1. Diálogo vivo tipo 'ping-pong': turnos cortos (máximo 2 a 4 frases por turno). Prohibidos los monólogos largos.
2. Minería exhaustiva de datos: extrae de las tablas los filtros de calidad (preguntas trampa, descartes), diferencias demográficas exactas (edad, empleo, etc.), los tamaños del efecto (d de Cohen, betas, etc.) y explica qué significan en la práctica.
3. Desglose analítico estructurado en los siguientes bloques:
   - Apertura (Bienvenida, ficha del estudio y la 'versión corta').
   - Por qué este estudio (Hueco que llena, contexto y definición conceptual de las variables).
   - Cómo se hizo (Muestra, cribado de calidad, comparabilidad de grupos y definición de usuario).
   - Comparación de grupos (Diferencias estadísticas vs tamaños del efecto, muestra grande y correlación vs causalidad).
   - Para qué se usa / Predictores (Frecuencias mayoritarias vs minoritarias, regresiones y solapamiento entre preguntas).
   - Modelos avanzados (Mediación o moderación, si es parcial/total o supresión, y mecanismo teórico propuesto).
   - Límites y fortalezas (Costuras del estudio y virtudes objetivas).
   - Qué podemos llevarnos (Titular a evitar en prensa, utilidad práctica y recordatorio de que la tecnología no sustituye el apoyo humano/profesional).

Toma como modelo absoluto de profundidad, tono y estructura este estándar de calidad:
ANA: ¡Hola y bienvenidos a Mente Digital! Hoy hablamos de algo que tienes abierto en otra pestaña: ChatGPT, Claude, Gemini... Dani, ¿qué estudio nos traes?
DANI: Uno publicado en Computers in Human Behavior por el equipo de Julia Brailovskaia en Alemania. Preguntaron a más de siete mil adultos si usan chatbots y cómo se sienten, y miraron qué se asocia con un uso problemático.
ANA: Y antes de nada, la versión corta.
DANI: Quienes usan chatbots puntúan algo más alto en soledad, depresión, ansiedad y estrés, y algo más bajo en satisfacción con la vida. Pero las diferencias son pequeñas y el estudio es de una sola foto en el tiempo, así que no dice qué causa qué.
"""

def generate_script_from_pdf(pdf_bytes: bytes, api_key: str) -> PodcastScript:
    client = genai.Client(api_key=api_key.strip())
    
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Analiza el documento PDF adjunto con el mismo nivel de detalle, ritmo conversacional y rigor metodológico "
        "mostrado en las instrucciones. Devuelve el resultado en formato JSON estricto con las claves 'title' y 'dialogue' "
        "(con lista de objetos conteniendo 'speaker' y 'text')."
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
                # Envío directo del PDF como documento nativo
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
tab_generator, tab_library, tab_music = st.tabs([
    "🚀 Crear Episodio", 
    "📻 Biblioteca de Episodios", 
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
                with st.spinner("1/2: Analizando artículo completo (lectura nativa de tablas y datos)..."):
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

                st.success("¡Episodio generado con el estándar analítico completo!")
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
        st.info("Aún no tienes episodios guardados.")
    else:
        for ep_id, ep_title, ep_date, ep_pdf, ep_audio, ep_json in episodes:
            with st.container():
                st.subheader(ep_title)
                st.caption(f"Generado el: {ep_date}")
                
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
# PESTAÑA 3: GESTOR DE MÚSICA
# ==========================================
with tab_music:
    st.header("Pistas de Música de Fondo")
    uploaded_music = st.file_uploader("Selecciona un archivo MP3", type=["mp3"], key="uploader_music")
    if uploaded_music is not None:
        target_path = os.path.join(MUSIC_DIR, uploaded_music.name)
        if not os.path.exists(target_path):
            with open(target_path, "wb") as f:
                f.write(uploaded_music.getbuffer())
            st.success(f"Pista guardada: {uploaded_music.name}")

    st.subheader("Tu Colección Musical")
    saved_tracks = [f for f in os.listdir(MUSIC_DIR) if f.lower().endswith(".mp3")]
    
    if saved_tracks:
        for track in saved_tracks:
            track_path = os.path.join(MUSIC_DIR, track)
            col_info, col_player = st.columns([1, 2])
            with col_info:
                st.write(f"🎵 **{track}**")
            with col_player:
                st.audio(track_path, format="audio/mp3")
            st.divider()
    else:
        st.info("No hay pistas de música subidas.")
