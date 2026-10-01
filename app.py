import streamlit as st
import fitz  # PyMuPDF
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

SYSTEM_PROMPT = """
Eres el guionista principal del podcast científico 'Mente Digital'. 
Tu objetivo es analizar un artículo científico empírico y convertirlo en un diálogo divulgativo, riguroso, pausado y profundamente analítico entre:
- ANA: Conduce el programa. Hace preguntas inteligentes, pide la 'versión corta', señala las implicaciones prácticas y resume los puntos clave con sentido común.
- DANI: Analista metodológico. Lee la letra pequeña, desglosa la muestra, contextualiza los datos (explica qué significa un tamaño de efecto pequeño, por qué una muestra grande influye en la significación, advierte de solapamientos entre preguntas y desmitifica los titulares sensacionalistas).

ESTRUCTURA OBLIGATORIA DEL DIÁLOGO (sigue este orden y ritmo):
1. Apertura: Bienvenida, presentación del tema actual, ficha del estudio (autores, revista, muestra general) y la 'versión corta' inicial sin tecnicismos exagerados.
2. Por qué este estudio: Qué hueco llena, en qué contexto geográfico/social se hace, y aclaración conceptual de las variables delicadas (ej. si 'adicción' o 'dependencia' se mide como continuo y no como diagnóstico clínico).
3. Cómo se hizo (Metodología): Procedimiento de recogida, filtros de calidad (preguntas trampa, tiempos de respuesta), tasa de respuesta, tamaño de grupos y diferencias sociodemográficas de partida (edad, ocupación, sesgos del panel). Definición operativa de qué se considera 'usuario'.
4. Comparaciones principales (Usuarios vs No usuarios): Diferencias estadísticas encontradas acompañadas obligatoriamente de sus tamaños del efecto (d de Cohen, etc.). Explicación de cómo las muestras grandes facilitan la significación estadística. Advertencia de correlación vs causalidad (diseño transversal).
5. Desglose detallado de motivos / predictores: Qué conductas son las más comunes y cuáles son minoritarias. Qué variables correlacionan más fuerte con el problema y posibles solapamientos metodológicos en las preguntas. Predictores estadísticos en regresión.
6. Modelos estadísticos avanzados (Mediación / Moderación si los hay): Explicación clara de si la variable puente explica total o parcialmente el efecto, patrones atípicos (supresión, mediaciones inconsistentes) y advertencia de que la mediación estadística en datos transversales no prueba causa. Mecanismo teórico propuesto por los autores.
7. Limitaciones y Fortalezas: Desglose punto por punto de las costuras del estudio (medidas ultracortas, autoinforme, sesgo de deseabilidad, representatividad) y sus méritos metodológicos.
8. Qué podemos llevarnos y Cierre: Aplicación práctica sensata, el titular que NO se debe dar a la prensa, recordatorio ético de que la tecnología o los tests no sustituyen el apoyo profesional/humano cualificado, y despedida.

ESTILO:
- Tono coloquial pero técnicamente impecable (hablado, natural, sin rodeos artificiales).
- No uses tecnicismos sin explicarlos brevemente (ej. si dices "supresión" o "d de Cohen", Dani lo aterriza con un ejemplo).
"""

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def generate_script(pdf_text: str, api_key: str) -> PodcastScript:
    client = genai.Client(api_key=api_key.strip())
    cleaned = pdf_text[:90000]
    
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "A continuación tienes el texto del artículo científico:\n"
        f"{cleaned}\n\n"
        "Genera el guion estructurado estrictamente en formato JSON con la siguiente estructura:\n"
        "- title: título del episodio\n"
        "- dialogue: lista de turnos donde cada turno tiene 'speaker' (ANA o DANI) y 'text' (su intervención)\n"
        "Devuelve únicamente el bloque JSON válido sin formato markdown ni texto adicional."
    )

    candidate_models = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    last_error = None

    for model_name in candidate_models:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
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
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    break
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

# --- OBTENCIÓN AUTOMÁTICA DE API KEY ---
api_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

# --- NAVEGACIÓN PRINCIPAL ---
tab_generator, tab_library = st.tabs(["🚀 Crear Nuevo Episodio", "📚 Biblioteca de Episodios y Música"])

# ==========================================
# PESTAÑA 1: GENERADOR
# ==========================================
with tab_generator:
    st.header("Generador de Podcasts Científicos")
    
    if not api_key:
        st.error("Falta configurar 'GEMINI_API_KEY' en los Secrets de Streamlit.")
    
    uploaded_pdf = st.file_uploader("Sube el artículo en PDF", type=["pdf"])

    if uploaded_pdf and api_key:
        if st.button("Generar y Guardar en Biblioteca"):
            timestamp = int(time.time())
            pdf_filename = f"paper_{timestamp}.pdf"
            pdf_save_path = os.path.join(PDF_DIR, pdf_filename)
            
            with open(pdf_save_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())

            try:
                with st.spinner("1/2: Analizando metodología y estructurando diálogo..."):
                    pdf_text = extract_text_from_pdf(open(pdf_save_path, "rb").read())
                    script_obj = generate_script(pdf_text, api_key)
                
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

                st.success("¡Episodio creado y guardado en tu biblioteca!")
                st.subheader(script_obj.title)
                st.audio(audio_save_path, format="audio/mp3")
                
                with st.expander("Ver transcripción completa"):
                    for turn in script_obj.dialogue:
                        st.markdown(f"**{turn.speaker}:** {turn.text}")

            except Exception as e:
                st.error(f"Error durante el proceso: {e}")

# ==========================================
# PESTAÑA 2: BIBLIOTECA
# ==========================================
with tab_library:
    st.header("Biblioteca Permanente")
    
    col_episodes, col_music = st.columns([2, 1])

    with col_episodes:
        st.subheader("📻 Episodios Guardados")
        conn = get_db()
        episodes = conn.execute("SELECT id, title, date, pdf_path, audio_path, transcript_json FROM episodes ORDER BY id DESC").fetchall()
        conn.close()

        if not episodes:
            st.info("Aún no tienes episodios guardados. Genera uno en la otra pestaña.")
        else:
            for ep_id, ep_title, ep_date, ep_pdf, ep_audio, ep_json in episodes:
                with st.container():
                    st.markdown(f"### {ep_title}")
                    st.caption(f"Fecha de creación: {ep_date}")
                    
                    if os.path.exists(ep_audio):
                        st.audio(ep_audio, format="audio/mp3")
                        with open(ep_audio, "rb") as af:
                            st.download_button(
                                label="⬇️ Descargar Audio MP3",
                                data=af.read(),
                                file_name=os.path.basename(ep_audio),
                                mime="audio/mpeg",
                                key=f"dl_audio_{ep_id}"
                            )

                    with st.expander("📄 Ver Transcripción"):
                        dialogue_data = json.loads(ep_json)
                        for turn in dialogue_data.get("dialogue", []):
                            st.markdown(f"**{turn['speaker']}:** {turn['text']}")

                    if os.path.exists(ep_pdf):
                        with open(ep_pdf, "rb") as pf:
                            st.download_button(
                                label="📑 Descargar PDF original",
                                data=pf.read(),
                                file_name=os.path.basename(ep_pdf),
                                mime="application/pdf",
                                key=f"dl_pdf_{ep_id}"
                            )
                    st.divider()

    with col_music:
        st.subheader("🎵 Pistas de Música")
        uploaded_music = st.file_uploader
