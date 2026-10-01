import streamlit as st
import fitz  # PyMuPDF
import json
import asyncio
import edge_tts
import time
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import os

st.set_page_config(page_title="Mente Digital - Podcast", layout="centered")

# Estructura del guion
class DialogueTurn(BaseModel):
    speaker: str = Field(description="Nombre del hablante: 'ANA' o 'DANI'")
    text: str = Field(description="Intervención hablada de este turno")

class PodcastScript(BaseModel):
    title: str = Field(description="Título riguroso del episodio")
    dialogue: List[DialogueTurn] = Field(description="Secuencia ordenada del diálogo")

SYSTEM_PROMPT = """
Eres un guionista científico experto para el podcast 'Mente Digital'. 
Tu objetivo es transformar artículos académicos en una conversación amena, fluida pero estrictamente rigurosa y fiel a los datos del artículo, protagonizada por dos conductores:
- ANA: Conduce el programa, hace las preguntas clave que se haría el oyente y plantea objeciones lógicas.
- DANI: Analista metodológico que ha leído el artículo, explica los datos, matiza los hallazgos y frena las conclusiones apresuradas.

REGLAS OBLIGATORIAS:
1. RIGOR ESTADÍSTICO: No confundas correlación o asociación con causalidad. Si el diseño es transversal o correlacional, Dani debe recalcarlo.
2. TAMAÑOS DE EFECTO: Menciona tamaños de efecto (d de Cohen, betas, OR, r) y matiza si los efectos son pequeños, moderados o si alcanzan relevancia clínica.
3. LIMITACIONES: No omitas sesgos de selección, tipos de medida o limitaciones de las escalas.
4. HONESTIDAD: No inventes construcciones que el paper no midió.
5. CIERRE: Recuerda siempre que ante problemas reales de salud mental, el apoyo profesional y humano es prioritario.
"""

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def generate_script(pdf_text: str, api_key: str) -> PodcastScript:
    client = genai.Client(api_key=api_key.strip())
    cleaned = pdf_text[:80000]
    
    prompt = f"""
{SYSTEM_PROMPT}

A continuación tienes el texto del artículo científico:
{cleaned}

Genera el guion estructurado estrictamente en formato JSON con las claves:
- "title": (string con el título del episodio)
- "dialogue": lista de objetos con "speaker" ("ANA" o "DANI") y "text" (su intervención).
No incluyas explicaciones adicionales, devuelve únicamente el bloque JSON.
"""
    # Lista de modelos: si 3.8 está saturado (503), recurre a 3.5-flash-lite
    candidate_models = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    last_error = None

    for model_name in candidate_models:
        # Hasta 3 reintentos por modelo ante picos de demanda (503)
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
                # Si es saturación (503 UNAVAILABLE), esperar brevemente y reintentar
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
            temp_name = f"temp_{idx}.mp3"
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

# Interfaz visual
st.title("🎙 Mente Digital")
st.caption("Transforma artículos científicos en podcasts dialogados y rigurosos")

api_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

user_key = st.text_input("Gemini API Key (déjalo vacío si ya lo pusiste en Secrets):", type="password")
if user_key:
    api_key = user_key

pdf_file = st.file_uploader("Sube el artículo en PDF", type=["pdf"])

if "script" not in st.session_state:
    st.session_state.script = None
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

if pdf_file and api_key:
    if st.button("Generar Podcast"):
        try:
            with st.spinner("Leyendo artículo y redactando guion..."):
                text = extract_text_from_pdf(pdf_file.read())
                st.session_state.script = generate_script(text, api_key)
            
            with st.spinner("Sintetizando voces de Ana y Dani..."):
                audio_path = "podcast_generado.mp3"
                asyncio.run(create_audio(st.session_state.script.dialogue, audio_path))
                st.session_state.audio_path = audio_path
            st.success("¡Podcast generado con éxito!")
        except Exception as e:
            st.error(f"Error al procesar: {e}")

elif not api_key:
    st.warning("Por favor, introduce tu Gemini API Key arriba o configúrala en los Secrets.")

if st.session_state.script:
    st.subheader(st.session_state.script.title)
    with st.expander("Ver transcripción completa"):
        for turn in st.session_state.script.dialogue:
            st.markdown(f"**{turn.speaker}:** {turn.text}")

st.divider()
st.subheader("Reproductor y Ajustes de Audio")

music_file = st.file_uploader("Sube música de fondo opcional (MP3)", type=["mp3"])

if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
    with open(st.session_state.audio_path, "rb") as f:
        audio_bytes = f.read()
    
    st.write("**Audio del Podcast:**")
    st.audio(audio_bytes, format="audio/mp3")
    
    st.download_button(
        label="Descargar Podcast en MP3",
        data=audio_bytes,
        file_name="mente_digital_episodio.mp3",
        mime="audio/mpeg"
    )

if music_file:
    st.write("**Música de fondo:**")
    st.audio(music_file.read(), format="audio/mp3")
