import streamlit as st

# Debe ser estrictamente la primera instrucción
st.set_page_config(page_title="Mente Digital", layout="wide", page_icon="🎙")

import json
import asyncio
import edge_tts
import time
import os
import base64
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List

# Importación del SDK oficial de Google GenAI
try:
    from google import genai
    from google.genai import types
except Exception as e:
    st.error(f"Error cargando SDK de Google: {e}")

# --- DIRECTORIOS LOCALES ---
DATA_DIR = "library_data"
PDF_DIR = os.path.join(DATA_DIR, "pending_papers")
AUDIO_DIR = os.path.join(DATA_DIR, "audios")
MUSIC_DIR = os.path.join(DATA_DIR, "music")
INDEX_FILE = os.path.join(DATA_DIR, "library_index.json")

for d in [DATA_DIR, PDF_DIR, AUDIO_DIR, MUSIC_DIR]:
    os.makedirs(d, exist_ok=True)

# --- GESTOR JSON (CERO BLOQUEOS DE BASE DE DATOS) ---
def load_episodes() -> list:
    if not os.path.exists(INDEX_FILE):
        return []
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_episodes(episodes: list):
    try:
        temp_file = INDEX_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(episodes, f, ensure_ascii=False, indent=2)
        os.replace(temp_file, INDEX_FILE)
    except Exception as e:
        st.error(f"Error guardando índice: {e}")

def add_episode_record(title: str, pdf_path: str, audio_path: str, transcript_json: str):
    episodes = load_episodes()
    new_ep = {
        "id": int(time.time() * 1000),
        "title": title,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "pdf_path": pdf_path,
        "audio_path": audio_path,
        "transcript_json": transcript_json
    }
    episodes.insert(0, new_ep)
    save_episodes(episodes)

def delete_episode_record(ep_id: int):
    episodes = load_episodes()
    remaining = []
    for ep in episodes:
        if ep["id"] == ep_id:
            if os.path.exists(ep.get("audio_path", "")):
                try:
                    os.remove(ep["audio_path"])
                except Exception:
                    pass
        else:
            remaining.append(ep)
    save_episodes(remaining)

# --- MODELOS DE DATOS ---
class DialogueTurn(BaseModel):
    speaker: str = Field(description="'ANA' o 'DANI'")
    text: str = Field(description="Intervención hablada de este turno")

class PodcastScript(BaseModel):
    title: str = Field(description="Título exacto del episodio")
    dialogue: List[DialogueTurn] = Field(description="Secuencia del podcast")

# --- PROMPT CALIBRADO: MENOS CIFRAS EN CRUDO, MÁS TRADUCCIÓN INTERPRETATIVA ---
SYSTEM_PROMPT = """
Eres el guionista principal del podcast científico 'Mente Digital'. 
Tu cometido es analizar minuciosamente un artículo empírico de investigación y transformarlo en un guion dialogado riguroso, pausado, sobrio y natural entre dos conductores:

- ANA: Conductora e inquisidora metodológica. Hace preguntas incisivas, directas y breves. Pide la 'versión corta', señala las implicaciones prácticas, cuestiona los titulares sensacionalistas y frena a Dani cuando la estadística se vuelve abstracta para pedirle qué significa en la práctica.
- DANI: Analista metodológico. Lee la letra pequeña del paper. Tiene espacio para explicarse con calma, pero NO es una calculadora parlante: utiliza los números justos y necesarios como anclas (tamaños de muestra, edades clave, porcentajes ilustrativos, rangos de escala) e inmediatamente TRADUCE qué significan esos datos en la vida real y en la interpretación clínica de los autores.

REGLAS DE ORO DE ESTILO (CRÍTICO PARA FORMATO AUDIO):
1. Menos fórmulas, más traducción conceptual: PROHIBIDO recitar estadísticos brutos de tabla como grados de libertad F(x, y), trazas de Hotelling, intervalos de confianza numéricos o fórmulas de regresión complejas. En audio eso no se procesa. Si hay una regresión o una mediación, Dani explica la historia detrás del dato: qué variable influye sobre cuál, si una anula a la otra (efecto supresión), si la mediación es total o parcial, y qué hipótesis teórica proponen los autores para explicarlo.
2. Los números como contraste de realidad: Cita únicamente los datos que aportan perspectiva al oyente (ejemplo: si la escala va de 6 a 30 y la media fue 8,6, Dani destaca que la gente puntúa muy bajo y que no hay epidemia; si el 77% nunca usa el bot para evadirse, destaca que el confidente emocional es una minoría).
3. Diálogo humano y reposado: Ana hace preguntas afiladas y concisas; Dani desarrolla párrafos explicativos fluidos, pedagógicos y conversacionales, sin muletillas teatrales de locutor barato ("¡Qué fascinante!", "¡Cuéntanos más!").
4. El guion debe estructurarse obligatoriamente siguiendo estos 8 bloques:
   - Apertura (Bienvenida, ficha del estudio y versión corta).
   - Por qué este estudio (Hueco en la literatura, contexto cultural y delimitación conceptual del constructo medido).
   - Cómo se hizo (Muestra, cribado de calidad, tasas de respuesta, diferencias sociodemográficas de partida y definición operativa de usuario/participante).
   - Comparación o hallazgos principales (Diferencias observadas, tamaños de efecto explicados en lenguaje común, muestra masiva y correlación vs causalidad).
   - Desglose de motivos / Predictores (Conductas comunes vs minoritarias, qué pesa más en la regresión y análisis de solapamiento de preguntas).
   - Modelos estadísticos avanzados o interpretaciones profundas (Explicación conceptual de mediaciones y el mecanismo teórico propuesto).
   - Límites (Costuras del estudio: medidas ultracortas, autoinforme, sesgos, representatividad) y fortalezas objetivas.
   - Qué podemos llevarnos (Titular a evitar en la prensa, aplicación práctica prudente y cierre ético de salud mental).
"""

def generate_script_from_pdf(pdf_bytes: bytes, api_key: str) -> PodcastScript:
    client = genai.Client(api_key=api_key.strip())
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Analiza el documento PDF adjunto. Extrae minuciosamente todos sus datos empíricos, tablas y detalles "
        "metodológicos reales, y redacta el diálogo completo entre ANA y DANI reproduciendo exactamente el mismo "
        "nivel de profundidad pedagógica, traducción de resultados numéricos y rigor conceptual.\n\n"
        "Devuelve únicamente el bloque JSON con las claves 'title' y 'dialogue' (con lista de turnos 'speaker' y 'text')."
    )

    candidate_models = ["gemini-3.0-pro", "gemini-3.8-flash", "gemini-3.5-flash-lite"]
    last_error = None

    for model_name in candidate_models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
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
                time.sleep(2)
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
                try:
                    os.remove(tf)
                except Exception:
                    pass

# --- API KEY ---
api_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

# --- NAVEGACIÓN ---
tab_papers, tab_library, tab_mixer, tab_music = st.tabs([
    "📥 Mis Papers (Subir y Procesar)", 
    "📻 Biblioteca de Episodios",
    "🎛️️ Mezclador de Estudio (Móvil & PC)",
    "🎵 Gestor de Música"
])

# ==========================================
# PESTAÑA 1: GESTOR DE PAPERS
# ==========================================
with tab_papers:
    st.header("Almacén de Artículos Científicos")
    
    if not api_key:
        st.warning("Configura tu 'GEMINI_API_KEY' en los Secrets de Streamlit.")

    uploaded_files = st.file_uploader(
        "Sube uno o varios archivos PDF", 
        type=["pdf"], 
        accept_multiple_files=True, 
        key="pdf_uploader_main"
    )

    if uploaded_files:
        if st.button("💾 Guardar Archivos en el Almacén", type="primary"):
            guardados = 0
            for uf in uploaded_files:
                destino = os.path.join(PDF_DIR, uf.name)
                with open(destino, "wb") as f:
                    f.write(uf.getbuffer())
                guardados += 1
            st.success(f"Se han guardado {guardados} archivo(s) en el almacén.")
            st.rerun()

    st.divider()

    archivos_pdf = []
    if os.path.exists(PDF_DIR):
        archivos_pdf = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    col_tit, col_btn_vaciar = st.columns([4, 1.5])
    with col_tit:
        st.subheader("📚 Artículos Listos para Procesar")
    with col_btn_vaciar:
        if archivos_pdf:
            if st.button("🗑️ Vaciar Todo el Almacén", key="btn_vaciar_todos"):
                for f in archivos_pdf:
                    try:
                        os.remove(os.path.join(PDF_DIR, f))
                    except Exception:
                        pass
                st.warning("Almacén vaciado.")
                st.rerun()

    if not archivos_pdf:
        st.info("No tienes artículos pendientes. Arrastra tus PDFs arriba para guardarlos.")
    else:
        for nombre_pdf in archivos_pdf:
            ruta_pdf = os.path.join(PDF_DIR, nombre_pdf)
            with st.container():
                col_info, col_generar, col_borrar = st.columns([3, 1.4, 0.8])
                with col_info:
                    st.write(f"📄 **{nombre_pdf}**")
                with col_generar:
                    btn_disabled = not api_key
                    if st.button("🎙️ Generar Podcast", key=f"btn_gen_{nombre_pdf}", disabled=btn_disabled):
                        with open(ruta_pdf, "rb") as f_pdf:
                            pdf_bytes = f_pdf.read()

                        try:
                            with st.spinner("1/2: Analizando artículo y tablas estadísticas..."):
                                script_obj = generate_script_from_pdf(pdf_bytes, api_key)

                            with st.spinner("2/2: Sintetizando voces de Ana y Dani..."):
                                audio_filename = f"podcast_{int(time.time() * 1000)}.mp3"
                                audio_save_path = os.path.join(AUDIO_DIR, audio_filename)
                                asyncio.run(create_audio(script_obj.dialogue, audio_save_path))

                            add_episode_record(
                                title=script_obj.title,
                                pdf_path=ruta_pdf,
                                audio_path=audio_save_path,
                                transcript_json=script_obj.model_dump_json()
                            )

                            st.success(f"¡Episodio '{script_obj.title}' creado! Ve a la 'Biblioteca de Episodios'.")
                            st.audio(audio_save_path, format="audio/mp3")
                        except Exception as e:
                            st.error(f"Error procesando {nombre_pdf}: {e}")
                with col_borrar:
                    if st.button("🗑", key=f"btn_del_{nombre_pdf}"):
                        try:
                            os.remove(ruta_pdf)
                        except Exception:
                            pass
                        st.rerun()
                st.divider()

# ==========================================
# PESTAÑA 2: BIBLIOTECA DE EPISODIOS
# ==========================================
with tab_library:
    st.header("Episodios Generados")
    episodes = load_episodes()

    if not episodes:
        st.info("Aún no tienes episodios generados en la biblioteca.")
    else:
        for ep in episodes:
            ep_id = ep["id"]
            ep_title = ep.get("title", "Sin título")
            ep_date = ep.get("date", "")
            ep_pdf = ep.get("pdf_path", "")
            ep_audio = ep.get("audio_path", "")
            ep_json = ep.get("transcript_json", "{}")

            with st.container():
                col_head, col_del = st.columns([5, 1])
                with col_head:
                    st.subheader(ep_title)
                    st.caption(f"Generado el: {ep_date}")
                with col_del:
                    if st.button("🗑 Eliminar", key=f"del_ep_{ep_id}"):
                        delete_episode_record(ep_id)
                        st.warning("Episodio eliminado.")
                        st.rerun()

                if os.path.exists(ep_audio):
                    st.audio(ep_audio, format="audio/mp3")
                    with open(ep_audio, "rb") as af:
                        st.download_button(
                            label="⬇ Descargar Episodio en MP3",
                            data=af.read(),
                            file_name=os.path.basename(ep_audio),
                            mime="audio/mpeg",
                            key=f"dl_audio_{ep_id}"
                        )

                with st.expander("📄 Ver Transcripción Completa"):
                    try:
                        dialogue_data = json.loads(ep_json)
                        for turn in dialogue_data.get("dialogue", []):
                            st.markdown(f"**{turn['speaker']}:** {turn['text']}")
                    except Exception:
                        st.write("Transcripción no disponible.")

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
# PESTAÑA 3: MEZCLADOR INDEPENDIENTE (MÓVIL & PC)
# ==========================================
with tab_mixer:
    st.header("🎛 Mezclador de Estudio con Control de Volumen Individual")
    st.write("Ajusta de forma táctil el volumen de la voz y de la música de fondo de manera independiente.")

    episodes = load_episodes()
    saved_tracks = []
    if os.path.exists(MUSIC_DIR):
        saved_tracks = [f for f in os.listdir(MUSIC_DIR) if f.lower().endswith(".mp3")]

    if not episodes:
        st.warning("No hay episodios generados. Genera uno primero en la pestaña 'Mis Papers'.")
    elif not saved_tracks:
        st.warning("No hay pistas de música de fondo. Sube alguna en el 'Gestor de Música'.")
    else:
        ep_dict = {f"[{ep['id']}] {ep['title']}": ep['audio_path'] for ep in episodes}
        selected_ep_label = st.selectbox("1. Selecciona el Episodio:", list(ep_dict.keys()), key="mix_ep")
        selected_track = st.selectbox("2. Selecciona la Música de Fondo:", saved_tracks, key="mix_bgm")

        ep_audio_file = ep_dict[selected_ep_label]
        track_audio_file = os.path.join(MUSIC_DIR, selected_track)

        if os.path.exists(ep_audio_file) and os.path.exists(track_audio_file):
            with open(ep_audio_file, "rb") as f_ep:
                b64_podcast = base64.b64encode(f_ep.read()).decode()
            with open(track_audio_file, "rb") as f_bg:
                b64_music = base64.b64encode(f_bg.read()).decode()

            mixer_markup = f"""
            <div style="background-color: #1a1c24; padding: 20px; border-radius: 12px; color: #ffffff; font-family: sans-serif;">
                <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                    <button id="btnPlayAll" style="flex: 1; padding: 14px; font-size: 16px; font-weight: bold; background-color: #00c853; color: white; border: none; border-radius: 8px; cursor: pointer;">▶ Reproducir Ambos</button>
                    <button id="btnPauseAll" style="flex: 1; padding: 14px; font-size: 16px; font-weight: bold; background-color: #d50000; color: white; border: none; border-radius: 8px; cursor: pointer;">⏸ Pausar</button>
                    <button id="btnRestartAll" style="flex: 0.6; padding: 14px; font-size: 16px; font-weight: bold; background-color: #424242; color: white; border: none; border-radius: 8px; cursor: pointer;">⏮ Inicio</button>
                </div>

                <div style="background: #262936; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-weight: bold;">🎙️ Volumen Podcast (Voz)</span>
                        <span id="txtPodVol" style="color: #00e5ff;">100%</span>
                    </div>
                    <input type="range" id="sliderPod" min="0" max="1" step="0.01" value="1.0" style="width: 100%; height: 10px; accent-color: #00e5ff; cursor: pointer;">
                    <audio id="audioPodcast" src="data:audio/mp3;base64,{b64_podcast}" controls style="width: 100%; margin-top: 10px;"></audio>
                </div>

                <div style="background: #262936; padding: 15px; border-radius: 8px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-weight: bold;">🎵 Volumen Música de Fondo</span>
                        <span id="txtBgVol" style="color: #ff4081;">20%</span>
                    </div>
                    <input type="range" id="sliderBg" min="0" max="1" step="0.01" value="0.20" style="width: 100%; height: 10px; accent-color: #ff4081; cursor: pointer;">
                    <audio id="audioMusic" src="data:audio/mp3;base64,{b64_music}" loop controls style="width: 100%; margin-top: 10px;"></audio>
                </div>
            </div>

            <script>
                const pod = document.getElementById('audioPodcast');
                const bgm = document.getElementById('audioMusic');
                const sPod = document.getElementById('sliderPod');
                const sBg = document.getElementById('sliderBg');
                const tPod = document.getElementById('txtPodVol');
                const tBg = document.getElementById('txtBgVol');

                pod.volume = 1.0;
                bgm.volume = 0.20;

                sPod.oninput = function() {{
                    pod.volume = parseFloat(this.value);
                    tPod.innerText = Math.round(this.value * 100) + '%';
                }};

                sBg.oninput = function() {{
                    bgm.volume = parseFloat(this.value);
                    tBg.innerText = Math.round(this.value * 100) + '%';
                }};

                document.getElementById('btnPlayAll').onclick = function() {{
                    pod.play();
                    bgm.play();
                }};

                document.getElementById('btnPauseAll').onclick = function() {{
                    pod.pause();
                    bgm.pause();
                }};

                document.getElementById('btnRestartAll').onclick = function() {{
                    pod.currentTime = 0;
                    bgm.currentTime = 0;
                    pod.play();
                    bgm.play();
                }};
            </script>
            """
            st.html(mixer_markup)

# ==========================================
# PESTAÑA 4: GESTOR DE MÚSICA
# ==========================================
with tab_music:
    st.header("Pistas de Música de Fondo")
    uploaded_music = st.file_uploader("Subir nueva pista MP3", type=["mp3"], key="uploader_music")
    if uploaded_music is not None:
        target_path = os.path.join(MUSIC_DIR, uploaded_music.name)
        if not os.path.exists(target_path):
            with open(target_path, "wb") as f:
                f.write(uploaded_music.getbuffer())
            st.success(f"Pista guardada: {uploaded_music.name}")
            st.rerun()

    st.subheader("Tu Colección Musical")
    saved_tracks = []
    if os.path.exists(MUSIC_DIR):
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
                    if os.path.exists(track_path):
                        try:
                            os.remove(track_path)
                        except Exception:
                            pass
                    st.rerun()
            st.divider()
    else:
        st.info("No hay pistas de música subidas.")
