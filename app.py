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

def clear_all_uploaded_papers():
    conn = get_db()
    c = conn.cursor()
    papers = c.execute("SELECT filepath FROM uploaded_papers").fetchall()
    c.execute("DELETE FROM uploaded_papers")
    conn.commit()
    conn.close()
    for (p_path,) in papers:
        if os.path.exists(p_path):
            try:
                os.remove(p_path)
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
# PESTAÑA 1: GESTOR DE PAPERS (SUBIDA MÚLTIPLE Y ELIMINACIÓN)
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
                st.success(f"Se han guardado {saved_count} artículo(s) en tu almacén.")
            else:
                st.info("Los archivos ya se encontraban guardados en el almacén.")
            st.rerun()

    st.divider()
    
    conn = get_db()
    papers = conn.execute("SELECT id, filename, filepath, upload_date FROM uploaded_papers ORDER BY id DESC").fetchall()
    conn.close()

    col_title_papers, col_clear_papers = st.columns([4, 1.5])
    with col_title_papers:
        st.subheader("📚 Artículos Listos para Procesar")
    with col_clear_papers:
        if papers:
            if st.button("🗑️ Vaciar Todo el Almacén", key="btn_clear_all_papers"):
                clear_all_uploaded_papers()
                st.warning("Se han eliminado todos los artículos pendientes.")
                st.rerun()

    if not papers:
        st.info("No tienes ningún artículo guardado en el almacén todavía.")
    else:
        for p_id, p_name, p_path, p_date in papers:
            with st.container():
                col_txt, col_action, col_del = st.columns([3, 1.4, 0.9])
                with col_txt:
                    st.write(f"📄 **{p_name}**")
                    st.caption(f"Subido el: {p_date}")
                with col_action:
                    if st.button("🎙️ Generar Podcast", key=f"btn_gen_{p_id}", disabled=not api_key):
                        if not os.path.exists(p_path):
                            st.error("El archivo PDF no se encontró en el servidor.")
                        else:
                            with open(p_path, "rb") as f:
                                pdf_bytes = f.read()

                            try:
                                with st.spinner("1/2: Analizando artículo y tablas estadísticas..."):
                                    script_obj = generate_script_from_pdf(pdf_bytes, api_key)
                                
                                with st.spinner("2/2: Sintetizando voces de Ana y Dani..."):
                                    audio_filename = f"podcast_{int(time.time() * 1000)}.mp3"
                                    audio_save_path = os.path.join(AUDIO_DIR, audio_filename)
                                    asyncio.run(create_audio(script_obj.dialogue, audio_save_path))

                                conn = get_db()
                                c = conn.cursor()
                                c.execute("""
                                    INSERT INTO episodes (title, date, pdf_path, audio_path, transcript_json)
                                    VALUES (?, ?, ?, ?, ?)
                                """, (
                                    script_obj.title,
                                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                                    p_path,
                                    audio_save_path,
                                    script_obj.model_dump_json()
                                ))
                                conn.commit()
                                conn.close()

                                st.success(f"¡Episodio '{script_obj.title}' creado con éxito!")
                                st.audio(audio_save_path, format="audio/mp3")
                            except Exception as e:
                                st.error(f"Error procesando '{p_name}': {e}")
                with col_del:
                    if st.button("🗑️ Eliminar", key=f"btn_del_paper_{p_id}"):
                        delete_uploaded_paper(p_id, p_path)
                        st.info(f"Artículo '{p_name}' eliminado.")
                        st.rerun()
                st.divider()

# ==========================================
# PESTAÑA 2: BIBLIOTECA DE EPISODIOS
# ==========================================
with tab_library:
    st.header("Episodios Generados")
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
                        st.warning("Episodio eliminado.")
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
# PESTAÑA 3: MEZCLADOR INDEPENDIENTE (MÓVIL & PC)
# ==========================================
with tab_mixer:
    st.header("🎛️ Mezclador de Estudio con Control de Volumen Individual")
    st.write("Ajusta de forma táctil el volumen de la voz y de la música de fondo de manera independiente.")

    conn = get_db()
    episodes = conn.execute("SELECT id, title, audio_path FROM episodes ORDER BY id DESC").fetchall()
    conn.close()

    saved_tracks = [f for f in os.listdir(MUSIC_DIR) if f.lower().endswith(".mp3")]

    if not episodes:
        st.warning("No hay episodios generados. Elige un PDF en la primera pestaña para crear uno.")
    elif not saved_tracks:
        st.warning("No hay pistas de música de fondo. Sube alguna en el 'Gestor de Música'.")
    else:
        ep_dict = {f"[{ep[0]}] {ep[1]}": ep[2] for ep in episodes}
        selected_ep_label = st.selectbox("1. Selecciona el Episodio:", list(ep_dict.keys()), key="mix_ep")
        selected_track = st.selectbox("2. Selecciona la Música de Fondo:", saved_tracks, key="mix_bgm")

        ep_audio_file = ep_dict[selected_ep_label]
        track_audio_file = os.path.join(MUSIC_DIR, selected_track)

        if os.path.exists(ep_audio_file) and os.path.exists(track_audio_file):
            with open(ep_audio_file, "rb") as f_ep:
                b64_podcast = base64.b64encode(f_ep.read()).decode()
            with open(track_audio_file, "rb") as f_bg:
                b64_music = base64.b64encode(f_bg.read()).decode()

            mixer_html = f"""
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

                sPod.addEventListener('input', (e) => {{
                    pod.volume = parseFloat(e.target.value);
                    tPod.innerText = Math.round(e.target.value * 100) + '%';
                }});

                sBg.addEventListener('input', (e) => {{
                    bgm.volume = parseFloat(e.target.value);
                    tBg.innerText = Math.round(e.target.value * 100) + '%';
                }});

                document.getElementById('btnPlayAll').addEventListener('click', () => {{
                    pod.play();
                    bgm.play();
                }});

                document.getElementById('btnPauseAll').addEventListener('click', () => {{
                    pod.pause();
                    bgm.pause();
                }});

                document.getElementById('btnRestartAll').addEventListener('click', () => {{
                    pod.currentTime = 0;
                    bgm.currentTime = 0;
                    pod.play();
                    bgm.play();
                }});
            </script>
            """
            components.html(mixer_html, height=430)

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
