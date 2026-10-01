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

# --- PROMPT CALIBRADO CON EL EJEMPLO 2 COMO GUÍA ---
SYSTEM_PROMPT = """
Eres el guionista principal del podcast científico 'Mente Digital'. 
Tu cometido es analizar un artículo científico empírico y escribir un guion dialogado con la máxima profundidad metodológica, rigor estadístico y un tono divulgativo sobrio, inteligente y calmado entre dos conductores:

- ANA: Conduce el programa. Plantea preguntas lógicas, directas y breves. Pide 'la versión corta', cuestiona la significación estadística en muestras gigantes, pide aclarar los términos delicados y resume lo práctico.
- DANI: Analista metodológico. Lee la letra pequeña del artículo. Tiene espacio para explicarse con calma: desglosa las muestras, extrae porcentajes, cita tamaños de efecto exactos, explica la construcción de las escalas, advierte de solapamientos entre preguntas y desmitifica titulares.

REGLAS DE ESTILO Y RITMO:
1. Sin teatralidad forzada: Prohibidas las frases de relleno como "¡Qué descubrimiento tan fascinante!", "¡Cuéntanos Dani, no nos dejes con la intriga!" o "¡Así es, amigos!". La conversación debe sonar a dos profesionales dialogando con naturalidad y pausa.
2. Dinámica de intervención: Ana hace preguntas directas y concisas. Dani puede y debe explayarse (un párrafo explicativo sólido cuando los datos o la metodología lo requieran).
3. Minería quirúrgica de datos: Busca en el texto y en las tablas los filtros de calidad (preguntas trampa, descartes), asimetrías demográficas (edad, ocupación), medias reales de las escalas (para ver si la muestra está cerca del mínimo o del máximo), porcentajes de uso mayoritarios vs minoritarios, y modelos estadísticos (mediaciones parciales vs totales, efectos de supresión).
4. El guion debe estructurarse obligatoriamente siguiendo estos 8 bloques temáticos:
   - Apertura (Bienvenida, ficha del estudio y versión corta).
   - Por qué este estudio (Hueco en la literatura, contexto cultural y delimitación del constructo principal).
   - Cómo se hizo (Muestra, cribado de calidad, tasas de respuesta, diferencias sociodemográficas de partida y definición operativa de usuario).
   - Comparación principal (Resultados estadísticos vs tamaños de efecto, muestra grande y correlación vs causalidad).
   - Desglose de motivos / Predictores (Conductas comunes vs minoritarias, análisis de regresión, pesos de predictores y análisis de solapamiento de ítems).
   - Modelos estadísticos avanzados (Mediaciones, si son parciales o totales, supresión estadística y el mecanismo teórico propuesto).
   - Límites (Costuras del estudio: medidas ultracortas, autoinforme, sesgos) y fortalezas objetivas.
   - Qué podemos llevarnos (Titular a evitar en la prensa, aplicación práctica prudente y cierre ético de salud mental).

ESTE ES EL ESTÁNDAR EXACTO DE TONO, PROFUNDIDAD Y ESTRUCTURA QUE DEBES REPLICAR:
\"\"\"
Apertura
ANA: ¡Hola y bienvenidos a Mente Digital! Hoy hablamos de algo que, probablemente, tienes abierto en otra pestaña: ChatGPT, Claude, Gemini y compañía. Dani, ¿qué estudio nos traes?
DANI: Uno publicado en septiembre de 2026 en la revista Computers in Human Behavior. Lo firma Julia Brailovskaia con su equipo de la Ruhr-Universität Bochum, en Alemania. Preguntaron a más de siete mil adultos si usan chatbots de IA y cómo se sienten, y luego miraron qué se asocia con un uso "adictivo".
ANA: Y antes de nada, la versión corta.
DANI: Quienes usan chatbots puntúan algo más alto en soledad, depresión, ansiedad y estrés, y algo más bajo en satisfacción con la vida. Pero las diferencias son pequeñas, y el estudio es de una sola foto en el tiempo, así que no dice qué causa qué.
ANA: Pues vamos con calma, porque aquí los matices importan mucho.

Por qué este estudio
ANA: ¿Qué hueco quería llenar?
DANI: La mayoría de los estudios previos se habían hecho en Asia, sobre todo en China y Taiwán, y con gente joven o estudiantes. Aquí la muestra es de adultos en Alemania, un país donde, según los autores, cerca del 58 % de la población ya usa estas herramientas. Además, el Parlamento Europeo ha publicado advertencias sobre los riesgos de un uso intensivo.
ANA: Y hay una palabra delicada en el título: "adictivo".
DANI: Los autores la usan con mucho cuidado. Aclaran que no es un diagnóstico psiquiátrico reconocido. Por eso lo tratan como un continuo, de más a menos tendencia, y no como "adicto" o "no adicto". Para medirlo adaptaron una escala pensada para redes sociales, la de Bergen, que se basa en seis componentes: pensar constantemente en ello, necesitar cada vez más, usarlo para cambiar el ánimo, recaer tras intentar reducirlo, malestar cuando no se usa y conflictos con otras personas.

Cómo se hizo
ANA: ¿Cómo reunieron los datos?
DANI: Con una encuesta online entre septiembre y noviembre de 2025, a través de un panel alemán de salud mental. Invitaron a unas 22.000 personas. Tras descartar a quien respondía demasiado rápido o fallaba preguntas trampa, como "¿de qué color es un plátano?", quedaron 7.312 personas con datos completos. Eso es una tasa de respuesta de un 33 %.
ANA: ¿Y cómo se repartían?
DANI: Casi a mitades: 3.729 usuarios y 3.583 no usuarios. Y aquí viene el primer matiz: los usuarios eran bastante más jóvenes, 42 años de media frente a 54, y había más gente trabajando o estudiando y menos jubilados.
ANA: O sea, no son grupos comparables sin más.
DANI: Los autores controlaron estadísticamente la edad, el estado civil y la ocupación. Pero admiten que pueden quedar factores sin medir, como los ingresos, el nivel de estudios o el estrés laboral. Y otra cosa: el panel es voluntario y todavía no representa a toda la población alemana.
ANA: ¿Y qué es exactamente ser "usuario"?
DANI: Haber usado alguna vez un chatbot de texto, para trabajo o para uso personal. Eso incluye a quien lo usa varias veces al día, pero también a quien lo usa menos de una vez al mes: alrededor del 12 % en uso personal y del 29 % en uso laboral. Se dejaron fuera los asistentes de voz y los avatares.

Usuarios frente a no usuarios
ANA: Vamos a la primera pregunta. ¿Se nota la diferencia en la salud mental?
DANI: Sí, estadísticamente. Los usuarios puntuaron más alto en depresión, ansiedad, estrés y soledad, y más bajo en satisfacción con la vida. Pero los tamaños del efecto son pequeños. Medidos con la d de Cohen van de 0,14 en satisfacción con la vida a 0,41 en estrés.
ANA: Y con tanta gente, cualquier diferencia sale "significativa".
DANI: Justo. Con miles de participantes, hasta diferencias modestas dan significación. Y hay otra cautela: esas medidas son muy breves. Depresión, ansiedad y estrés se midieron con tres escalas ultracortas, y la soledad con una sola pregunta: cuántas veces te has sentido solo en las últimas dos semanas.
ANA: Y no sabemos quién va primero.
DANI: Exacto. Puede que usar chatbots influya en cómo uno se siente, o que quien ya se siente peor o más solo los use más, o ambas cosas. Los autores citan un estudio longitudinal, que siguió a gente durante un año, que apunta a una influencia en los dos sentidos. Este estudio, al ser transversal, no puede distinguirlo.

Para qué se usan
ANA: ¿Y para qué usa la gente estos chatbots?
DANI: Les propusieron quince motivos. Los más frecuentes son funcionales: buscar información e inspiración, revisar textos, redactar correos, traducir. Los más raros son los emocionales: buscar emociones positivas, entretenimiento y, el último de la lista, escapar de emociones negativas. De hecho, un 77,5 % dice que nunca lo usa para eso.
ANA: O sea, la imagen del chatbot como confidente es minoritaria.
DANI: Lo es. Y aun así, es el motivo que más se asocia con el uso adictivo. Entre las personas que lo usan para escapar de emociones negativas, la correlación con la escala de uso adictivo es de 0,70, la más alta de todas, y en la regresión es también el predictor más fuerte.
ANA: Suena casi demasiado alto.
DANI: Y hay una razón para ser prudentes. Uno de los ítems de la escala de uso adictivo pregunta si lo usas para olvidarte de tus problemas personales, que se parece mucho a ese motivo. Los autores lo comprobaron quitando ese ítem, y la correlación baja a 0,66. Sigue siendo la más alta, así que el resultado se sostiene, pero conviene saber que parte de la fuerza viene de ese solapamiento.
ANA: ¿Y los motivos más prácticos quedan libres de sospecha?
DANI: No del todo, y esto es importante. En la regresión, después del motivo emocional, aparecen programar y planificar u organizar como predictores significativos del uso adictivo. Son más débiles, pero no son cero. Y la frecuencia de uso laboral también se relaciona con la escala. Así que lo que se puede decir es que el motivo emocional destaca más, no que el resto sea inocuo. Además, el estudio no dice qué pasa en cada persona, solo asociaciones entre grupos.
ANA: ¿Y cuánto "uso adictivo" había en general?
DANI: Poco. La escala va de 6 a 30 y la media fue 8,67, cerca del mínimo. O sea, la mayoría de los usuarios puntúa bajo, y lo que describe el estudio son tendencias, no una epidemia.

El papel de la soledad
ANA: Llegamos a la parte estadísticamente más sofisticada: la mediación con la soledad.
DANI: Primero, lo básico. En los usuarios, el uso adictivo se correlaciona de forma pequeña con depresión, ansiedad y estrés, entre 0,08 y 0,21, y la soledad se correlaciona más fuerte, de forma moderada, con todos esos síntomas. Con eso, los autores probaron si la soledad "explica" parte de la relación entre uso adictivo y malestar.
ANA: ¿Y qué salió?
DANI: En depresión y ansiedad, la soledad explica aproximadamente la mitad de la asociación: queda un efecto directo que sigue siendo significativo. En estrés, en cambio, el efecto directo desaparece al incluir la soledad, así que ahí es una mediación total.
ANA: ¿Y en satisfacción con la vida?
DANI: Es el caso más raro. El uso adictivo y la satisfacción con la vida no se asociaban de forma significativa. Pero al meter la soledad, aparecen dos efectos que van en sentidos contrarios: uno directo, ligeramente positivo, y uno a través de la soledad, negativo. Se compensan casi exactamente. Es lo que en estadística se llama supresión. Y los propios autores avisan de que, siendo los efectos tan pequeños, hay que interpretarlo con mucha cautela.
ANA: Entonces no podemos decir que "el chatbot hunde la satisfacción".
DANI: No. Lo que se puede decir es que hay un patrón estadístico curioso que habrá que replicar. Y, en general, la palabra "mediación" aquí es estadística, no causal. Con datos de un solo momento, la soledad podría ser causa, consecuencia o ambas.
ANA: Pero los autores sí proponen una explicación.
DANI: La proponen como hipótesis, y la presentan así. La idea, tomada de modelos sobre redes sociales, es un círculo: alguien se siente mal, se refugia en el chatbot, que siempre está disponible y es amable; a corto plazo alivia, pero a la larga podría sustituir contactos humanos y aumentar la soledad. Los autores subrayan que ese mecanismo es teórico: ni siquiera midieron el apego emocional al chatbot, y piden estudios longitudinales y experimentales para ponerlo a prueba.

Límites
ANA: Recapitulemos las costuras del estudio.
DANI: Primero, es transversal: no hay orden temporal ni causalidad. Segundo, todo es autoinformado, y puede haber sesgo de deseabilidad social. Tercero, las medidas de salud mental son ultracortas, y la soledad es una sola pregunta. Cuarto, la escala de uso adictivo es una adaptación de la de redes sociales y los autores reconocen que no pueden asegurar que sea válida para chatbots, porque en las redes hablas con personas y en un chatbot el interlocutor es el propio sistema. Quinto, la muestra es de un panel voluntario, y usuarios y no usuarios difieren en edad y circunstancias. Y sexto, el propio concepto de "usuario" es muy amplio.
ANA: ¿Y qué ven los autores como fortalezas?
DANI: El tamaño de la muestra, el número casi igual de usuarios y no usuarios, y que miran tanto síntomas como bienestar. Es una primera radiografía europea, no un veredicto.

Qué podemos llevarnos
ANA: Si alguien me pregunta qué hacer con esto, ¿qué le digo?
DANI: Que no hay base para alarmarse por usar un chatbot para trabajar o buscar información, y tampoco para decir que sea totalmente inocuo. Lo que sí sugiere el estudio es fijarse en el para qué. Si se convierte en la vía principal para escapar de lo que sientes, o notas que te cuesta controlarlo o que se come tiempo con otras personas, quizá merezca la pena parar y reflexionar. Y los autores recuerdan que su recomendación de usar con conciencia y reforzar los contactos cara a cara se apoya sobre todo en lo aprendido con redes sociales, no en experimentos con chatbots.
ANA: Y un titular que conviene evitar.
DANI: "Los chatbots causan depresión y soledad". El estudio no dice eso. Dice que hay diferencias pequeñas y asociaciones que merecen seguirse con estudios mejores.
ANA: Y un último apunte, porque hablamos de soledad: si estás pasando por un mal momento, hablar con alguien de confianza o con un profesional sigue siendo lo primero, y un chatbot no sustituye eso.
DANI: Totalmente.
ANA: Gracias por escucharnos en Mente Digital. Si te ha gustado, compártelo. ¡Hasta la próxima!
\"\"\"
"""

def generate_script_from_pdf(pdf_bytes: bytes, api_key: str) -> PodcastScript:
    client = genai.Client(api_key=api_key.strip())
    
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Analiza el documento PDF adjunto. Extrae minuciosamente todos sus datos empíricos y metodológicos reales, "
        "y redacta el diálogo completo entre ANA y DANI reproduciendo exactamente el mismo nivel de detalle, la misma "
        "profundidad analítica en las explicaciones de Dani y el mismo tono que el ejemplo de referencia.\n\n"
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

                st.success("¡Episodio generado con el formato exacto del Ejemplo 2!")
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
