# app/modelo_multirama.py

import os
import re
import string
import numpy as np
import pandas as pd
import joblib
import torch
from collections import Counter

import tensorflow as tf
from transformers import DistilBertTokenizer, DistilBertModel
import spacy
import nltk
from nltk.corpus import stopwords

# ============================
# RUTAS BASE
# ============================

# Este archivo está en C:\Tesis\env\modelo_3.0\app
# El modelo y los .pkl están en la carpeta padre: C:\Tesis\env\modelo_3.0
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)

MODEL_PATH         = os.path.join(BASE_DIR, "mejor_modelo_multirama_mejorado.h5")
SCALER_BERT_PATH   = os.path.join(BASE_DIR, "scaler_bert.pkl")
SCALER_META_PATH   = os.path.join(BASE_DIR, "scaler_metadata.pkl")
SCALER_HEUR_PATH   = os.path.join(BASE_DIR, "scaler_heuristic.pkl")
ENCODER_AUTOR_PATH = os.path.join(BASE_DIR, "encoder_autor.pkl")  # ajusta si se llama distinto
DICCIONARIO_PATH   = os.path.join(BASE_DIR, "diccionario", "diccionario.csv")
BERT_MODEL_DIR     = os.path.join(BASE_DIR, "bert_model")
BERT_TOKEN_DIR     = os.path.join(BASE_DIR, "bert_tokenizer")

# ============================
# CARGA DE MODELOS Y RECURSOS
# ============================

# Modelo Keras multirama
model_v2 = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"AUC": tf.keras.metrics.AUC}
)

# Scalers y encoder
scaler_bert      = joblib.load(SCALER_BERT_PATH)
scaler_metadata  = joblib.load(SCALER_META_PATH)
scaler_heuristic = joblib.load(SCALER_HEUR_PATH)
ohe_autor        = joblib.load(ENCODER_AUTOR_PATH)

# Tokenizer y BERT locales
device = torch.device("cpu")
tokenizer  = DistilBertTokenizer.from_pretrained(BERT_TOKEN_DIR)
model_bert = DistilBertModel.from_pretrained(BERT_MODEL_DIR)
model_bert.to(device)
model_bert.eval()

# Diccionario emocional
df_diccionario = pd.read_csv(DICCIONARIO_PATH)

# ============================
# NLP: SpaCy & NLTK
# ============================

nlp = spacy.load("es_core_news_sm", disable=["ner", "tagger"])

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# ============================
# FUNCIONES DEL MODELO
# (son básicamente las mismas que usas en el notebook)
# ============================

def limpiar_basico(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    for c in string.punctuation:
        texto = texto.replace(c, "")
    return texto

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# ---- Diccionario emocional ----
def construir_diccionario_emocional(df_diccionario):
    emociones = df_diccionario.columns[1:]
    diccionario_emocional = {}
    for _, fila in df_diccionario.iterrows():
        palabra = fila['palabra']
        emociones_asociadas = [emocion for emocion in emociones if fila[emocion] == 1]
        if emociones_asociadas:
            diccionario_emocional[palabra] = emociones_asociadas
    return diccionario_emocional

diccionario_emocional = construir_diccionario_emocional(df_diccionario)

def calcular_emotividad(texto, diccionario):
    palabras = re.findall(r'\b\w+\b', str(texto).lower())
    total = len(palabras)
    emocionales = sum(1 for palabra in palabras if palabra in diccionario)
    return emocionales / total if total > 0 else 0

def emocion_predominante(texto, diccionario):
    palabras = re.findall(r'\b\w+\b', str(texto).lower())
    emociones = []
    for palabra in palabras:
        if palabra in diccionario:
            emociones.extend(diccionario[palabra])
    if emociones:
        return Counter(emociones).most_common(1)[0][0]
    else:
        return 'ninguna'

# ---- Heurísticas lingüísticas ----

verbos_modales = ['debería', 'podría', 'tendría', 'habría', 'puede', 'pueden',
                  'debe', 'deben', 'sería']
terminos_generalizadores = ['siempre', 'nunca', 'todos', 'nadie', 'jamás',
                            'ninguno', 'cualquiera']
verbos_opinion = ['creo', 'considero', 'opino', 'pienso', 'me parece',
                  'supongo', 'siento', 'estimo']
lexico_polarizado = ['horrible', 'excelente', 'terrible', 'maravilloso',
                     'desastroso', 'perfecto', 'abominable', 'magnífico']

def frecuencia_heuristica(texto, palabras_clave):
    texto_limpio = limpiar_basico(texto)
    palabras = texto_limpio.split()
    return sum(1 for palabra in palabras if palabra in palabras_clave)

def diversidad_lexica(texto):
    texto_limpio = limpiar_basico(texto)
    palabras = [p for p in texto_limpio.split() if p.isalpha() and p not in stop_words]
    if len(palabras) == 0:
        return 0
    return len(set(palabras)) / len(palabras)

def aplicar_heuristicas_linguisticas(df_heuristica):
    df_heuristica['incertidumbre'] = df_heuristica['contenido'].apply(
        lambda x: frecuencia_heuristica(x, verbos_modales + terminos_generalizadores)
    )
    df_heuristica['subjetividad'] = df_heuristica['contenido'].apply(
        lambda x: frecuencia_heuristica(x, verbos_opinion + lexico_polarizado)
    )
    df_heuristica['diversidad_lexica'] = df_heuristica['contenido'].apply(diversidad_lexica)
    return df_heuristica[['incertidumbre', 'subjetividad', 'diversidad_lexica']]

# ---- Patrones sintácticos ----
def aplicar_patrones_sintacticos(df_heuristica):
    resultados = []
    for doc in nlp.pipe(df_heuristica['contenido'], batch_size=32):
        oraciones = list(doc.sents)
        total_palabras = len([token.text for token in doc if token.is_alpha])
        total_palabras_mayus = len([token.text for token in doc if token.is_upper])
        total_exclamaciones = doc.text.count('!')
        frases_cortas = sum(1 for s in oraciones if len(s.text.split()) <= 7)
        porcentaje_mayus = total_palabras_mayus / total_palabras if total_palabras > 0 else 0
        frases_cortas_porcentaje = frases_cortas / len(oraciones) if oraciones else 0
        resultados.append({
            "porcentaje_mayusculas": porcentaje_mayus,
            "exclamaciones_totales": total_exclamaciones,
            "frases_cortas": frases_cortas,
            "frases_cortas_%": frases_cortas_porcentaje
        })
    df_patrones = pd.DataFrame(resultados, index=df_heuristica.index)
    return df_patrones

# ---- Relación título-cuerpo ----
def consistencia_jaccard(titulo, contenido):
    titulo = limpiar_basico(titulo)
    contenido = limpiar_basico(contenido)
    set_titulo = set([p for p in titulo.split() if p not in stop_words])
    set_contenido = set([p for p in contenido.split() if p not in stop_words])
    if len(set_titulo) == 0 or len(set_contenido) == 0:
        return 0
    interseccion = set_titulo.intersection(set_contenido)
    union = set_titulo.union(set_contenido)
    return len(interseccion) / len(union)

def aplicar_relacion_titulo_cuerpo(df_heuristica):
    df_heuristica['consistencia_titulo_cuerpo'] = df_heuristica.apply(
        lambda row: consistencia_jaccard(row['titulo'], row['contenido']), axis=1)
    return df_heuristica[['consistencia_titulo_cuerpo']]

# ---- Embedding heurístico (9 dims) ----
def generar_embedding_heuristico(df):
    emotividad = df['contenido'].apply(
        lambda x: calcular_emotividad(x, diccionario_emocional)
    ).to_frame(name='score_emotividad')

    heuristicas = aplicar_heuristicas_linguisticas(df.copy())
    patrones = aplicar_patrones_sintacticos(df.copy())
    relacion = aplicar_relacion_titulo_cuerpo(df.copy())

    heuristico_df = pd.concat([emotividad, heuristicas, patrones, relacion], axis=1)
    heuristico_df = heuristico_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return heuristico_df.to_numpy()

# ---- Feature engineering metadata ----

COL_NUM = [
    'exclamaciones',
    'interrogaciones',
    'palabras_sensacionalistas',
    'noticias_autor',
    'relacion_titulo_contenido',
    'es_muy_corta',
    'es_corta',
    'es_media',
    'es_larga',
    'es_muy_larga',
    'titulo_corto',
    'titulo_largo'
]

def feature_engineering(df):
    long_contenido = df['contenido'].apply(lambda x: len(str(x).split()))
    long_titulo = df['titulo'].apply(lambda x: len(str(x).split()))

    df['longitud_bruta'] = long_contenido
    df['longitud_titulo_bruta'] = long_titulo

    df['es_muy_corta']  = (long_contenido < 80).astype(int)
    df['es_corta']      = ((long_contenido >= 80) & (long_contenido < 200)).astype(int)
    df['es_media']      = ((long_contenido >= 200) & (long_contenido < 800)).astype(int)
    df['es_larga']      = ((long_contenido >= 800) & (long_contenido < 2000)).astype(int)
    df['es_muy_larga']  = (long_contenido >= 2000).astype(int)

    df['titulo_corto']  = (long_titulo < 6).astype(int)
    df['titulo_largo']  = (long_titulo > 15).astype(int)

    df['exclamaciones'] = df['contenido'].apply(lambda x: str(x).count('!'))
    df['interrogaciones'] = df['contenido'].apply(lambda x: str(x).count('?'))

    sensacionalistas = ['increíble', 'urgente', 'impactante', 'viral', 'escándalo']
    df['palabras_sensacionalistas'] = df['contenido'].apply(
        lambda x: sum(str(x).lower().count(p) for p in sensacionalistas)
    )

    df['fecha'] = pd.to_datetime(df['fecha'], format='%d-%m-%Y', errors='coerce')
    df['dia_semana'] = df['fecha'].dt.day_name()
    df['mes'] = df['fecha'].dt.month

    df['noticias_autor'] = df['autor'].map(df['autor'].value_counts())

    ratio = long_titulo / np.maximum(long_contenido, 1)
    ratio_clipped = np.clip(ratio, 0, 0.5)
    df['relacion_titulo_contenido'] = ratio_clipped

    autor_freq = df['autor'].value_counts()
    df['autor'] = df['autor'].apply(
        lambda x: x if x in autor_freq[autor_freq > 20].index else 'Otro'
    )

    df['mes'] = df['mes'].fillna(0)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

    dias = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
            'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['dia_semana_num'] = df['dia_semana'].map(dias).fillna(0).astype(int)
    df['dia_sin'] = np.sin(2 * np.pi * df['dia_semana_num'] / 7)
    df['dia_cos'] = np.cos(2 * np.pi * df['dia_semana_num'] / 7)

    df[COL_NUM] = df[COL_NUM].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df

# ============================
# FUNCIÓN PÚBLICA: lo único que Flask necesita
# ============================

def clasificar_noticia(titulo, autor, fecha, contenido, umbral=0.5):
    df_noticia = pd.DataFrame({
        "titulo": [titulo],
        "autor": [autor],
        "fecha": [fecha],
        "contenido": [contenido]
    })

    # Heurístico
    X_heur = generar_embedding_heuristico(df_noticia.copy())
    X_heur = scaler_heuristic.transform(X_heur)

    # Metadata
    df_meta = feature_engineering(df_noticia.copy())
    X_num = df_meta[COL_NUM].values
    X_num_scaled = scaler_metadata.transform(X_num)
    X_ciclicas = df_meta[['mes_sin', 'mes_cos', 'dia_sin', 'dia_cos']].values
    X_autor = ohe_autor.transform(df_meta[['autor']])
    X_metadata = np.concatenate([X_num_scaled, X_ciclicas, X_autor], axis=1)

    # BERT
    df_noticia['contenido_limpio'] = df_noticia['contenido'].astype(str).apply(limpiar_texto)

    inputs = tokenizer(
        df_noticia['contenido_limpio'].iloc[0],
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512
    )
    for k in inputs:
        inputs[k] = inputs[k].to(device)

    with torch.no_grad():
        outputs = model_bert(**inputs)
        X_bert = outputs.last_hidden_state[:, 0, :].cpu().numpy().reshape(1, -1)

    X_bert = scaler_bert.transform(X_bert)

    # Predicción
    prob = float(model_v2.predict([X_bert, X_metadata, X_heur], verbose=0)[0][0])
    clasificacion = "Verdadera" if prob >= umbral else "Falsa"

    return clasificacion, prob
