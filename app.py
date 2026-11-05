from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import re
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
import tensorflow as tf
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
import string
import nltk
from nltk.corpus import stopwords
import spacy

# ============================
# FUNCIONES DE DICCIONARIO EMOCIONAL
# ============================

def construir_diccionario_emocional(df_diccionario):
    emociones = df_diccionario.columns[1:]  # Todas menos 'palabra'
    diccionario_emocional = {}
    for _, fila in df_diccionario.iterrows():
        palabra = fila['palabra']
        emociones_asociadas = [emocion for emocion in emociones if fila[emocion] == 1]
        if emociones_asociadas:
            diccionario_emocional[palabra] = emociones_asociadas
    return diccionario_emocional

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

def aplicar_emotividad(df_heuristica, df_diccionario):
    diccionario_emocional = construir_diccionario_emocional(df_diccionario)
    df_heuristica['score_emotividad'] = df_heuristica['contenido'].apply(
        lambda x: calcular_emotividad(x, diccionario_emocional)
    )
    df_heuristica['emocion_predominante'] = df_heuristica['contenido'].apply(
        lambda x: emocion_predominante(x, diccionario_emocional)
    )
    return df_heuristica[['score_emotividad']]

# ============================
# HEURÍSTICAS LINGÜÍSTICAS
# ============================

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

verbos_modales = ['debería', 'podría', 'tendría', 'habría', 'puede', 'pueden', 'podría', 'debe', 'deben', 'sería']
terminos_generalizadores = ['siempre', 'nunca', 'todos', 'nadie', 'jamás', 'ninguno', 'cualquiera']
verbos_opinion = ['creo', 'considero', 'opino', 'pienso', 'me parece', 'supongo', 'siento', 'estimo']
lexico_polarizado = ['horrible', 'excelente', 'terrible', 'maravilloso', 'desastroso', 'perfecto', 'abominable', 'magnífico']

def limpiar_texto_heuristica(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    for c in string.punctuation:
        texto = texto.replace(c, "")
    return texto

def frecuencia_heuristica(texto, palabras_clave):
    texto_limpio = limpiar_texto_heuristica(texto)
    palabras = texto_limpio.split()
    return sum(1 for palabra in palabras if palabra in palabras_clave)

def diversidad_lexica(texto):
    texto_limpio = limpiar_texto_heuristica(texto)
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

# ============================
# PATRONES SINTÁCTICOS (SPACY)
# ============================

nlp = spacy.load("es_core_news_sm", disable=["ner", "tagger"])

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

# ============================
# RELACIÓN TÍTULO–CUERPO (JACCARD)
# ============================

def limpiar_texto_jaccard(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    for c in string.punctuation:
        texto = texto.replace(c, "")
    return texto

def consistencia_jaccard(titulo, contenido):
    titulo = limpiar_texto_jaccard(titulo)
    contenido = limpiar_texto_jaccard(contenido)
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

# ============================
# EMBEDDING HEURÍSTICO (9 DIMENSIONES)
# ============================

def generar_embedding_heuristico(df, df_diccionario):
    emotividad = aplicar_emotividad(df.copy(), df_diccionario)
    heuristicas = aplicar_heuristicas_linguisticas(df.copy())
    patrones = aplicar_patrones_sintacticos(df.copy())
    relacion = aplicar_relacion_titulo_cuerpo(df.copy())

    heuristico_df = pd.concat([emotividad, heuristicas, patrones, relacion], axis=1)
    heuristico_df = heuristico_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return heuristico_df.to_numpy()

# ============================
# FEATURE ENGINEERING (VERSIÓN NUEVA DEL MODELO)
# ============================

# Columnas numéricas usadas en el scaler_metadata entrenado
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
    # 1) Longitudes brutas
    long_contenido = df['contenido'].apply(lambda x: len(str(x).split()))
    long_titulo = df['titulo'].apply(lambda x: len(str(x).split()))

    df['longitud_bruta'] = long_contenido
    df['longitud_titulo_bruta'] = long_titulo

    # 2) Flags de longitud del contenido
    df['es_muy_corta']  = (long_contenido < 80).astype(int)
    df['es_corta']      = ((long_contenido >= 80) & (long_contenido < 200)).astype(int)
    df['es_media']      = ((long_contenido >= 200) & (long_contenido < 800)).astype(int)
    df['es_larga']      = ((long_contenido >= 800) & (long_contenido < 2000)).astype(int)
    df['es_muy_larga']  = (long_contenido >= 2000).astype(int)

    # 3) Flags de longitud del título
    df['titulo_corto']  = (long_titulo < 6).astype(int)
    df['titulo_largo']  = (long_titulo > 15).astype(int)

    # 4) Puntuación
    df['exclamaciones'] = df['contenido'].apply(lambda x: str(x).count('!'))
    df['interrogaciones'] = df['contenido'].apply(lambda x: str(x).count('?'))

    # 5) Palabras sensacionalistas
    sensacionalistas = ['increíble', 'urgente', 'impactante', 'viral', 'escándalo']
    df['palabras_sensacionalistas'] = df['contenido'].apply(
        lambda x: sum(str(x).lower().count(p) for p in sensacionalistas)
    )

    # 6) Fecha → componentes
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d-%m-%Y', errors='coerce')
    df['dia_semana'] = df['fecha'].dt.day_name()
    df['mes'] = df['fecha'].dt.month

    # Frecuencia por autor
    df['noticias_autor'] = df['autor'].map(df['autor'].value_counts())

    # 7) Relación título / contenido (ratio acotada)
    ratio = long_titulo / np.maximum(long_contenido, 1)
    ratio_clipped = np.clip(ratio, 0, 0.5)
    df['relacion_titulo_contenido'] = ratio_clipped

    # 8) Agrupación de autores poco frecuentes
    autor_freq = df['autor'].value_counts()
    df['autor'] = df['autor'].apply(
        lambda x: x if x in autor_freq[autor_freq > 20].index else 'Otro'
    )

    # 9) Variables cíclicas de mes y día
    df['mes'] = df['mes'].fillna(0)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

    dias = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
            'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['dia_semana_num'] = df['dia_semana'].map(dias).fillna(0).astype(int)
    df['dia_sin'] = np.sin(2 * np.pi * df['dia_semana_num'] / 7)
    df['dia_cos'] = np.cos(2 * np.pi * df['dia_semana_num'] / 7)

    # 10) Limpiar columnas numéricas
    df[COL_NUM] = df[COL_NUM].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df

# ============================
# BERT: LIMPIEZA Y MODELO
# ============================

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model_bert = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

# ============================
# CARGA DE MODELO Y SCALERS (NUEVO MODELO MULTIRAMA)
# ============================

model = tf.keras.models.load_model(
    'mejor_modelo_multirama_mejorado.h5',
    custom_objects={"AUC": tf.keras.metrics.AUC},
    safe_mode=False
)


scaler_metadata = joblib.load('scaler_metadata.pkl')
scaler_bert = joblib.load('scaler_bert.pkl')
scaler_heuristic = joblib.load('scaler_heuristic.pkl')
ohe_autor = joblib.load('encoder_autor.pkl')

df_diccionario = pd.read_csv('diccionario.csv')

# ============================
# FUNCIÓN DE PREDICCIÓN (CONSOLa)
# ============================

def predecir_noticia(noticia):
    df_noticia = pd.DataFrame(noticia)

    # 1) Embeddings heurísticos
    X_heuristico = generar_embedding_heuristico(df_noticia.copy(), df_diccionario)
    X_heuristico = scaler_heuristic.transform(X_heuristico)

    # 2) Metadata (feature engineering + escalado)
    df_meta = feature_engineering(df_noticia.copy())
    X_num = df_meta[COL_NUM].values
    X_num_scaled = scaler_metadata.transform(X_num)

    X_ciclicas = df_meta[['mes_sin', 'mes_cos', 'dia_sin', 'dia_cos']].values
    X_autor = ohe_autor.transform(df_meta[['autor']])

    X_metadata = np.concatenate([X_num_scaled, X_ciclicas, X_autor], axis=1)

    # 3) Embedding BERT
    df_noticia['contenido_limpio'] = df_noticia['contenido'].apply(limpiar_texto)
    inputs = tokenizer(
        df_noticia['contenido_limpio'].iloc[0],
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model_bert(**inputs)
        X_bert = outputs.last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)
    X_bert = scaler_bert.transform(X_bert)

    # 4) Predicción con modelo multirama
    prob = float(model.predict([X_bert, X_metadata, X_heuristico], verbose=0)[0][0])
    etiqueta = "Verdadera" if prob >= 0.5 else "Falsa"

    print(f"\nProbabilidad de ser verdadera: {prob:.2%}")
    print(f"Clasificación: {etiqueta}")

    # Explicación rápida (igual que antes)
    titulo = df_noticia['titulo'].iloc[0]
    contenido = df_noticia['contenido'].iloc[0]
    autor = df_noticia['autor'].iloc[0]

    coincidencias = len(set(re.findall(r'\b\w+\b', titulo.lower())).intersection(
                       set(re.findall(r'\b\w+\b', contenido.lower()))))
    print(f"- Coincidencias título-contenido: {coincidencias} palabras clave")

    if autor.lower() == "desconocido":
        print("- ⚠️ Autor desconocido (reduce credibilidad)")

    longitud = len(contenido.split())
    if longitud < 150:
        print(f"- Contenido corto ({longitud} palabras)")
    elif longitud > 800:
        print(f"- Contenido muy largo ({longitud} palabras)")
    else:
        print(f"- Longitud adecuada ({longitud} palabras)")

    exclamaciones = contenido.count('!')
    if exclamaciones > 3:
        print(f"- ⚠️ {exclamaciones} signos de exclamación (posible sensacionalismo)")

    dic_emocional = construir_diccionario_emocional(df_diccionario)
    emotividad = calcular_emotividad(contenido, dic_emocional)
    emocion = emocion_predominante(contenido, dic_emocional)
    print(f"- Score de emotividad: {emotividad:.2f} (emoción predominante: {emocion})")

# ============================
# FLASK APP
# ============================

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    data = request.get_json()
    df_input = {
        'titulo': [data['titulo']],
        'autor': [data['autor']],
        'fecha': [data['fecha']],
        'contenido': [data['contenido']]
    }

    resultado = predecir_noticia_web(df_input)

    return jsonify(resultado)

def predecir_noticia_web(noticia):
    df_noticia = pd.DataFrame(noticia)

    # 1) Embeddings heurísticos
    X_heuristico = generar_embedding_heuristico(df_noticia.copy(), df_diccionario)
    X_heuristico = scaler_heuristic.transform(X_heuristico)

    # 2) Metadata
    df_meta = feature_engineering(df_noticia.copy())
    X_num = df_meta[COL_NUM].values
    X_num_scaled = scaler_metadata.transform(X_num)

    X_ciclicas = df_meta[['mes_sin', 'mes_cos', 'dia_sin', 'dia_cos']].values
    X_autor = ohe_autor.transform(df_meta[['autor']])

    X_metadata = np.concatenate([X_num_scaled, X_ciclicas, X_autor], axis=1)

    # 3) Embedding BERT
    df_noticia['contenido_limpio'] = df_noticia['contenido'].apply(limpiar_texto)
    inputs = tokenizer(
        df_noticia['contenido_limpio'].iloc[0],
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model_bert(**inputs)
        X_bert = outputs.last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)
    X_bert = scaler_bert.transform(X_bert)

    # 4) Predicción con modelo multirama
    prob = float(model.predict([X_bert, X_metadata, X_heuristico], verbose=0)[0][0])
    etiqueta = "Verdadera" if prob >= 0.5 else "Falsa"

    titulo = df_noticia['titulo'].iloc[0]
    contenido = df_noticia['contenido'].iloc[0]
    autor = df_noticia['autor'].iloc[0]

    coincidencias = len(set(re.findall(r'\b\w+\b', titulo.lower())).intersection(
                       set(re.findall(r'\b\w+\b', contenido.lower()))))

    dic_emocional = construir_diccionario_emocional(df_diccionario)
    emotividad = round(calcular_emotividad(contenido, dic_emocional), 2)
    emocion = emocion_predominante(contenido, dic_emocional)

    result = {
        "probabilidad": f"{prob:.2%}",
        "etiqueta": etiqueta,
        "coincidencias_titulo_contenido": coincidencias,
        "longitud": len(contenido.split()),
        "exclamaciones": contenido.count('!'),
        "autor_desconocido": autor.lower() == "desconocido",
        "emotividad": emotividad,
        "emocion": emocion
    }

    return result

if __name__ == '__main__':
    app.run(debug=False)
