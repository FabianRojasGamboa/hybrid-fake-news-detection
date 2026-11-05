# Notas de publicación y reproducción

Este repositorio presenta **el código original usado para la demostración**, más notebooks de trabajo y documentación bilingüe. No es una refactorización completamente validada ni una versión publicada de la aplicación.

## Qué contiene

- `app.py`, `templates/index.html`: la aplicación Flask original que fue ejecutada con éxito en Windows, con un único ajuste de seguridad en esta copia: `debug=False`.
- `notebooks/`: copias de los notebooks originales **sin outputs ejecutados ni metadatos personales de Colab**. Se conservó el código para trazabilidad.
- `legacy/modelo_multirama.py`: script alternativo de inferencia. **No lo importa `app.py`** y sus rutas antiguas requieren ajustes si alguien desea usarlo.
- `diccionario.csv`: recurso de vocabulario emocional utilizado por la aplicación; revisar licencia/procedencia antes de publicar.
- `assets/`: capturas de la aplicación.

## Qué falta para inferencia completa

El `app.py` original requiere que se copien a la raíz estos archivos **del entorno propio y de confianza**:

- `mejor_modelo_multirama_mejorado.h5`
- `scaler_bert.pkl`, `scaler_metadata.pkl`, `scaler_heuristic.pkl`, `encoder_autor.pkl`

DistilBERT se descarga por nombre desde Hugging Face en el primer uso si aún no está en caché y hay conectividad. El modelo Keras y los objetos serializados no están incluidos en este repositorio.

> Advertencia: `joblib.load` puede ejecutar código al deserializar archivos no confiables. El `app.py` original utiliza además `safe_mode=False` al cargar Keras. Ejecuta únicamente artefactos propios o de origen comprobado, en un entorno aislado. No expongas la aplicación directamente en Internet ni dejes `debug=True` en un despliegue.

## Limitaciones conocidas

- Los notebooks originales se conservaron para documentación; algunas etapas usan rutas de Google Drive y necesitan datasets que no se redistribuyen.
- La fecha tiene formato ISO en el CSV revisado, mientras que una transformación espera `DD-MM-AAAA`.
- Los escaladores se ajustaron antes del split en la evaluación original, con riesgo de fuga de información. Las métricas reportadas no constituyen una evaluación corregida.
- El cómputo de frecuencias de autores en inferencia difiere del cómputo sobre lotes de entrenamiento. Se preservó el comportamiento original por decisión del autor.
- La aplicación cuenta con una prueba de funcionamiento manual en Windows; no se realizó validación automatizada de extremo a extremo de este paquete.

## Antes de hacerlo público

- Confirmar condiciones de redistribución del diccionario emocional y retirarlo si no están claras.
- Confirmar la autorización del equipo y las contribuciones.
- Considerar un archivo `requirements-lock.txt` generado desde el entorno que funciona y una guía de obtención de artefactos de inferencia.
