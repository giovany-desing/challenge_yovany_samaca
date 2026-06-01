Documentación del Proyecto

## Introducción

Este proyecto tiene como objetivo **operacionalizar un modelo de Machine Learning** que predice si un vuelo sufrirá un retraso superior a 15 minutos (`delay` = 1) o no (`delay` = 0) en el aeropuerto SCL. El modelo fue creado por un Data Scientist en un Jupyter Notebook (`exploration.ipynb`). El trabajo realizado consistió en transcribir ese análisis a código robusto (`model.py`), exponerlo mediante una API REST con FastAPI (`api.py`), desplegar la API en Google Cloud Platform (GCP) y automatizar el proceso con CI/CD usando GitHub Actions.

## Correcciones realizadas al notebook original

Durante la transcripción se detectaron y corrigieron los siguientes errores:

- **Error en `sns.barplot`**: El notebook usaba `sns.barplot(flights_by_airline.index, flights_by_airline.values, alpha=0.9)`, lo que en versiones recientes de seaborn produce `TypeError`. Se corrigió usando los parámetros explícitos `x=` e `y=` en todas las visualizaciones.
- **Ruta del dataset en las pruebas**: Los tests (`tests/model/test_model.py`) cargaban el archivo con `../data/data.csv`, lo que causaba `FileNotFoundError` al ejecutar `pytest` desde la raíz. Se cambió a `data/data.csv`.
- **Manejo del target en `preprocess`**: El método debía devolver `target` como `DataFrame` (no como `Series`) para cumplir con las aserciones de los tests. Se ajustó convirtiendo la serie en DataFrame de una columna.
- **Cálculo del balanceo en `fit`**: Se corrigió la forma de contar ejemplos de clase 0 y 1 usando `target_series = target.iloc[:, 0]` para extraer la columna como `Series`.
- **Imagen Docker para Cloud Run**: Se modificó el `Dockerfile` para que la aplicación escuche en el puerto indicado por la variable de entorno `$PORT` (necesario para Cloud Run) y se especificó la plataforma `linux/amd64` para compatibilidad.

## Desarrollo según las partes del desafío

### Parte I – Modelo (`model.py`)

Se seleccionó el modelo **XGBoost con las 10 características más importantes y balanceo de clases** (modelo `xgb_model_2` del notebook). Las razones:

- Las 10 columnas (ej. `OPERA_Latin American Wings`, `MES_7`, `TIPOVUELO_I`, etc.) concentran la mayor importancia.
- El balanceo mediante `scale_pos_weight = n_y0 / n_y1` mejora el *recall* de la clase minoritaria (retrasos), que es lo que realmente importa para el negocio.
- Las pruebas `make model-test` se ejecutan sin errores.

### Parte II – API con FastAPI (`api.py`)

Se implementaron dos endpoints:

- `GET /health`: retorna `{"status": "OK"}`.
- `POST /predict`: recibe un JSON con una lista de vuelos. Valida que `OPERA` sea conocido, `TIPOVUELO` sea `I` o `N`, y `MES` esté entre 1 y 12.  
  La respuesta incluye:
  - `"predict"`: lista de enteros (0 = sin retraso, 1 = con retraso) – requerido por las pruebas.
  - `"meaning"`: lista de textos ("Sin retraso" / "Con retraso") para facilitar la interpretación humana (adicional, no rompe los tests).

Las pruebas `make api-test` se ejecutan correctamente.

### Parte III – Despliegue en la nube (GCP)

Se utilizó **Google Cloud Run**. Los pasos fueron:

1. Construir la imagen Docker para `linux/amd64`:
   ```bash
   docker build --platform linux/amd64 -t gcr.io/textil-476019/delay-api:latest .



### Tutorial para desarrolladores

1. Prerrequisitos

* Python 3.9+
* git, pip, virtualenv (o venv)
* (Solo para despliegue) docker, gcloud CLI y una cuenta de Google Cloud Platform con facturación habilitada


2. Clonar y preparar entorno local

git clone https://github.com/giovany-desing/challenge_yovany_samaca.git
cd challenge_MLE
python -m venv venv
source venv/bin/activate               # Linux/macOS
# venv\Scripts\activate                # Windows
pip install -r requirements.txt -r requirements-test.txt -r requirements-dev.txt


3. Ejecutar la API localmente

uvicorn challenge.api:app --reload --host 0.0.0.0 --port 8000

La API estará disponible en http://127.0.0.1:8000.
Endpoints:

GET /health
POST /predict (ver ejemplos abajo)

4. Probar la API

Health check:
curl http://localhost:8000/health

Predicción:

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"flights":[{"OPERA":"Aerolineas Argentinas","TIPOVUELO":"N","MES":3}]}'


Respuesta esperada:
  {"predict":[0],"meaning":["Sin retraso"]}

5. Ejecutar las pruebas

make model-test      # pruebas del modelo
make api-test        # pruebas de la API
make stress-test     # prueba de carga (requiere API corriendo)


6. Desplegar en Google Cloud Run (GCP)

6.1. Configurar GCP (solo una vez)
    * Crear un proyecto en console.cloud.google.com. o solicitar acceso a un proyecto existente
    * Habilitar las APIs: Cloud Run, Artifact Registry.
    * Instalar y autenticar gcloud:
        gcloud auth login
        gcloud config set project <PROJECT_ID>

6.2. Construir y subir la imagen Docker
    # Construir para linux/amd64 (necesario para Cloud Run)
    docker build --platform linux/amd64 -t gcr.io/<PROJECT_ID>/delay-api:latest .

    # Subir a Google Container Registry (o Artifact Registry)
    docker push gcr.io/<PROJECT_ID>/delay-api:latest


6.3. Desplegar en Cloud Run

    gcloud run deploy delay-api \
  --image gcr.io/<PROJECT_ID>/delay-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --timeout=600s \
  --memory=2Gi

6.4. Probar el despliegue

    # Health check
    curl https://<TU-URL>/health

    # Predicción
    curl -X POST https://<TU-URL>/predict \
        -H "Content-Type: application/json" \
        -d '{"flights":[{"OPERA":"Aerolineas Argentinas","TIPOVUELO":"N","MES":3}]}'

6.5. (Opcional) Prueba de estrés contra la URL desplegada

    Edita el Makefile (línea 26) reemplazando STRESS_URL por la URL de Cloud Run y ejecuta make stress-test



7. CI/CD (GitHub Actions)

Los workflows están en .github/workflows/.
ci.yml: corre make model-test y make api-test en cada push/pull request.
cd.yml: se activa en push a main. Necesita los secretos:

GCP_PROJECT_ID = ID del proyecto
GCP_SA_KEY = contenido del archivo JSON de una cuenta de servicio con roles run.admin, artifactregistry.writer e iam.serviceAccountUser.
