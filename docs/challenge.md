# Documentación del Proyecto

## Introducción

Este proyecto tiene como objetivo **operacionalizar un modelo de Machine Learning** que predice si un vuelo sufrirá un retraso superior a 15 minutos (`delay` = 1) o no (`delay` = 0) en el aeropuerto SCL. El modelo fue creado por un Data Scientist en un Jupyter Notebook (`exploration.ipynb`). El trabajo realizado consistió en transcribir ese análisis a código robusto (`model.py`), exponerlo mediante una API REST con FastAPI (`api.py`), desplegar la API en la nube y automatizar el proceso con CI/CD usando GitHub Actions.

## Correcciones y hallazgos durante el desarrollo

Esta sección separa dos tipos de intervenciones: **bugs reales heredados del notebook original** y **decisiones de ingeniería** necesarias para llevar el análisis exploratorio a un servicio productivo.

### Bugs del notebook original

- **Error en `sns.barplot`**: el notebook usaba `sns.barplot(flights_by_airline.index, flights_by_airline.values, alpha=0.9)`, lo que en versiones recientes de seaborn produce `TypeError`. Se corrigió usando los parámetros explícitos `x=` e `y=` en todas las visualizaciones.
- **Comparaciones estrictas en `get_period_day`**: la función original usa `>` / `<` estrictos en los límites de cada franja horaria (ej. `05:00`, `11:59`), dejando `None` cuando la hora coincide exactamente con un límite. **Se documenta para trazabilidad, pero no se corrigió en `model.py`** porque el modelo productivo final (definido por el propio DS en la sección de conclusiones del notebook) **no utiliza `period_day` como feature** — el set de entrenamiento final solo usa `OPERA`, `TIPOVUELO` y `MES`. El bug no afecta el resultado del modelo.

### Decisiones de ingeniería para productivizar el modelo

- **Ruta del dataset en los tests**: `tests/model/test_model.py` carga el dataset con `data/data.csv` (ruta relativa a la raíz del repo, para que `pytest` funcione ejecutado desde ahí).
- **`target` como `DataFrame`, no `Series`**: `preprocess()` devuelve el target envuelto en un `DataFrame` de una columna, para cumplir el contrato que esperan los tests oficiales.
- **Cálculo del balanceo en `fit`**: se extrae la columna real con `target.iloc[:, 0]` antes de contar clases, ya que `target` llega como `DataFrame`.
- **Tipado explícito de `MES`**: se fuerza `astype(int)` antes de generar las dummies, para evitar que un `MES` recibido como `float` (p. ej. `7.0`) genere una columna `MES_7.0` que no coincida con `top_10_features` y pierda esa señal silenciosamente.
- **Robustez en `predict`**: se usa `reindex(columns=top_10_features, fill_value=0)` en vez de indexado directo, para no lanzar `KeyError` si llega un DataFrame con columnas faltantes.

## Desarrollo según las partes del desafío

### Parte I – Modelo (`model.py`)

Se seleccionó el modelo **XGBoost con las 10 características más importantes y balanceo de clases** (equivalente al `xgb_model_2` del notebook). Razones:

- El notebook concluye que no hay diferencia notable de desempeño entre XGBoost y Regresión Logística, y que reducir a las 10 features más importantes no perjudica el resultado — pero deja abierta la elección final entre ambos modelos balanceados.
- Se optó por **XGBoost** porque maneja de forma nativa relaciones no lineales entre variables dummy de alta cardinalidad (como las decenas de categorías de `OPERA`), es más robusto ante ese tipo de codificación que una regresión lineal, y ya contaba con un ajuste explícito de hiperparámetros (`learning_rate=0.01`) en el notebook original.
- El balanceo mediante `scale_pos_weight = n_y0 / n_y1` (calculado dinámicamente en `fit`, no hardcodeado) mejora el *recall* de la clase minoritaria (retrasos), que es la que realmente importa para el negocio: es preferible predecir de más un retraso que pasar uno por alto.

**Validado con ejecución real**: `make model-test` → `4 passed`.

### Parte II – API con FastAPI (`api.py`)

Se implementaron dos endpoints:

- `GET /health`: retorna `{"status": "OK"}`.
- `POST /predict`: recibe un JSON con una lista de vuelos (`flights`). Valida que `OPERA` sea conocido (según los valores presentes en `data.csv`), que `TIPOVUELO` sea `I` o `N`, y que `MES` esté entre 1 y 12 — devolviendo `400` en caso contrario. Responde:
```json
  {"predict": [0]}
```
  (lista de enteros: 0 = sin retraso, 1 = con retraso)

Los endpoints están declarados como funciones **síncronas** (`def`, no `async def`). Esto es intencional: dentro de `/predict` se ejecuta código CPU-bound (`model.predict`, que corre inferencia de XGBoost). FastAPI ejecuta automáticamente los endpoints síncronos en un threadpool aparte, liberando el event loop principal para atender otras solicitudes en paralelo. Declararlos como `async def` sin ningún `await` real habría bloqueado el event loop en cada predicción, degradando la latencia bajo carga concurrente (ver hallazgo de performance más abajo).

**Validado con ejecución real**: `make api-test` → `4 passed`.

### Persistencia del modelo

El modelo entrenado se persiste como artefacto (`challenge/artifacts/model.joblib`) en vez de reentrenarse en cada arranque de la API:

- `scripts/train.py`: script independiente que entrena el modelo desde `data/data.csv` y lo guarda con `joblib`. Se ejecuta con `python scripts/train.py` (o `make train-model`).
- `DelayModel.save()` / `DelayModel.load()`: métodos agregados en `model.py` para serializar y cargar el modelo (no se modificaron los métodos provistos por el template).
- `challenge/api.py`: al arrancar, la API intenta cargar el artefacto existente; si no lo encuentra, entrena el modelo y lo guarda automáticamente, sin intervención manual.

**Validado con ejecución real**: arranque con artefacto existente (rápido, sin reentrenar) y arranque sin artefacto (reentrena y regenera el archivo automáticamente), ambos probados en local y en producción (Render), con predicciones consistentes en ambos casos.

### Parte III – Despliegue en la nube (Render)

**Nota importante**: el despliegue se realizó en [Render](https://render.com) en lugar de GCP/Cloud Run. La cuenta de GCP disponible solo contaba con la capa gratuita de 24 horas, y no fue posible habilitar facturación para un despliegue persistente. El enunciado del challenge indica *"deploy the API in your favorite cloud provider (we recommend to use GCP)"* — es una recomendación, no un requisito estricto, por lo que se optó por Render como alternativa gratuita, sin tarjeta de crédito y con disponibilidad indefinida mientras el servicio se mantenga activo.

**Pasos realizados:**

1. Se ajustó el `Dockerfile` para escuchar en el puerto indicado dinámicamente por la variable de entorno `$PORT` (Render, igual que Cloud Run, inyecta esta variable en tiempo de ejecución):
```dockerfile
   CMD exec uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT:-8080}
```
2. Se creó un servicio Web en Render (tipo **Docker**, instancia **Free**), conectado al repositorio de GitHub, rama `main`.
3. Se desactivó el Auto-Deploy nativo de Render, delegando el disparo de despliegues al pipeline de CI/CD (`cd.yml`) vía un **Deploy Hook**, para mantener el flujo "push a `main` → deploy" bajo control del pipeline versionado en el repo (ver Parte IV).
4. URL pública del servicio: `https://delay-api-dkkx.onrender.com`

**Validado con ejecución real:**
- Health check: `GET /health` → `{"status": "OK"}`
- Predicción: `POST /predict` con el payload de ejemplo → `{"predict": [0]}`
- `make stress-test` (100 usuarios concurrentes, 60s) → **0% de fallos**

**Hallazgo de performance y su corrección:**

En la primera corrida de `make stress-test` contra Render, con los endpoints declarados como `async def`, se observó una **latencia creciente sostenida durante toda la prueba** (mediana pasando de ~200ms a ~1300ms a medida que se sumaban usuarios, con máximo de 4707ms), aunque sin ningún fallo (0% error rate). La causa: el event loop único del proceso (Render asigna `WEB_CONCURRENCY=1` por defecto en el tier gratuito) se bloqueaba en cada predicción, encolando las solicitudes concurrentes de forma secuencial en vez de procesarlas en paralelo.

**Corrección aplicada**: se cambiaron los endpoints de `async def` a `def` (síncronos), permitiendo que FastAPI los delegue a un threadpool. Tras el fix, se repitió `make stress-test` contra la URL desplegada bajo las mismas condiciones (100 usuarios, 60s): la latencia siguió creciendo de forma prácticamente idéntica (mediana final ~1900ms, máximo ~4700ms, 0% de fallos en ambos casos). Esto indica que el cuello de botella **no estaba en el manejo de concurrencia de la aplicación**, sino en los recursos de CPU compartidos y limitados de la instancia gratuita de Render (`WEB_CONCURRENCY=1`). Bajo tráfico bajo o moderado la latencia se mantiene en rangos aceptables (~150-300ms, confirmado con requests individuales sin carga concurrente); el crecimiento sostenido solo aparece con 100 usuarios simultáneos, consistente con una limitación de infraestructura y no de la implementación. En ambos escenarios, `make stress-test` se ejecuta sin fallos (0% error rate).

### Parte IV – CI/CD (GitHub Actions)

Los workflows están en `.github/workflows/`:

- **`ci.yml`**: se ejecuta en cada `push` y `pull_request` a `main`. Corre `make model-test` y `make api-test`.
- **`cd.yml`**: se dispara mediante `workflow_run`, escuchando la finalización de `ci.yml` en `main` — **solo despliega si `ci.yml` terminó exitosamente** (`conclusion == 'success'`), actuando como quality gate:
```yaml
  on:
    workflow_run:
      workflows: ["Continuous Integration"]
      types:
        - completed
      branches: [ main ]

  jobs:
    deploy:
      runs-on: ubuntu-latest
      if: ${{ github.event.workflow_run.conclusion == 'success' }}
      steps:
      - name: Trigger Render Deploy
        run: curl -f -X POST "${{ secrets.RENDER_DEPLOY_HOOK }}"
```
  Requiere el secreto `RENDER_DEPLOY_HOOK` configurado en GitHub (`Settings → Secrets and variables → Actions`), obtenido desde `Settings → Deploy Hook` del servicio en el dashboard de Render. Antes de este ajuste, `cd.yml` escuchaba `push` directamente, desplegando incluso si los tests de `ci.yml` fallaban — se corrigió para encadenar el despliegue a un CI exitoso.

## Flujo de trabajo (GitFlow)

Los cambios posteriores a la entrega inicial se desarrollaron siguiendo GitFlow: rama `develop` para integración, ramas `feature/*` para cambios puntuales (ej. `feature/model-persistence`), y Pull Requests hacia `develop` y luego hacia `main`. Las ramas de desarrollo se conservan sin eliminar, conforme a la recomendación del enunciado original.

---

## Tutorial para desarrolladores

### 1. Prerrequisitos

- Python 3.9+ (recomendado, para coherencia con el notebook original; el `Dockerfile` de producción usa 3.11)
- git, pip, virtualenv (o venv)
- (Solo si se quiere redesplegar) cuenta en [Render](https://render.com), sin necesidad de tarjeta de crédito

### 2. Clonar y preparar entorno local

```bash
git clone https://github.com/giovany-desing/challenge_yovany_samaca.git
cd challenge_yovany_samaca
python -m venv venv
source venv/bin/activate               # Linux/macOS
# venv\Scripts\activate                # Windows
pip install -r requirements.txt -r requirements-test.txt -r requirements-dev.txt
```

### 3. Ejecutar la API localmente

```bash
uvicorn challenge.api:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en `http://127.0.0.1:8000`. Documentación interactiva (Swagger UI) en `http://127.0.0.1:8000/docs`.

Endpoints:
- `GET /health`
- `POST /predict`

### 4. Probar la API

**Health check:**
```bash
curl http://localhost:8000/health
```

**Predicción:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"flights":[{"OPERA":"Aerolineas Argentinas","TIPOVUELO":"N","MES":3}]}'
```

Respuesta esperada:
```json
{"predict":[0]}
```

### 5. Ejecutar las pruebas

```bash
make model-test      # pruebas del modelo
make api-test         # pruebas de la API
make stress-test      # prueba de carga (requiere la API desplegada o corriendo localmente)
```

### 6. Despliegue en Render

6.1. Crear cuenta en [render.com](https://render.com) (sin tarjeta de crédito).

6.2. **New +** → **Web Service** → conectar el repositorio de GitHub → Render detecta el `Dockerfile` automáticamente (Environment: Docker).

6.3. Configuración del servicio:
   - Branch: `main`
   - Instance Type: **Free**

6.4. En **Settings → Build & Deploy**, desactivar **Auto-Deploy** (el pipeline de CD del repo se encarga de disparar los despliegues).

6.5. En **Settings → Deploy Hook**, copiar la URL generada y guardarla como secreto `RENDER_DEPLOY_HOOK` en GitHub (`Settings → Secrets and variables → Actions`).

6.6. Probar el despliegue:
```bash
curl https://delay-api-dkkx.onrender.com/health

curl -X POST https://delay-api-dkkx.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"flights":[{"OPERA":"Aerolineas Argentinas","TIPOVUELO":"N","MES":3}]}'
```

6.7. Prueba de estrés contra la URL desplegada:

La URL ya está configurada en el `Makefile` (línea 26). Ejecutar:
```bash
make stress-test
```

### 7. CI/CD (GitHub Actions)

Los workflows están en `.github/workflows/`:
- **`ci.yml`**: corre `make model-test` y `make api-test` en cada push/pull request a `main`.
- **`cd.yml`**: se activa en push a `main`. Dispara el despliegue en Render vía Deploy Hook. Requiere el secreto:
  - `RENDER_DEPLOY_HOOK` = URL del Deploy Hook del servicio en Render.