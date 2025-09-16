<context>
I want to deploy a a fine-tuned BERT model designed to classify pairs of sentences as either paraphrases or not using fastAPI and render free web service. The first step is creating a search API with fastAPI
</context>

<example>
Below is an example fastAPI code for a semantic search tool
Main.py : from fastapi import FastAPI
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.metrics import DistanceMetric
import numpy as np
from app.functions import returnSearchResultIndexes

# define model info
model_name = 'all-MiniLM-L6-v2'
model_path = "app/data/" + model_name

# load model
model = SentenceTransformer(model_path)

# load video index
df = pl.scan_parquet('app/data/video-index.parquet')

# create distance metric object
dist_name = 'manhattan'
dist = DistanceMetric.get_metric(dist_name)


# create FastAPI object
app = FastAPI()

# API operations
@app.get("/")
def health_check():
    return {'health_check': 'OK'}

@app.get("/info")
def info():
    return {'name': 'yt-search', 'description': "Search API for Shaw Talebi's YouTube videos."}

@app.get("/search")
def search(query: str):
    idx_result = returnSearchResultIndexes(query, df, model, dist)
    return df.select(['title', 'video_id']).collect()[idx_result].to_dict(as_series=False)
functions.py: import numpy as np
import polars
import sentence_transformers
import sklearn

# helper function
def returnSearchResultIndexes(query: str, 
                        df: polars.lazyframe.frame.LazyFrame, 
                        model, 
                        dist: sklearn.metrics._dist_metrics.ManhattanDistance) -> np.ndarray:
    """
        Function to return indexes of top search results
    """
    
    # embed query
    query_embedding = model.encode(query).reshape(1, -1)
    
    # compute distances between query and titles/transcripts
    dist_arr = dist.pairwise(df.select(df.columns[4:388]).collect(), query_embedding) + dist.pairwise(df.select(df.columns[388:]).collect(), query_embedding)

    # search paramaters
    threshold = 40 # eye balled threshold for manhatten distance
    top_k = 5

    # evaluate videos close to query based on threshold
    idx_below_threshold = np.argwhere(dist_arr.flatten()<threshold).flatten()
    # keep top k closest videos
    idx_sorted = np.argsort(dist_arr[idx_below_threshold], axis=0).flatten()

    # return indexes of search results
    return idx_below_threshold[idx_sorted][:top_k]
</example>

<render_deployment_guide>
1. Lock down the FastAPI project layout.
   - Move the code that powers your paraphrase search API into an importable module such as `app/main.py`, make sure `model_save/` sits at the repo root, and add `__init__.py` files so `uvicorn` can find `app.main:app`.
   - Add lightweight docs (README) describing request/response payloads so future deploy steps stay reproducible.

2. Record Python dependencies.
   - Create `requirements.txt` (for example: `fastapi`, `uvicorn[standard]`, `pydantic`, `sentence-transformers`, `scikit-learn`, `polars`, `numpy`, `torch`, any utilities you import).
   - Pin versions you have already tested (e.g. `fastapi==0.114.1`) to avoid build-time surprises on Render.

3. Externalize configuration.
   - Introduce environment variables for values that differ between local and cloud runs, e.g. `MODEL_DIR`, `DISTANCE_METRIC`, or thresholds.
   - Add a `.env.example` showing expected keys; never commit secrets.
   - In `app/main.py`, resolve paths from `os.getenv("MODEL_DIR", "model_save")` so the model location is configurable on Render.

4. Decide how to ship the fine-tuned model.
   - The weights in `model_save/` (~438 MB) are too large for a normal Git push. Upload a compressed archive (`model_save.tar.gz`) to storage that Render can reach (GitHub Release asset, S3, Hugging Face repo, Render Static Site, etc.).
   - Add a script such as `scripts/download_model.py` that downloads and extracts the archive into `model_save/` when `MODEL_DIR` does not already exist.
   - Reference the download URL through an env var like `MODEL_DOWNLOAD_URL` so you can rotate locations without code changes.

5. Make startup automation resilient.
   - Create `scripts/prestart.sh` (or a Python equivalent) that: installs system deps if needed, runs `python scripts/download_model.py`, and performs lightweight warm-up inference to populate caches.
   - Ensure the script is executable (`git update-index --chmod=+x scripts/prestart.sh` on Unix). On Windows you can rely on Python scripts only.

6. Provide Render build instructions.
   - Optional but recommended: add a `render.yaml` describing the web service so deployments stay in code. Example:

     ```yaml
     services:
       - type: web
         name: paraphrase-api
         env: python
         plan: free
         buildCommand: pip install -r requirements.txt
         startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
         envVars:
           - key: MODEL_DOWNLOAD_URL
             sync: false
           - key: MODEL_DIR
             value: /opt/render/project/src/model_save
     ```
   - Without `render.yaml`, be ready to enter the same build/start commands manually in the dashboard.

7. Test locally before pushing.
   - Run `pip install -r requirements.txt` in a clean virtualenv, clear any cached paths, and execute `uvicorn app.main:app --reload`.
   - Hit `http://127.0.0.1:8000/` and `/search` with sample sentence pairs to ensure the download script populates `model_save/` and responses match expectations.

8. Prepare the Git repository.
   - Add `.gitignore` entries for `model_save/`, caches, and virtualenvs.
   - `git init`, commit all deployment assets (API code, scripts, requirements, `render.yaml`), and push to GitHub/GitLab/Bitbucket.
   - Enable auto-deploy (e.g. Render needs access to the default branch), so each push can trigger a redeploy.

9. Create the Render web service.
   - In the Render dashboard choose "New +" ? "Web Service", connect the repository, pick the branch, and select the Free plan.
   - Set the build command to `pip install -r requirements.txt` and the start command to `./scripts/prestart.sh && uvicorn app.main:app --host 0.0.0.0 --port $PORT` (omit the shell script if you only rely on Python download logic).
   - Add environment variables: `MODEL_DOWNLOAD_URL`, `MODEL_DIR=/opt/render/project/src/model_save`, any API keys, and optionally `PYTHONUNBUFFERED=1` for better logs.

10. Kick off the first deploy and monitor logs.
    - Watch the build logs for dependency compilation issues (Torch wheels sometimes require extra CPU build time; pin to CPU wheels to avoid GPU requirements).
    - Confirm the download step succeeds and the server starts listening on the supplied `$PORT`.
    - Once "Live", open the public Render URL and hit `/info` or `/docs` to verify the API.

11. Add observability and scaling basics.
    - Configure a Render health check (`/`) so the platform restarts unhealthy instances automatically.
    - If cold starts are slow, keep a lightweight background task to pre-load the model, or upgrade plan when you outgrow free tier limits (512 MB RAM).

12. Document maintenance routines.
    - Record how to rotate the model artifact (upload new archive, update `MODEL_DOWNLOAD_URL`, redeploy).
    - Note the process for regenerating requirements and running smoke tests so future updates remain smooth.
</render_deployment_guide>
# Deployment Guide for BERT Paraphrase Classifier

## 1. Project Structure Setup
```
my-paraphrase-api/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI application
│   └── functions.py   # Helper functions
├── model_save/        # Model files (gitignored)
├── scripts/
│   ├── download_model.py
│   └── prestart.sh
├── .env.example
├── .gitignore
├── requirements.txt
└── render.yaml
```

## 2. Dependencies Setup
Create requirements.txt:
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
sentence-transformers==2.2.2
scikit-learn==1.3.2
numpy==1.26.1
torch==2.1.0+cpu
transformers==4.34.1
```

## 3. Model Preparation
1. Compress your model_save directory:
```bash
tar -czf model_save.tar.gz model_save/
```

2. Upload to a GitHub release or other accessible storage

3. Create download_model.py:
```python
import os
import requests
import tarfile

def download_model():
    model_dir = os.getenv("MODEL_DIR", "model_save")
    if not os.path.exists(model_dir):
        url = os.getenv("MODEL_DOWNLOAD_URL")
        response = requests.get(url, stream=True)
        with open("model_save.tar.gz", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        with tarfile.open("model_save.tar.gz", "r:gz") as tar:
            tar.extractall(".")
        os.remove("model_save.tar.gz")

if __name__ == "__main__":
    download_model()
```

## 4. Configuration Files
Create .env.example:
```
MODEL_DOWNLOAD_URL=https://github.com/your-repo/releases/download/v1.0/model_save.tar.gz
MODEL_DIR=model_save
DISTANCE_METRIC=manhattan
```

Create render.yaml:
```yaml
services:
  - type: web
    name: paraphrase-classifier
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: python scripts/download_model.py && uvicorn app.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: MODEL_DOWNLOAD_URL
        sync: false
      - key: MODEL_DIR
        value: /opt/render/project/src/model_save
```

## 5. Deployment Steps
1. Push code to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

2. Deploy on Render:
- Visit dashboard.render.com
- Click "New +" → "Web Service"
- Connect your GitHub repository
- Select the main branch
- Choose "Free" plan
- Set environment variables:
  - MODEL_DOWNLOAD_URL
  - MODEL_DIR=/opt/render/project/src/model_save
- Click "Create Web Service"

3. Monitor deployment:
- Watch build logs for successful model download
- Verify API is live by visiting the provided Render URL
- Test endpoints:
  - /docs for SwaggerUI
  - / for health check
  - /search for classification

## 6. Testing
Local testing before deployment:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --reload
```

Test API with curl:
```bash
curl -X 'GET' \
  'http://localhost:8000/search?query=your test sentence' \
  -H 'accept: application/json'
```

