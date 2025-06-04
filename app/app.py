from fastapi import FastAPI
from app.routes.api import router as api_router
from app.core.models import AIModels
from config import settings
from fastapi import FastAPI


app = FastAPI(title=settings.APP_NAME)

def load_models():
    if settings.DEBUG:
        print(f"Loading models: {settings.model_paths}")
    
    nlp = spacy.load(settings.SPACY_MODEL)

app = FastAPI(title="French Grammar AI")
models = AIModels()

@app.on_event("startup")
async def startup_event():
    models.load_models()

app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)