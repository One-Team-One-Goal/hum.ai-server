from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import analyze

app = FastAPI(
    title="Hum.AI Rice Grading API",
    description="API for analyzing rice grain images and providing grading metrics.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# routers go here
app.include_router(analyze.router)

# health check endpoints ni here
@app.get("/", tags=["Health"])
async def root():
    return {"message": "Hum.AI Rice Grading API", "status": "running"}

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}