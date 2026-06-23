from fastapi import FastAPI

app = FastAPI(
    title="Intelligent OCR API"
)

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "OCR Backend"
    }