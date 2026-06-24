from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.api.routes.upload import router as upload_router
from backend.api.routes.preprocess import router as preprocess_router
from backend.api.routes.ocr import router as ocr_router
from backend.api.routes.correct import router as correct_router
from backend.api.routes.classify import router as classify_router
from backend.api.routes.extract import router as extract_router
from backend.api.routes.validate import router as validate_router
from backend.api.routes.qa import router as qa_router

app = FastAPI(
    title="Intelligent OCR API",
    version="1.0.0"
)
app.mount(
    "/static",
    StaticFiles(directory="backend/static"),
    name="static"
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(
        "backend/static/favicon.ico"
    )
app.include_router(upload_router)
app.include_router(preprocess_router)
app.include_router(ocr_router)
app.include_router(correct_router)
app.include_router(classify_router)
app.include_router(extract_router)
app.include_router(validate_router)
app.include_router(qa_router)


@app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "OCR Backend"
    }