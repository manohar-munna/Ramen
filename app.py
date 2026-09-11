import os
import sys
import uuid
import json
import base64
import asyncio
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymupdf

from engine import (
    DocumentData, PageData, DocumentElement,
    PDFAnalyzer, DigitalExtractor, ScannedExtractor, ImageReconstructor,
    HTMLRenderer, Exporter, FidelityChecker, ModelManager
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI(
    title="Ramen — Precision PDF Reconstruction & Editor",
    description="Deterministic 2D-coordinate PDF-to-HTML converter without VLM/LLM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# In-memory tracking
JOB_PROGRESS: Dict[str, Dict[str, Any]] = {}
CONVERTED_DOCUMENTS: Dict[str, DocumentData] = {}

class CompareRequest(BaseModel):
    origImageSrc: str
    reconImageSrc: str

class ExportRequest(BaseModel):
    document: Dict[str, Any]
    format: str = "zip"  # 'zip', 'html', 'json'

def update_job_stage(job_id: str, stage: str, detail: str, percent: int, completed: bool = False, error: str = None):
    JOB_PROGRESS[job_id] = {
        "jobId": job_id,
        "stage": stage,
        "detail": detail,
        "percent": percent,
        "completed": completed,
        "error": error
    }

@app.get("/")
def get_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="index.html not found")

@app.get("/sample.pdf")
def get_sample_pdf():
    sample_path = os.path.join(BASE_DIR, "sample.pdf")
    if os.path.exists(sample_path):
        return FileResponse(sample_path, media_type="application/pdf")
    raise HTTPException(status_code=404, detail="sample.pdf not found")

@app.get("/api/models/status")
def get_models_status():
    manager = ModelManager()
    return manager.get_all_status()

@app.post("/api/models/download")
def download_models(background_tasks: BackgroundTasks):
    manager = ModelManager()
    background_tasks.add_task(manager.download_models)
    return {"message": "Model verification & download initiated in background"}

@app.get("/api/progress/{job_id}")
def get_progress(job_id: str):
    if job_id not in JOB_PROGRESS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOB_PROGRESS[job_id]

@app.post("/api/convert")
async def convert_pdf(file: UploadFile = File(...)):
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    file_path = os.path.join(CACHE_DIR, f"{job_id}_{file.filename}")

    update_job_stage(job_id, "Reading PDF", f"Receiving {file.filename}...", 5)

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        ext = os.path.splitext(file.filename)[1].lower()
        is_image = ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']
        asset_dir = os.path.join(OUTPUTS_DIR, job_id, "assets")
        os.makedirs(asset_dir, exist_ok=True)
        doc_title = os.path.splitext(file.filename)[0]

        if is_image:
            update_job_stage(job_id, "Layered Visual Engine", "Analyzing image resolution, color palette, typography & shapes...", 25)
            p_data = ImageReconstructor.reconstruct_image(file_path, asset_dir=asset_dir)
            update_job_stage(job_id, "Finalizing HTML", "Assembling 2D coordinate canvas and intermediate JSON...", 90)
            analysis = {
                "pageCount": 1,
                "isScannedDoc": True,
                "pages": [{"page": 1, "isScanned": True, "textLength": len(p_data.elements)}]
            }
            final_doc = DocumentData(
                title=doc_title,
                pageCount=1,
                pages=[p_data],
                metadata={
                    "filename": file.filename,
                    "isImage": True,
                    "analysis": analysis
                }
            )
        else:
            update_job_stage(job_id, "Analyzing Document Structure", "Detecting digital objects vs scanned pages...", 15)
            analysis = PDFAnalyzer.analyze_document(file_path)
            page_count = analysis["pageCount"]

            doc = pymupdf.open(file_path)
            reconstructed_pages: List[PageData] = []

            for p_idx in range(page_count):
                p_info = analysis["pages"][p_idx]
                is_scanned = p_info["isScanned"]
                p_num = p_idx + 1
                pct = int(20 + (p_idx / max(page_count, 1)) * 70)

                if is_scanned:
                    update_job_stage(
                        job_id,
                        "Layout & OCR Analysis",
                        f"Processing scanned page {p_num} of {page_count} (PP-DocLayout / OCR / Table)...",
                        pct
                    )
                    p_data = ScannedExtractor.extract_page(doc, p_idx, asset_dir=asset_dir)
                else:
                    update_job_stage(
                        job_id,
                        "Native Extraction",
                        f"Extracting digital geometry, fonts, tables & vectors for page {p_num} of {page_count}...",
                        pct
                    )
                    p_data = DigitalExtractor.extract_page(doc, p_idx, asset_dir=asset_dir)

                reconstructed_pages.append(p_data)

            update_job_stage(job_id, "Finalizing HTML", "Assembling 2D coordinate canvas and intermediate JSON...", 95)
            final_doc = DocumentData(
                title=doc_title,
                pageCount=page_count,
                pages=reconstructed_pages,
                metadata={
                    "filename": file.filename,
                    "analysis": analysis
                }
            )

        CONVERTED_DOCUMENTS[job_id] = final_doc
        update_job_stage(job_id, "Completed", "Document successfully reconstructed!", 100, completed=True)

        return {
            "jobId": job_id,
            "document": final_doc.model_dump(),
            "analysis": analysis
        }

    except ValueError as ve:
        update_job_stage(job_id, "Failed", str(ve), 0, completed=True, error=str(ve))
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job_stage(job_id, "Failed", str(e), 0, completed=True, error=str(e))
        raise HTTPException(status_code=500, detail=f"Conversion error: {str(e)}")

@app.post("/api/compare")
def compare_fidelity(req: CompareRequest):
    try:
        orig_b64 = req.origImageSrc.split(",")[-1]
        orig_bytes = base64.b64decode(orig_b64)
        recon_b64 = req.reconImageSrc.split(",")[-1]
        recon_bytes = base64.b64decode(recon_b64)
        result = FidelityChecker.compare_images(orig_bytes, recon_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fidelity comparison failed: {str(e)}")

@app.post("/api/export")
def export_document(req: ExportRequest):
    try:
        doc_data = DocumentData.model_validate(req.document)
        fmt = req.format.lower()

        if fmt == "html":
            html_content = Exporter.export_standalone_html(doc_data)
            return Response(
                content=html_content,
                media_type="text/html",
                headers={"Content-Disposition": f"attachment; filename={doc_data.title or 'document'}.html"}
            )
        elif fmt == "json":
            json_content = Exporter.export_json(doc_data)
            return Response(
                content=json_content,
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename={doc_data.title or 'document'}.json"}
            )
        else:  # zip bundle
            zip_bytes = Exporter.export_zip_bundle(doc_data)
            return Response(
                content=zip_bytes,
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={doc_data.title or 'document'}_bundle.zip"}
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Export failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
