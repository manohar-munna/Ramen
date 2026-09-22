import os
import uuid
import base64
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from fastapi import (FastAPI, UploadFile, File, Form, HTTPException,
                     BackgroundTasks, WebSocket, WebSocketDisconnect)
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import pymupdf

from engine import (
    DocumentData, PageData,
    PDFAnalyzer, DigitalExtractor, ScannedExtractor, ImageReconstructor,
    Exporter, FidelityChecker, ModelManager
)
import enhancer
import logging

logger = logging.getLogger("ramen.api")

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

# What each job is doing, for /api/progress. Bounded: this is a desktop server that
# stays up for days, and an unbounded dict keyed by a fresh uuid per upload only ever
# grows. A finished job's line is worth keeping for a while and worthless forever.
JOB_PROGRESS: Dict[str, Dict[str, Any]] = {}
JOB_HISTORY = 64

class CompareRequest(BaseModel):
    origImageSrc: str
    reconImageSrc: str

class ExportRequest(BaseModel):
    document: Dict[str, Any]
    format: str = "zip"  # 'zip', 'html', 'json'

class EnhanceRequest(BaseModel):
    # Either a document to render first, or HTML that is already rendered.
    document: Optional[Dict[str, Any]] = None
    html: Optional[str] = None
    instructions: Optional[str] = None
    model: Optional[str] = None
    flatten: bool = True
    # A data URI or file path. Left unset, the document's own screenshot is used.
    referenceImage: Optional[str] = None
    useReference: bool = True
    # Render the result and let the model compare it with the original, this many times.
    refine: Optional[int] = None
    targetScore: Optional[float] = None


# The phases a conversion actually goes through, as short keys the editor can match
# against without parsing prose. The two paths differ, and saying so is the point: an
# image has no page loop and a digital PDF has no OCR.
STAGE_READ = "read"
STAGE_ANALYZE = "analyze"
STAGE_PAGES = "pages"
STAGE_FINALIZE = "finalize"


def update_job_stage(job_id: str, stage: str, detail: str, percent: int,
                     completed: bool = False, error: Optional[str] = None,
                     key: str = ""):
    while len(JOB_PROGRESS) >= JOB_HISTORY and job_id not in JOB_PROGRESS:
        JOB_PROGRESS.pop(next(iter(JOB_PROGRESS)))
    JOB_PROGRESS[job_id] = {
        "jobId": job_id,
        "stage": stage,
        "key": key,
        "detail": detail,
        "percent": percent,
        "completed": completed,
        "error": error
    }


_ID_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")


def job_id_from(requested: Optional[str]) -> str:
    """The caller's job id if it is one, otherwise a fresh one.

    The caller chooses it so it can watch the conversion from the first moment. It
    only ever keys an in-memory dict, but it is also part of a path, so anything that
    is not a plain identifier is thrown away rather than sanitised into something
    surprising.
    """
    if requested and len(requested) <= 40 and set(requested) <= _ID_OK:
        return requested
    return f"job-{uuid.uuid4().hex[:8]}"

@app.get("/")
def get_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="index.html not found")

@app.get("/sample.pdf")
def get_sample_pdf():
    sample_path = os.path.join(BASE_DIR, "benchmarks", "pdfs", "sample.pdf")
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

def _reconstruct(file_path: str, filename: str,
                 job_id: str, asset_dir: str) -> Tuple[DocumentData, Dict[str, Any]]:
    """The whole reconstruction, start to finish, on a worker thread.

    Split out from the route because it is seconds to minutes of solid CPU. Called
    inline from an async handler it held the event loop for the entire conversion:
    every other request queued behind it, including the progress endpoint that is
    supposed to say how the conversion is going, and the live-build socket, which
    could not even be opened while a page was being read.
    """
    ext = os.path.splitext(filename)[1].lower()
    is_image = ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']
    doc_title = os.path.splitext(filename)[0]

    if is_image:
        update_job_stage(job_id, "Reading the picture",
                         "Colours, typography, shapes and photographic regions...", 25,
                         key=STAGE_ANALYZE)
        p_data = ImageReconstructor.reconstruct_image(file_path, asset_dir=asset_dir)
        update_job_stage(job_id, "Assembling the page",
                         "Placing every element at its measured coordinates...", 90,
                         key=STAGE_FINALIZE)
        analysis = {
            "pageCount": 1,
            "isScannedDoc": True,
            "pages": [{"page": 1, "isScanned": True, "textLength": len(p_data.elements)}]
        }
        return DocumentData(
            title=doc_title,
            pageCount=1,
            pages=[p_data],
            metadata={"filename": filename, "isImage": True, "analysis": analysis}
        ), analysis

    update_job_stage(job_id, "Reading the document",
                     "Telling digital pages from scanned ones...", 15, key=STAGE_ANALYZE)
    analysis = PDFAnalyzer.analyze_document(file_path)
    page_count = analysis["pageCount"]
    reconstructed_pages: List[PageData] = []

    # Closed explicitly. Left to the garbage collector the handle outlived the
    # request, and on Windows that keeps the cached upload locked against deletion.
    doc = pymupdf.open(file_path)
    try:
        for p_idx in range(page_count):
            p_num = p_idx + 1
            pct = int(20 + (p_idx / max(page_count, 1)) * 70)
            if analysis["pages"][p_idx]["isScanned"]:
                update_job_stage(
                    job_id, "Reading the pages",
                    f"Page {p_num} of {page_count}: scanned, so layout and OCR...",
                    pct, key=STAGE_PAGES)
                p_data = ScannedExtractor.extract_page(doc, p_idx, asset_dir=asset_dir)
            else:
                update_job_stage(
                    job_id, "Reading the pages",
                    f"Page {p_num} of {page_count}: geometry, fonts, tables, vectors...",
                    pct, key=STAGE_PAGES)
                p_data = DigitalExtractor.extract_page(doc, p_idx, asset_dir=asset_dir)
            reconstructed_pages.append(p_data)
    finally:
        doc.close()

    update_job_stage(job_id, "Assembling the page",
                     "Placing every element at its measured coordinates...", 95,
                     key=STAGE_FINALIZE)
    return DocumentData(
        title=doc_title,
        pageCount=page_count,
        pages=reconstructed_pages,
        metadata={"filename": filename, "analysis": analysis}
    ), analysis


@app.post("/api/convert")
async def convert_pdf(file: UploadFile = File(...),
                      jobId: Optional[str] = Form(None)):
    job_id = job_id_from(jobId)
    filename = file.filename or "upload"
    file_path = os.path.join(CACHE_DIR, f"{job_id}_{filename}")

    update_job_stage(job_id, "Reading the file", f"Receiving {filename}...", 5,
                     key=STAGE_READ)

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    asset_dir = os.path.join(OUTPUTS_DIR, job_id, "assets")
    os.makedirs(asset_dir, exist_ok=True)

    try:
        final_doc, analysis = await asyncio.to_thread(
            _reconstruct, file_path, filename, job_id, asset_dir)
        update_job_stage(job_id, "Done", "Reconstructed.", 100, completed=True)
        return {
            "jobId": job_id,
            "document": final_doc.model_dump(),
            "analysis": analysis
        }

    except ValueError as ve:
        update_job_stage(job_id, "Failed", str(ve), 0, completed=True, error=str(ve))
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("conversion failed")
        update_job_stage(job_id, "Failed", str(e), 0, completed=True, error=str(e))
        raise HTTPException(status_code=500, detail=f"Conversion error: {e}")

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
    except ValidationError as e:
        raise HTTPException(status_code=400,
                            detail=f"That document did not validate: {e}")
    except Exception as e:
        # A 400 here blamed the caller for a fault in the exporter, which is the same
        # mistake /api/enhance made and the same wasted afternoon looking at the
        # request. The traceback goes to the log; the status says where to look.
        logger.exception("export failed")
        raise HTTPException(
            status_code=500,
            detail=f"Export failed inside the server: {type(e).__name__}: {e}. "
                   f"The full traceback is in the server log.")

@app.websocket("/ws/generate")
async def generate_live(ws: WebSocket):
    """Writes a page from the screenshot and streams it as it is written.

    A socket rather than a request because the interesting part is the middle: the page
    assembles over a minute or two and watching it happen is most of the value. Each
    chunk is forwarded the moment it arrives, so the editor can paint a live preview.
    """
    await ws.accept()
    try:
        request = await ws.receive_json()
    except Exception:
        await ws.close(code=1003)
        return

    doc_raw = request.get("document")
    if not doc_raw:
        await ws.send_json({"type": "error", "detail": "Send {'document': ...}."})
        await ws.close()
        return

    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue" = asyncio.Queue()
    # Set when the browser goes away. Generation costs a minute of model time and a
    # slice of a daily quota, and it used to run to completion for a window that had
    # already been closed -- every chunk forwarded into a queue with no reader.
    abandoned = False

    def produce():
        """Runs the blocking stream on a worker thread and feeds the queue."""
        try:
            doc = DocumentData.model_validate(doc_raw)
            for kind, payload in enhancer.generate_from_document(
                    doc, model=request.get("model"),
                    verify=int(request.get("verify", enhancer.VERIFY_ROUNDS)),
                    target_score=float(request.get("targetScore",
                                                   enhancer.TARGET_ACCURACY_SCORE)),
                    page_index=int(request.get("page", 0))):
                if abandoned:
                    logger.info("live generation abandoned by the client")
                    return
                loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))
        except enhancer.EnhancementError as e:
            # The socket's error goes straight into an alert, so it gets the readable
            # form. str(e) put a wall of JSON in front of the user.
            loop.call_soon_threadsafe(
                queue.put_nowait, ("error", enhancer.short_reason(e)))
        except Exception as e:
            logger.exception("live generation failed")
            loop.call_soon_threadsafe(
                queue.put_nowait, ("error", f"{type(e).__name__}: {e}"))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("eof", None))

    await ws.send_json({"type": "status", "detail": "Reading the screenshot..."})
    loop.run_in_executor(None, produce)

    try:
        while True:
            kind, payload = await queue.get()
            if kind == "eof":
                break
            if kind == "status":
                await ws.send_json({"type": "status", "detail": payload})
            elif kind == "phase":
                # The stage it is in, for the progress bar's heading. The editor read a
                # `phase` field that nothing ever sent, so the heading said "Processing"
                # from start to finish.
                await ws.send_json({"type": "phase", "label": payload})
            elif kind == "assets":
                await ws.send_json({"type": "assets", "assets": payload})
            elif kind == "issue":
                await ws.send_json({"type": "issue", "detail": payload})
            elif kind == "score":
                await ws.send_json({"type": "score", **payload})
            elif kind == "revision":
                await ws.send_json({"type": "revision", **payload})
            elif kind == "chunk":
                await ws.send_json({"type": "chunk", "text": payload})
            elif kind == "done":
                # The finished page carries megabytes of restored image data, so it goes
                # as its own message rather than through the chunk stream.
                await ws.send_json({"type": "done", **payload})
            elif kind == "error":
                await ws.send_json({"type": "error", "detail": payload})
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("live generation socket failed")
    finally:
        # Tells the worker to stop at its next yield rather than writing a page for
        # a window that is no longer there.
        abandoned = True
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/api/enhance/status")
def enhance_status():
    """Whether the enhancement pass can run. Never returns the key itself."""
    return {
        "configured": enhancer.is_configured(),
        "model": os.environ.get("GEMINI_MODEL") or enhancer.DEFAULT_MODEL,
    }

@app.post("/api/enhance")
def enhance_document(req: EnhanceRequest):
    """Rewrites a reconstruction into a laid-out, animated page via Gemini.

    Strictly optional and strictly downstream: nothing here feeds back into the engine,
    and the faithful reconstruction the caller sent is unchanged by it. The response
    carries both the new HTML and what the model did to the images, because a page that
    quietly lost one is worse than an error.
    """
    if not req.html and not req.document:
        raise HTTPException(status_code=400, detail="Send either 'document' or 'html'.")
    try:
        reference = req.referenceImage if req.useReference else None
        if req.html:
            # Flattening needs the element model; raw HTML has already lost it.
            html_content = req.html
        else:
            doc_data = DocumentData.model_validate(req.document)
            # Every reconstruction carries the screenshot it came from, so the model can
            # be shown what the page is supposed to look like without the caller having
            # to send it twice.
            if reference is None and req.useReference:
                reference = enhancer.reference_from_document(doc_data)
            if req.flatten:
                # Compose the raster fragments first. They only form a picture at their
                # measured coordinates, and the rewrite puts everything into flow.
                doc_data = enhancer.flatten_document(doc_data)
            html_content = Exporter.export_standalone_html(doc_data, interactive=False)
        result = enhancer.enhance_html(
            html_content, model=req.model, extra=req.instructions, reference=reference,
            refine=(enhancer.REFINE_ROUNDS if req.refine is None else req.refine),
            target_score=(req.targetScore if req.targetScore is not None else enhancer.TARGET_ACCURACY_SCORE),
        )
        return result
    except enhancer.EnhancementError as e:
        # The model or the key is the problem, not the request. 502 says so.
        raise HTTPException(status_code=502, detail=enhancer.short_reason(e))
    except ValidationError as e:
        raise HTTPException(status_code=400,
                            detail=f"That document did not validate: {e}")
    except Exception as e:
        # Anything reaching here is a fault on this side, and 400 said the opposite --
        # it sent one debugging session looking at the browser's payload when the real
        # cause was the server holding a half-reloaded module. Log the traceback and
        # name the exception type, so the next one is readable from either end.
        logger.exception("enhancement failed")
        raise HTTPException(
            status_code=500,
            detail=f"Enhancement failed inside the server: "
                   f"{type(e).__name__}: {e}. The full traceback is in the server log.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
