import os
import io
import pytest
from fastapi.testclient import TestClient
import pymupdf

from app import app
from engine import (
    DocumentData, PageData, DocumentElement, TextStyle,
    PDFAnalyzer, DigitalExtractor, ScannedExtractor, TableExtractor,
    HTMLRenderer, Exporter, FidelityChecker
)

client = TestClient(app)

def test_health_and_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "RAMEN" in response.text
    assert "<style>" in response.text
    assert "<script>" in response.text

def test_models_status_endpoint():
    response = client.get("/api/models/status")
    assert response.status_code == 200
    data = response.json()
    assert "components" in data
    assert len(data["components"]) >= 3

def test_sample_pdf_endpoint():
    response = client.get("/sample.pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"

def test_digital_extraction_and_rendering():
    # Create simple in-memory digital PDF
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Financial Performance Summary", fontsize=18)
    page.insert_text((72, 130), "Revenue increased by 25% year-over-year.", fontsize=11)
    page.draw_rect(pymupdf.Rect(72, 160, 523, 220), color=(0.2, 0.4, 0.8), fill=(0.9, 0.95, 1.0))
    pdf_bytes = doc.tobytes()
    doc.close()

    test_doc = pymupdf.open("pdf", pdf_bytes)
    page_data = DigitalExtractor.extract_page(test_doc, 0)
    assert page_data.pageNumber == 1
    assert not page_data.isScanned
    assert len(page_data.elements) >= 2

    # Render HTML
    doc_data = DocumentData(title="Test Report", pageCount=1, pages=[page_data])
    html_output = HTMLRenderer.render_document(doc_data)
    assert "Financial Performance Summary" in html_output
    assert "position: absolute" in html_output

    # Exporter tests
    zip_bytes = Exporter.export_zip_bundle(doc_data)
    assert len(zip_bytes) > 0

    json_str = Exporter.export_json(doc_data)
    assert "Financial Performance Summary" in json_str

def test_table_extraction_validation():
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    p1 = pymupdf.Point(50, 50)
    p2 = pymupdf.Point(500, 50)
    page.draw_line(p1, p2)
    tables = TableExtractor.extract_digital_tables(page)
    # A single line must NOT be recognized as a valid table
    assert len(tables) == 0

def test_convert_endpoint():
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=500)
    page.insert_text((50, 50), "Test Document Title", fontsize=14)
    pdf_bytes = doc.tobytes()
    doc.close()

    res = client.post("/api/convert", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
    assert res.status_code == 200
    data = res.json()
    assert "document" in data
    assert data["document"]["pageCount"] == 1

def test_fidelity_checker_and_compare_endpoint():
    import numpy as np
    import cv2
    import base64

    # Create two slightly different images
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 250
    _, enc1 = cv2.imencode(".png", img1)
    _, enc2 = cv2.imencode(".png", img2)

    res = FidelityChecker.compare_images(enc1.tobytes(), enc2.tobytes())
    assert "ssim" in res
    assert res["similarityPercent"] > 90.0

    b64_1 = f"data:image/png;base64,{base64.b64encode(enc1.tobytes()).decode('utf-8')}"
    b64_2 = f"data:image/png;base64,{base64.b64encode(enc2.tobytes()).decode('utf-8')}"
    resp = client.post("/api/compare", json={"origImageSrc": b64_1, "reconImageSrc": b64_2})
    assert resp.status_code == 200
    assert "ssim" in resp.json()

def test_export_endpoint():
    doc_dict = {
        "title": "Export Test",
        "pageCount": 1,
        "pages": [{
            "pageNumber": 1,
            "width": 600.0,
            "height": 800.0,
            "isScanned": False,
            "elements": [{
                "id": "t1",
                "type": "text",
                "bbox": [50.0, 50.0, 200.0, 70.0],
                "text": "Exported Text"
            }]
        }],
        "metadata": {}
    }
    # Test HTML export
    res_html = client.post("/api/export", json={"document": doc_dict, "format": "html"})
    assert res_html.status_code == 200
    assert "Exported Text" in res_html.text

    # Test JSON export
    res_json = client.post("/api/export", json={"document": doc_dict, "format": "json"})
    assert res_json.status_code == 200
    assert "Exported Text" in res_json.text

    # Test ZIP export
    res_zip = client.post("/api/export", json={"document": doc_dict, "format": "zip"})
    assert res_zip.status_code == 200
    assert len(res_zip.content) > 0

def test_clean_table_data_and_styling():
    # Test phantom boundary column stripping and callout detection
    raw_data = [
        ["", "Property", "Expected", ""],
        ["", "Asset type", "Raster image", ""],
        ["", "Editing", "Image remains independent", ""],
        ["EXPECTED BEHAVIOUR NOTE THAT IS MERGED ACROSS ALL COLUMNS", "", "", ""]
    ]
    cleaned_rows, num_cols = TableExtractor.clean_table_data(raw_data)
    assert num_cols == 2
    assert len(cleaned_rows) == 4
    assert cleaned_rows[0]["cells"] == ["Property", "Expected"]
    assert cleaned_rows[1]["cells"] == ["Asset type", "Raster image"]
    assert cleaned_rows[3]["is_merged"] is True
    assert "EXPECTED BEHAVIOUR" in cleaned_rows[3]["text"]


