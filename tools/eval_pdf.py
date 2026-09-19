"""Fidelity evaluation for the digital-PDF path.

Usage:  python tools/eval_pdf.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pymupdf
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from engine import DigitalExtractor, HTMLRenderer, DocumentData
from _chrome import find_chrome, screenshot

CHROME = find_chrome()
pdf_path = "benchmarks/pdfs/reconstruction-20-pages.pdf"
doc = pymupdf.open(pdf_path)

os.makedirs("scratch/pdf_eval", exist_ok=True)
scores = []

for p_idx in range(len(doc)):
    page_num = p_idx + 1
    page = doc[p_idx]
    
    # 1. PDF rendering at 150 DPI
    pix = page.get_pixmap(dpi=150)
    actual_img_path = os.path.abspath(f"scratch/pdf_eval/eval_actual_p{page_num}.png")
    pix.save(actual_img_path)
    
    # 2. Reconstruct HTML
    page_data = DigitalExtractor.extract_page(doc, p_idx, asset_dir="scratch/pdf_eval/assets")
    doc_data = DocumentData(
        title=f"Page {page_num}",
        pageCount=1,
        pages=[page_data]
    )
    html_content = HTMLRenderer.render_document(doc_data, editable=False, title=f"Page {page_num}")
    html_content = html_content.replace("padding: 24px;", "padding: 0;").replace("gap: 24px;", "gap: 0;")
    html_path = os.path.abspath(f"scratch/pdf_eval/eval_p{page_num}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    # 3. Headless Chrome screenshot
    recon_img_path = os.path.abspath(f"scratch/pdf_eval/eval_recon_p{page_num}.png")
    screenshot(CHROME, html_path, recon_img_path,
               page.rect.width, page.rect.height, scale=150.0 / 72.0)

    # 4. Compute SSIM
    im1 = Image.open(actual_img_path).convert("L")
    im2 = Image.open(recon_img_path).convert("L")
    w, h = min(im1.width, im2.width), min(im1.height, im2.height)
    arr1 = np.array(im1.crop((0, 0, w, h)))
    arr2 = np.array(im2.crop((0, 0, w, h)))
    
    score, _ = ssim(arr1, arr2, full=True)
    scores.append(score * 100.0)
    print(f"Page {page_num:2d}: SSIM = {score * 100.0:.2f}%")

print(f"\nAverage SSIM across 20 pages: {np.mean(scores):.2f}%")
