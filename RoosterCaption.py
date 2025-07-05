import os
import gradio as gr
from PIL import Image, ExifTags
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
import numpy as np
import tempfile
import csv
from io import StringIO

# --- MODELE ---
os.environ["TRANSFORMERS_CACHE"] = os.path.abspath("./models")
os.environ["HF_HOME"] = os.path.abspath("./models")

caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# --- GLOBAL STATE ---
IMAGES = []

def get_exif(img):
    try:
        exif = img._getexif()
        if not exif:
            return {}
        labeled = {}
        for (k, v) in exif.items():
            label = ExifTags.TAGS.get(k)
            if label:
                labeled[label] = v
        return labeled
    except Exception:
        return {}

def make_caption(image):
    inputs = caption_processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    return caption_processor.decode(out[0], skip_special_tokens=True)

def extract_tags(caption):
    words = set([w.strip(".,!?").lower() for w in caption.split() if len(w) > 3])
    return sorted(words)

def process_images(files, user_tags=""):
    global IMAGES
    IMAGES = []
    tags_user = [t.strip().lower() for t in user_tags.split(",") if t.strip()]
    for file in files:
        img = Image.open(file.name).convert("RGB")
        caption = make_caption(img)
        embedding = embedder.encode([caption])[0]
        tags = extract_tags(caption)
        tags_all = sorted(set(tags + tags_user))
        exif = get_exif(img)
        IMAGES.append({
            "img": img,
            "caption": caption,
            "embedding": embedding,
            "filename": os.path.basename(file.name),
            "tags": tags_all,
            "exif": exif,
            "path": file.name,
        })
    all_tags = sorted(list({t for im in IMAGES for t in im["tags"]}))
    return f"Processed {len(IMAGES)} images.", all_tags, len(IMAGES)

def search_images(query, tags_filter=None, sort_by="score", top_k=30):
    if not IMAGES:
        return [], 0
    if not query and not tags_filter:
        results = [
            (im["img"], make_html_caption(im, 1.0))
            for im in IMAGES
        ]
        return results, len(results)

    filtered = IMAGES
    if tags_filter:
        filtered = [im for im in filtered if any(tag in im["tags"] for tag in tags_filter)]
    if query:
        q_emb = embedder.encode([query])[0]
        embs = [x["embedding"] for x in filtered]
        sims = np.dot(embs, q_emb) / (
            np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb) + 1e-9
        )
        results = []
        for idx, sim in enumerate(sims):
            if sim < 0.18:
                continue
            entry = filtered[idx]
            results.append((entry["img"], make_html_caption(entry, sim)))
        # Sortowanie
        if sort_by == "score":
            def extract_score(caption):
                import re
                m = re.search(r"\(sim: ([0-9.]+)\)", caption)
                return float(m.group(1)) if m else 0
            results = sorted(results, key=lambda x: -extract_score(x[1]))
        elif sort_by == "name":
            results = sorted(results, key=lambda x: entry['filename'])
        elif sort_by == "date":
            results = sorted(results, key=lambda x: entry.get("exif", {}).get("DateTimeOriginal", ""))
        return results, len(results)
    else:
        # Only tags filter
        results = [
            (im["img"], make_html_caption(im, 1.0))
            for im in filtered
        ]
        return results, len(results)

def make_html_caption(entry, score=None):
    cap = entry["caption"]
    tags = ", ".join(entry["tags"])
    exif = entry.get("exif", {})
    exif_short = []
    if exif:
        if "Model" in exif:
            exif_short.append(f"<b>Cam:</b> {exif['Model']}")
        if "DateTimeOriginal" in exif:
            exif_short.append(f"<b>Date:</b> {exif['DateTimeOriginal']}")
        if "ExifImageWidth" in exif and "ExifImageHeight" in exif:
            exif_short.append(f"<b>Res:</b> {exif['ExifImageWidth']}x{exif['ExifImageHeight']}")
    score_html = f"<span style='color:#9ca3af;'>(sim: {score:.2f})</span>" if score is not None else ""
    html = f"""<div>
    <b>{entry['filename']}</b> {score_html}<br>
    <span style='font-size:1.09em;'>{cap}</span><br>
    <span style='color:#be185d;font-size:0.99em;'>Tags:</span> {tags}<br>
    {" | ".join(exif_short)}
    </div>"""
    return html

def export_results_to_csv(query, tags_filter):
    results, _ = search_images(query, tags_filter, top_k=len(IMAGES))
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["caption"])
    for _, html in results:
        import re
        cap = re.sub(r"<.*?>", "", html)
        cap = cap.strip()
        writer.writerow([cap])
    output.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".csv") as tmp:
        tmp.write(output.getvalue())
        return tmp.name

def get_full_image(idx):
    if idx is None or not IMAGES or idx >= len(IMAGES):
        return gr.Image.update(visible=False)
    img = IMAGES[idx]["img"]
    return gr.Image.update(value=img, visible=True)

def download_image(idx):
    if idx is None or not IMAGES or idx >= len(IMAGES):
        return None
    img = IMAGES[idx]["img"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name, format="JPEG")
        return tmp.name

# --- UI ---
custom_css = """
body, .gradio-container {
    font-family: 'Inter', Arial, sans-serif !important;
    background: #f6f7fb !important;
}
@media (prefers-color-scheme: dark) {
    body, .gradio-container {
        background: #1a1c22 !important;
        color: #f1f5f9 !important;
    }
    #gallery img {box-shadow:0 2px 6px #0009;}
}
#gallery img {border-radius:1rem;box-shadow:0 3px 8px #0001;}
.card-img {border-radius:1rem;}
#custom-footer {color:#6b7280;}
/* Ukryj stopkę Gradio */
.footer-wrap, .svelte-1ipelgc {display:none !important;}
"""

with gr.Blocks(css=custom_css, title="Rooster Image Captioner") as demo:
    gr.Markdown("""
    <div style="display:flex;align-items:center;gap:1.5rem;margin-bottom:2rem;">
      <img src="file/rooster_logo.png" style="height:52px;">
      <div>
        <span style="font-size:2.2rem;font-weight:900;letter-spacing:2px;color:#b91c1c;">Rooster Image Captioner</span><br>
        <span style="font-size:1.1rem;color:#4b5563;font-weight:500;">AI captions + semantic search for your photos, offline</span>
      </div>
    </div>
    """)
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload images (JPG/PNG, you can select folder!)", file_types=[".jpg", ".jpeg", ".png"], file_count="multiple")
            tag_input = gr.Textbox(label="Add your own tags (comma separated)", placeholder="e.g. holiday, family, event")
            btn_process = gr.Button("Process images")
            process_status = gr.Markdown()
            tagbox = gr.CheckboxGroup(label="Filter by tag", choices=[], interactive=True)
            sort_box = gr.Radio(["score", "name", "date"], value="score", label="Sort by")
            btn_export = gr.Button("Export results to CSV")
            csv_file = gr.File(label="Download CSV", visible=False)
        with gr.Column(scale=2):
            search_box = gr.Textbox(label="Search images by text (semantic, AI-powered)", placeholder="Type e.g. mountain, people, sunset, car ...")
            gallery = gr.Gallery(label="Results", columns=[4], rows=[3], height="auto", elem_id="gallery")
            counter = gr.Markdown("No results yet.", elem_id="counter")
            img_popup = gr.Image(visible=False, label="Preview", type="pil")
            btn_download = gr.Button("Download selected image", visible=False)
            hidden_idx = gr.Number(value=None, visible=False)

    def on_process(files, user_tags):
        status, all_tags, n = process_images(files, user_tags)
        return status, gr.CheckboxGroup.update(choices=all_tags, value=[]), [], "No results yet.", gr.Image.update(visible=False), gr.File.update(visible=False), None

    def on_search(query, tags, sort_by):
        results, count = search_images(query, tags, sort_by)
        text = f"Showing {count} / {len(IMAGES)} image(s)." if results else "No results found."
        return results, text

    def on_tag_filter(tags, query, sort_by):
        results, count = search_images(query, tags, sort_by)
        text = f"Showing {count} / {len(IMAGES)} image(s)." if results else "No results found."
        return results, text

    def on_sort(sort_by, query, tags):
        results, count = search_images(query, tags, sort_by)
        text = f"Showing {count} / {len(IMAGES)} image(s)." if results else "No results found."
        return results, text

    def on_export(query, tags):
        fpath = export_results_to_csv(query, tags)
        return gr.File.update(value=fpath, visible=True)

    def open_image_popup(evt: gr.SelectData):
        idx = evt.index
        if idx is None or not IMAGES or idx >= len(IMAGES):
            return gr.Image.update(visible=False), gr.Number.update(value=None), gr.Button.update(visible=False)
        img = IMAGES[idx]["img"]
        return gr.Image.update(value=img, visible=True), gr.Number.update(value=idx), gr.Button.update(visible=True)

    def on_download(idx):
        idx = int(idx)
        fname = download_image(idx)
        return gr.File.update(value=fname, visible=True)

    btn_process.click(on_process, [file_input, tag_input], [process_status, tagbox, gallery, counter, img_popup, csv_file, hidden_idx])
    search_box.submit(on_search, [search_box, tagbox, sort_box], [gallery, counter])
    tagbox.change(on_tag_filter, [tagbox, search_box, sort_box], [gallery, counter])
    sort_box.change(on_sort, [sort_box, search_box, tagbox], [gallery, counter])
    btn_export.click(on_export, [search_box, tagbox], [csv_file])
    gallery.select(open_image_popup, None, [img_popup, hidden_idx, btn_download])
    btn_download.click(on_download, [hidden_idx], [csv_file])

    gr.Markdown("""
    <hr>
    <div id="custom-footer" style="text-align:center;margin-top:2rem;">
      <img src="file/rooster_logo.png" style="height:28px;vertical-align:middle;">
      <span style="font-weight:700;font-size:1.1rem;vertical-align:middle;">Rooster Image Captioner</span> &copy; 2025
    </div>
    """)

demo.launch(inbrowser=True)
