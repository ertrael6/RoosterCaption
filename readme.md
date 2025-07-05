# Rooster Image Captioner

AI-powered app for generating English image captions and searching your photo collection using semantic (meaning-based) queries. All runs offline, no data is sent anywhere.

## Features

- **Automatic captioning** of all your photos (JPG/PNG), using a local AI model (BLIP).
- **Semantic search**: find images by what is *in* them, not just filename.
- **Custom tags**: add your own labels to photos.
- **EXIF support**: see camera, date, and resolution info.
- **Export**: save all captions as CSV for further use.
- **Modern UI**: drag-and-drop, filters, no Gradio branding.

## Usage

1. **Clone/download this repo and place your images in a folder (or just select them in the app).**

2. **Install requirements** (Python 3.9+):
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the app**:
    ```bash
    python RoosterCaption.py
    ```

4. The app will open in your browser (`http://127.0.0.1:7860` by default).

5. **Drag & drop images (or folder)**, click "Process images", and search by content.

## Model downloads

- The first run will automatically download models:
    - `Salesforce/blip-image-captioning-base`
    - `all-MiniLM-L6-v2` (semantic embeddings)

These will be stored in the `models/` directory inside your project.

## Notes

- Captioning is **only in English** (for now).
- App works fully offline after first model download.
- Works best in Chrome or Edge (folder upload by drag & drop).

---

**Questions or ideas?**  
Open an issue or ping the author!

---

(c) 2024 Rooster AI Tools
