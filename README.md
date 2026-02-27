# SpectraLens — IR Spectrum Analyzer

Free AI-powered IR spectrum analysis tool. Upload 2 to 400 infrared spectrum images, identify chemical compounds, compare peaks, get similarity scores.

## Live Demo
https://YOUR-RAILWAY-DOMAIN.up.railway.app

## Features
- Upload 2 to 400 IR spectrum images at once
- AI identifies compound name and chemical formula
- Compares troughs, crests, and peak patterns
- Similarity score 0–100%
- 100% accuracy check
- CSV export of all results
- Analysis history

## Tech Stack
- Python / Flask
- Groq AI (Llama Vision) — Free API
- Vanilla HTML/CSS/JS

## Run Locally
```
pip install flask requests werkzeug gunicorn
set GROQ_API_KEY=your_groq_key_here
python app.py
```
Open: http://localhost:5000

## Topics
ir-spectroscopy chemistry spectrum-analysis infrared python flask ai groq
