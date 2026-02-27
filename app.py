"""
SpectraLens - IR Spectrum Bulk Analyzer
Supports 2 to 400+ image comparisons
Uses Groq FREE API
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
import base64, csv, os, json, re, requests, threading, uuid
from datetime import datetime
from itertools import combinations

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULTS_CSV = "results/analysis_history.csv"
JOBS = {}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("results", exist_ok=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def image_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

def get_media_type(filename):
    ext = filename.lower().split(".")[-1]
    return {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","webp":"image/webp"}.get(ext,"image/jpeg")

def analyze_pair(img1_path, img2_path, img1_name, img2_name):
    img1_b64 = image_to_base64(img1_path)
    img2_b64 = image_to_base64(img2_path)
    mt1 = get_media_type(img1_name)
    mt2 = get_media_type(img2_name)

    prompt = """You are a world-class expert analytical chemist with deep knowledge of IR (Infrared) Spectroscopy.

Your job is to SCIENTIFICALLY ANALYZE the curve shape of each IR spectrum image and IDENTIFY the chemical compound from the curve pattern alone — like a real chemist would do in a laboratory.

HOW TO IDENTIFY THE COMPOUND FROM THE CURVE:
1. Look at WHERE the troughs (dips downward) appear on the wavenumber axis (4000 to 500 cm-1)
2. Look at HOW DEEP the troughs are (transmittance percentage)
3. Identify functional groups from peak positions:
   - Broad trough at 3200-3500 = O-H stretch (alcohols, carboxylic acids)
   - Sharp peak at 2850-2960 = C-H stretch (alkanes)
   - Strong trough at 1700-1750 = C=O stretch (ketones, aldehydes, esters)
   - Trough at 1600-1650 = C=C stretch (alkenes) or N-H bend
   - Strong broad trough at 1000-1300 = C-O stretch (ethers, alcohols, esters)
   - Sharp peaks at 1400-1500 = C-H bending
   - Trough at 2200-2260 = C≡N or C≡C (nitriles, alkynes)
   - Broad trough at 2500-3300 = O-H of carboxylic acid
4. Use the fingerprint region (500-1500 cm-1) pattern to narrow down exact compound
5. Based on ALL functional groups detected, identify the most likely compound and its formula

IMPORTANT: Even if no text is visible in the image, you MUST identify the compound from the curve shape. Use your chemistry knowledge to give the best scientific identification.

Return ONLY valid JSON, no markdown, no extra text:

{
  "image1": {
    "compound_name": "Your best scientific identification of the compound based on curve analysis e.g. Ethanol, Acetone, Glucose, Aspirin etc",
    "chemical_formula": "The chemical formula e.g. C2H5OH, CH3COCH3, C6H12O6 etc",
    "identification_confidence": "High / Medium / Low",
    "identification_reasoning": "Explain step by step how you identified the compound from the curve shape - which peaks led you to which functional groups and finally to the compound name",
    "possible_alternatives": "Other compounds it could possibly be if not 100% sure",
    "sample_type": "KBR DISC or LIQUID FILM or NUJOL MULL etc if visible, else Unknown",
    "major_peaks": [
      {"wavenumber": 3408, "transmittance": 5, "assignment": "O-H stretch — indicates hydroxyl group"}
    ],
    "curve_description": "Describe the full curve: where it starts, where big troughs are, flat regions, dense peak clusters, overall shape",
    "key_regions": {
      "4000_2500": "Detailed description of what peaks appear here and what they mean",
      "2500_1500": "Detailed description of this region",
      "1500_500": "Fingerprint region — describe the pattern in detail"
    }
  },
  "image2": {
    "compound_name": "Scientific identification from curve analysis",
    "chemical_formula": "Chemical formula",
    "identification_confidence": "High / Medium / Low",
    "identification_reasoning": "Step by step reasoning from curve to compound",
    "possible_alternatives": "Other possible compounds",
    "sample_type": "sample type if visible",
    "major_peaks": [
      {"wavenumber": 3342, "transmittance": 81, "assignment": "functional group assignment"}
    ],
    "curve_description": "Full curve description",
    "key_regions": {
      "4000_2500": "description",
      "2500_1500": "description",
      "1500_500": "fingerprint region description"
    }
  },
  "comparison": {
    "similarity_score": 15,
    "is_same_compound": false,
    "accuracy_100_percent": false,
    "accuracy_explanation": "Detailed explanation of why these are or are not the same compound",
    "matching_peaks": [
      {"wavenumber_img1": 3408, "wavenumber_img2": 3342, "difference": 66, "note": "Both show O-H stretch indicating hydroxyl group in both compounds"}
    ],
    "non_matching_peaks": [
      {"wavenumber": 1710, "present_in": "image1 only", "note": "C=O carbonyl stretch — compound 1 has ketone/ester, compound 2 does not"}
    ],
    "trough_crest_comparison": "Detailed scientific comparison of where troughs and crests occur — which wavenumber positions match and which do not",
    "curve_up_down_analysis": {
      "image1_pattern": "Detailed up-down pattern: e.g. high at 4000, sharp deep trough at 3400 (O-H), rises to 2800, moderate trough at 2900 (C-H), flat at 2300-2000, multiple sharp troughs at 1700 (C=O) and 1000-1500 (fingerprint)",
      "image2_pattern": "Detailed up-down pattern for image 2",
      "how_similar": "Scientific assessment of how similar the two curve patterns are"
    },
    "similarities": [
      "Both show broad absorption in 3000-3500 region indicating O-H or N-H groups",
      "Both have C-H stretching peaks around 2900 cm-1"
    ],
    "differences": [
      "Compound 1 has strong C=O peak at 1710 cm-1 absent in Compound 2",
      "Compound 2 has complex fingerprint region suggesting aromatic structure"
    ],
    "functional_groups_comparison": "Compare all functional groups identified in each compound and explain what chemical differences this reveals",
    "conclusion": "Final scientific conclusion: what are these two compounds, how similar or different are they chemically, and what does this comparison tell a chemist about their structures and properties"
  }
}"""

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "max_tokens": 3000,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mt1};base64,{img1_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:{mt2};base64,{img2_b64}"}},
            {"type": "text", "text": prompt}
        ]}]
    }

    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=90)
    if resp.status_code != 200:
        raise Exception(f"Groq error {resp.status_code}: {resp.text[:300]}")

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    return json.loads(raw)


def append_csv(img1_name, img2_name, result):
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp","image1","image2","compound1","formula1",
                        "compound2","formula2","similarity_score","is_same_compound",
                        "accuracy_100_percent","matching_peaks","conclusion"])
        comp = result.get("comparison", {})
        i1 = result.get("image1", {})
        i2 = result.get("image2", {})
        w.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            img1_name, img2_name,
            i1.get("compound_name","?"), i1.get("chemical_formula","?"),
            i2.get("compound_name","?"), i2.get("chemical_formula","?"),
            comp.get("similarity_score", 0),
            comp.get("is_same_compound", False),
            comp.get("accuracy_100_percent", False),
            len(comp.get("matching_peaks", [])),
            comp.get("conclusion","")[:250]
        ])


def run_batch_job(job_id, file_pairs):
    JOBS[job_id]["status"] = "running"
    results = []
    total = len(file_pairs)

    for i, (p1, p2, n1, n2) in enumerate(file_pairs):
        try:
            result = analyze_pair(p1, p2, n1, n2)
            append_csv(n1, n2, result)
            results.append({"pair_index": i, "image1_name": n1, "image2_name": n2, "status": "done", "result": result})
        except Exception as e:
            results.append({"pair_index": i, "image1_name": n1, "image2_name": n2, "status": "error", "error": str(e)})

        JOBS[job_id]["progress"] = i + 1
        JOBS[job_id]["results"] = results

    JOBS[job_id]["status"] = "complete"


# ── ROUTES ──────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/robots.txt")
def robots():
    return send_from_directory(".", "robots.txt")

@app.route("/sitemap.xml")
def sitemap():
    return send_from_directory(".", "sitemap.xml", mimetype="application/xml")

@app.route("/submit", methods=["POST"])
def submit():
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY not set."}), 400

    files = request.files.getlist("images")
    mode = request.form.get("mode", "all_pairs")

    if len(files) < 2:
        return jsonify({"error": "Upload at least 2 images."}), 400
    if len(files) > 400:
        return jsonify({"error": "Maximum 400 images at once."}), 400

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []
    for f in files:
        if f.filename:
            safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", f.filename)
            path = os.path.join(UPLOAD_FOLDER, f"{ts}_{safe_name}")
            f.save(path)
            saved.append((path, f.filename))

    if mode == "all_pairs":
        pairs = [(saved[a][0], saved[b][0], saved[a][1], saved[b][1]) for a, b in combinations(range(len(saved)), 2)]
    elif mode == "sequential":
        pairs = [(saved[i][0], saved[i+1][0], saved[i][1], saved[i+1][1]) for i in range(len(saved)-1)]
    elif mode == "vs_first":
        pairs = [(saved[0][0], saved[i][0], saved[0][1], saved[i][1]) for i in range(1, len(saved))]
    else:
        pairs = [(saved[0][0], saved[1][0], saved[0][1], saved[1][1])]

    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {"status": "queued", "progress": 0, "total": len(pairs), "results": [], "mode": mode, "num_images": len(saved)}

    threading.Thread(target=run_batch_job, args=(job_id, pairs), daemon=True).start()
    return jsonify({"job_id": job_id, "total_pairs": len(pairs), "num_images": len(saved)})


@app.route("/job/<job_id>")
def job_status(job_id):
    if job_id not in JOBS:
        return jsonify({"error": "Job not found"}), 404
    job = JOBS[job_id]
    return jsonify({"status": job["status"], "progress": job["progress"], "total": job["total"], "results": job["results"], "mode": job.get("mode"), "num_images": job.get("num_images")})


@app.route("/history")
def history():
    if not os.path.exists(RESULTS_CSV):
        return jsonify([])
    rows = []
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return jsonify(rows[-200:])


@app.route("/export-csv")
def export_csv():
    if os.path.exists(RESULTS_CSV):
        return send_from_directory("results", "analysis_history.csv", as_attachment=True)
    return jsonify({"error": "No data yet"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
