import os
import uuid
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from dotenv import load_dotenv
from xray_model import load_model, analyze_xray
from report import generate_report
from database import (
    init_db, save_analysis, save_feedback,
    get_stats, get_recent_analyses, get_recent_feedback
)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize database
init_db()

# Configuration
UPLOAD_FOLDER   = os.getenv("UPLOAD_FOLDER", "uploads")
LOG_FOLDER      = os.getenv("LOG_FOLDER", "logs")
MAX_SIZE_MB     = int(os.getenv("MAX_IMAGE_SIZE_MB", 10))
MAX_SIZE_BYTES  = MAX_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Setup logging
logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load AI model at startup
print("Starting RadVision AI Server...")
model, device = load_model()
print("Server ready!")


# ── Helper functions ──────────────────────────────────
def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def delete_file(filepath):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        logger.error(f"Could not delete file {filepath}: {e}")


# ── Routes ────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return send_from_directory('.', 'website.html')

@app.route("/admin", methods=["GET"])
def admin():
    return send_from_directory('.', 'admin.html')


@app.route("/analyze", methods=["POST"])
def analyze():
    filepath = None
    try:
        # Check image in request
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]

        # Check filename
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Check extension
        if not allowed_file(file.filename):
            return jsonify({
                "error": "Invalid file type. Use JPG or PNG"
            }), 400

        # Check file size
        file.seek(0, 2)
        size = file.tell()
        file.seek(0)
        if size > MAX_SIZE_BYTES:
            return jsonify({
                "error": f"File too large. Maximum {MAX_SIZE_MB}MB"
            }), 400

        # Save file with unique name
        ext         = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4()}.{ext}"
        filepath    = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(filepath)
        logger.info(f"Image received: {unique_name} ({size} bytes)")

        # Get patient info from form
        patient_name   = request.form.get('patient_name',   '')
        patient_age    = request.form.get('patient_age',    '')
        patient_gender = request.form.get('patient_gender', '')
        referred_by    = request.form.get('referred_by',    '')

        # Step 1 - Analyze with CheXNet
        print(f"Analyzing image: {unique_name}")
        findings = analyze_xray(filepath, model, device)
        logger.info(f"CheXNet findings: {findings}")

        # Step 2 - Generate report with Claude
        print("Generating medical report...")
        report = generate_report(findings)
        logger.info("Report generated successfully")

        # Step 3 - Save to database
        save_analysis(
            image_id       = unique_name,
            patient_name   = patient_name,
            patient_age    = patient_age,
            patient_gender = patient_gender,
            referred_by    = referred_by,
            findings       = findings,
            report         = report
        )
        logger.info(f"Analysis saved to database: {unique_name}")

        # Step 4 - Return results
        return jsonify({
            "success":   True,
            "findings":  findings,
            "report":    report,
            "timestamp": datetime.now().isoformat(),
            "image_id":  unique_name
        })

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({
            "error": f"Analysis failed: {str(e)}"
        }), 500

    finally:
        # Always delete uploaded image for privacy
        if filepath:
            delete_file(filepath)
            logger.info(f"Image deleted: {filepath}")


@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json()
        save_feedback(
            analysis_id = data.get('analysis_id', ''),
            doctor_name = data.get('doctorName',  ''),
            hospital    = data.get('hospital',    ''),
            rating      = data.get('rating',      ''),
            comments    = data.get('comments',    ''),
            recommend   = data.get('recommend',   '')
        )
        logger.info(f"Feedback saved from Dr. {data.get('doctorName')}")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def stats():
    try:
        return jsonify(get_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def history():
    try:
        return jsonify({
            "analyses": get_recent_analyses(10),
            "feedback": get_recent_feedback(10)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Start Server ──────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    print(f"RadVision AI running on http://localhost:{port}")
    app.run(
        host  = "0.0.0.0",
        port  = port,
        debug = False
    )