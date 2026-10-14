import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME = "claude-sonnet-4-5"


def describe_location(x, y):
    """Plain-language region for a normalised (x, y) on a frontal radiograph.

    On a PA/AP film the image's left side is the patient's right.
    """
    side = "right" if x < 0.45 else ("left" if x > 0.55 else "central/mediastinal")
    zone = "upper" if y < 0.40 else ("middle" if y < 0.65 else "lower")
    return f"patient's {side} {zone} zone (x={x:.2f}, y={y:.2f})"


def generate_report(findings, image_base64=None, include_location=False,
                    temperature=None):
    """Generate a structured report from the classifier findings.

    include_location=True adds each top-5 finding's marker position (GradCAM or
    anatomical prior, whichever the finding carries) to the prompt. This is what
    makes the GradCAM ablation meaningful: without it both arms send identical text.
    temperature: pass 0 for reproducible evaluation runs; None keeps the API default.

    Returns the parsed JSON report. On failure the dict carries "error": True so
    evaluation code can exclude it instead of scoring the error message.
    """
    if not findings:
        diseases_text = "No significant findings detected."
    else:
        lines = []
        for rank, f in enumerate(findings):
            line = f"- {f['name']}: {f['probability']}% probability ({f['severity']} severity)"
            if include_location and rank < 5 and 'x' in f and 'y' in f:
                line += f"; marker location: {describe_location(f['x'], f['y'])}"
            lines.append(line)
        diseases_text = "\n".join(lines)

    location_rule = ("- Where a marker location is given, use it to state the side and zone of the finding\n"
                     if include_location else "")

    prompt = f"""You are a senior radiologist. Analyze these chest X-ray AI findings and write a CONCISE clinical report.

AI Findings:
{diseases_text}

Respond in JSON only, no markdown:
{{
    "findings": "2-3 sentences maximum covering key abnormalities only",
    "normal_structures": "1 sentence only",
    "impression": "1-2 sentences maximum",
    "recommendations": ["rec 1", "rec 2", "rec 3"],
    "report_quality": "good|fair|poor",
    "urgency": "routine|urgent|emergency"
}}

Rules:
- findings: maximum 3 sentences
- normal_structures: maximum 1 sentence
- impression: maximum 2 sentences
- recommendations: maximum 5 items, each under 10 words
- Use medical terminology but be brief and precise
- No repetition between sections
{location_rule}"""

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    body = {
        "model": MODEL_NAME,
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    if temperature is not None:
        body["temperature"] = temperature

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        text = data["content"][0]["text"]
        clean = text.replace("```json", "").replace("```", "").strip()
        report = json.loads(clean)
        report["error"] = False
        return report

    except Exception as e:
        return {
            "findings": f"Report generation error: {str(e)}",
            "normal_structures": "Unable to assess",
            "impression": "Technical error occurred during report generation",
            "recommendations": ["Please retry analysis", "Contact support if issue persists"],
            "report_quality": "poor",
            "urgency": "routine",
            "error": True
        }
