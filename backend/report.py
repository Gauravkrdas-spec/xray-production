import requests
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def generate_report(findings, image_base64=None):
    if not findings:
        diseases_text = "No significant findings detected."
    else:
        diseases_text = "\n".join([
            f"- {f['name']}: {f['probability']}% probability ({f['severity']} severity)"
            for f in findings
        ])

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
"""

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    body = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        text = data["content"][0]["text"]
        clean = text.replace("```json", "").replace("```", "").strip()
        import json
        return json.loads(clean)

    except Exception as e:
        return {
            "findings": f"Report generation error: {str(e)}",
            "normal_structures": "Unable to assess",
            "impression": "Technical error occurred during report generation",
            "recommendations": ["Please retry analysis", "Contact support if issue persists"],
            "report_quality": "poor",
            "urgency": "routine"
        }