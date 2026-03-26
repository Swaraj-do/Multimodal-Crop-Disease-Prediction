import numpy as np

ENV_COLS = ["temperature", "humidity", "rainfall", "wind_speed", "season_code"]

def env_influence_score(env: dict) -> float:
    """
    Returns a score between 0 and 1 indicating how favorable
    the environmental conditions are for disease spread.
    """
    temp = env["temperature"]
    hum = env["humidity"]
    rain = env["rainfall"]
    wind = env["wind_speed"]

    temp_n = np.clip((temp - 20) / 15, 0, 1)
    hum_n = np.clip((hum - 40) / 55, 0, 1)
    rain_n = np.clip(rain / 20, 0, 1)
    wind_n = np.clip(wind / 6, 0, 1)

    score = (
        0.45 * hum_n +
        0.30 * rain_n +
        0.20 * temp_n +
        0.05 * wind_n
    )

    return float(np.clip(score, 0, 1))


def risk_level(disease_prob: float, env_score: float) -> str:
    """
    Combines disease probability and environment score.
    """
    combined = 0.7 * disease_prob + 0.3 * env_score

    if combined >= 0.75:
        return "HIGH"
    elif combined >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"


def disease_family_from_name(class_name: str) -> str:
    name = class_name.lower()

    if "healthy" in name:
        return "healthy"
    if "bacterial" in name:
        return "bacterial"
    if "mosaic" in name or "virus" in name or "yellow_leaf_curl" in name:
        return "viral"

    fungal_keywords = [
        "blight", "rust", "mildew", "mold", "scab",
        "leaf_spot", "septoria", "powdery", "spot"
    ]
    if any(k in name for k in fungal_keywords):
        return "fungal"

    return "unknown"


def recommendations(class_name: str, risk: str, env: dict) -> list[str]:
    family = disease_family_from_name(class_name)
    hum = env["humidity"]
    rain = env["rainfall"]

    recs = []

    if risk == "HIGH":
        recs.append("Take action quickly to reduce further spread.")
    elif risk == "MEDIUM":
        recs.append("Monitor the crop daily and apply preventive measures.")
    else:
        recs.append("Continue monitoring and maintain preventive care.")

    if family == "healthy":
        recs.extend([
            "Maintain balanced irrigation and nutrition.",
            "Inspect leaves regularly for early symptoms.",
            "Keep field hygiene and remove weeds."
        ])
        return recs

    recs.extend([
        "Remove severely infected leaves or affected parts.",
        "Do not leave infected plant debris in the field.",
        "Sanitize tools after handling infected plants."
    ])

    if family == "fungal":
        recs.extend([
            "Avoid overhead irrigation and reduce leaf wetness.",
            "Improve airflow by proper spacing or pruning.",
            "Ensure proper drainage to avoid excess moisture."
        ])
        if hum > 80:
            recs.append("Humidity is high, so ventilation and morning watering are important.")
        if rain > 10:
            recs.append("Rainfall is high, so improve drainage and reduce water stagnation.")
        recs.append("Consult local agricultural guidelines for suitable fungicide options.")

    elif family == "bacterial":
        recs.extend([
            "Avoid working in wet fields to reduce bacterial spread.",
            "Reduce splashing water on leaves.",
            "Use clean seeds and disease-free planting material."
        ])
        recs.append("Consult local agricultural guidelines for suitable bacterial control methods.")

    elif family == "viral":
        recs.extend([
            "Control insect vectors such as aphids and whiteflies.",
            "Remove heavily infected plants early.",
            "Use disease-free seeds or seedlings in future planting."
        ])

    else:
        recs.extend([
            "Consult a local agricultural expert for confirmation.",
            "Capture more images under natural light for better diagnosis."
        ])

    return recs