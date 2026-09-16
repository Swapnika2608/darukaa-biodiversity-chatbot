"""
reasoning_engine.py
Parses structured environmental inputs and produces a multi-metric context
string that is injected into the LLM prompt alongside RAG chunks.
"""

from typing import Dict, Any, List, Tuple

# ---------------------------------------------------------------------------
# Thresholds derived from scientific literature
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "soil_organic_carbon_pct": [
        (0.5,  "critical",  "SOC < 0.5%: severely degraded – microbial collapse likely"),
        (1.0,  "poor",      "SOC 0.5–1%: low – limited invertebrate and plant support"),
        (2.0,  "moderate",  "SOC 1–2%: moderate – functional but improvable"),
        (float("inf"), "good", "SOC > 2%: healthy – supports diverse soil food web"),
    ],
    "soil_ph": [
        (5.5,  "acidic",   "pH < 5.5: acidic – mycorrhizal suppression, Al/Mn toxicity"),
        (7.0,  "optimal",  "pH 5.5–7.0: optimal range for most plant and soil communities"),
        (float("inf"), "alkaline", "pH > 7.0: alkaline – P lock-up, micronutrient deficiency"),
    ],
    "rainfall_mm_year": [
        (250,  "hyper_arid", "< 250 mm/yr: hyper-arid – extreme water stress"),
        (400,  "arid",       "250–400 mm/yr: arid – water harvesting critical"),
        (700,  "semi_arid",  "400–700 mm/yr: semi-arid – drought-tolerant species needed"),
        (float("inf"), "adequate", "> 700 mm/yr: adequate rainfall"),
    ],
    "temperature_c": [
        (15,   "cool",    "< 15 °C mean: cool – slow decomposition, peat risk"),
        (25,   "optimal", "15–25 °C: optimal for most temperate biodiversity"),
        (float("inf"), "hot", "> 25 °C mean: heat stress – phenological mismatch risk"),
    ],
}

LAND_USE_RISK = {
    "monoculture":       ("high",   "70–90% plant species loss vs native baseline"),
    "intensive_grazing": ("high",   "soil compaction, grass monoculture, invertebrate loss"),
    "agroforestry":      ("low",    "2–5× species richness vs open cropland"),
    "intercropping":     ("low",    "30–50% arthropod diversity gain"),
    "native_grassland":  ("low",    "high baseline biodiversity"),
    "degraded_forest":   ("medium", "edge effects, invasive species risk"),
    "urban":             ("high",   "impervious surfaces, light/noise pollution"),
}


def classify(variable: str, value: float) -> Tuple[str, str]:
    """Return (status_label, description) for a numeric variable."""
    for threshold, label, desc in THRESHOLDS.get(variable, []):
        if value <= threshold:
            return label, desc
    return "unknown", "no threshold data"


def analyse_inputs(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts a dict of environmental metrics and returns a structured
    analysis with per-variable status and cross-variable interactions.
    """
    analysis: Dict[str, Any] = {"variables": {}, "interactions": [], "missing": []}

    # Numeric variables
    numeric_map = {
        "soil_organic_carbon_pct": data.get("soil_organic_carbon_pct"),
        "soil_ph":                 data.get("soil_ph"),
        "rainfall_mm_year":        data.get("rainfall_mm_year"),
        "temperature_c":           data.get("temperature_c"),
    }
    for var, val in numeric_map.items():
        if val is None:
            analysis["missing"].append(var)
        else:
            label, desc = classify(var, float(val))
            analysis["variables"][var] = {"value": val, "status": label, "description": desc}

    # Land use
    land_use = data.get("land_use", "").lower().replace(" ", "_")
    if land_use:
        risk, note = LAND_USE_RISK.get(land_use, ("unknown", "no data for this land use type"))
        analysis["variables"]["land_use"] = {"value": land_use, "status": risk, "description": note}
    else:
        analysis["missing"].append("land_use")

    # Cross-variable interaction flags
    soc = data.get("soil_organic_carbon_pct")
    rain = data.get("rainfall_mm_year")
    temp = data.get("temperature_c")

    if soc is not None and soc < 1.0 and land_use in ("monoculture", "intensive_grazing"):
        analysis["interactions"].append(
            "LOW SOC + HIGH-RISK LAND USE: compounding degradation – "
            "soil food web collapse accelerates under continued monoculture."
        )
    if rain is not None and rain < 400 and soc is not None and soc < 1.0:
        analysis["interactions"].append(
            "LOW RAINFALL + LOW SOC: water retention critically impaired – "
            "soil carbon acts as a sponge; restoring SOC is prerequisite for any vegetation recovery."
        )
    if temp is not None and temp > 25 and rain is not None and rain < 400:
        analysis["interactions"].append(
            "HIGH TEMPERATURE + LOW RAINFALL: heat-drought compound stress – "
            "species range shifts accelerated; connectivity corridors become urgent."
        )
    if land_use in ("monoculture", "intensive_grazing") and rain is not None and rain < 700:
        analysis["interactions"].append(
            "MONOCULTURE/GRAZING + DRY CLIMATE: runoff and erosion risk elevated – "
            "bare soil between crop rows loses 10–100× more topsoil than covered land."
        )

    return analysis


def format_analysis_for_prompt(analysis: Dict[str, Any]) -> str:
    """Render the analysis dict as a readable string for LLM injection."""
    lines: List[str] = ["=== ENVIRONMENTAL METRICS ANALYSIS ==="]
    for var, info in analysis["variables"].items():
        lines.append(f"• {var}: {info['value']} → [{info['status'].upper()}] {info['description']}")
    if analysis["interactions"]:
        lines.append("\n=== CROSS-VARIABLE INTERACTIONS DETECTED ===")
        for ix in analysis["interactions"]:
            lines.append(f"⚠ {ix}")
    if analysis["missing"]:
        lines.append(f"\n⚠ Missing variables: {', '.join(analysis['missing'])}")
    return "\n".join(lines)
