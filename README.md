# 🌿 Darukaa.Earth — AI Biodiversity Intelligence Chatbot

An AI-powered environmental scientist chatbot that gives scientifically grounded, evidence-backed biodiversity recommendations by reasoning across multiple environmental variables simultaneously.

---

## Architecture

```
User Input (Text or JSON)
        │
        ▼
chatbot.py ──── JSON parser ──► _pending_data (accumulated metrics)
        │
        ├──► reasoning_engine.py
        │         • Classifies each metric against scientific thresholds
        │         • Detects cross-variable interactions (SOC × rainfall, temp × aridity)
        │         • Flags missing variables → triggers clarifying questions
        │
        ├──► knowledge_base.py  (ChromaDB + sentence-transformers)
        │         • 16 citable knowledge chunks (FAO, IPCC, Nature, Science, etc.)
        │         • Semantic retrieval: top-4 chunks relevant to query
        │
        └──► LLM (Google Gemini via LangChain)
                  • System prompt enforces: WHAT/WHY/METRIC/SOURCE/TIME HORIZON
                  • Injected: RAG chunks + metrics analysis + conversation history
                  • ChatMessageHistory (last 20 messages = 10 turns)
```

---

## File Structure

```
Biodiversity_Challenge/
├── knowledge_base.py      # ChromaDB vector store + 16 scientific knowledge chunks
├── reasoning_engine.py    # Multi-metric threshold classification + interaction detection
├── chatbot.py             # Conversational agent with memory, RAG, and reasoning
├── app.py                 # Streamlit web UI
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
└── .streamlit/
    └── config.toml        # Streamlit configuration
```

---

## Knowledge System

**Vector Database:** ChromaDB (in-memory)  
**Embedding Model:** sentence-transformers `all-MiniLM-L6-v2`  
**Knowledge Chunks:** 16 scientific findings covering:

| Category | Chunks | Sources |
|---|---|---|
| Soil Health | soil_001, soil_002, soil_003 | FAO 2017, Rillig et al. 2019, Bardgett & van der Putten 2014 |
| Land Use | land_001, land_002, land_003 | Tscharntke et al. 2012, Haddad et al. 2015, IPCC AR6 2022 |
| Climate | climate_001, climate_002, climate_003 | Reij et al. 2009, Memmott et al. 2007, Mayer et al. 2007 |
| Biodiversity | bio_001, bio_002 | MacArthur & Wilson 1967, CBD GBF 2022, Tilman et al. 2014 |
| Human Impact | human_001, human_002, human_003 | Woodcock et al. 2017, Hansen et al. 2013, Gaston et al. 2013 |
| Multi-variable | multi_001, multi_002, multi_003 | Bardgett 2014, Whitford & Duval 2009, Urban 2015 |

**Retrieval:** Semantic similarity search — user query + detected interactions → top-4 chunks injected into LLM prompt.

---

## Reasoning Engine

Rule-based multi-metric analysis in `reasoning_engine.py` runs **before** the LLM:

**Scientific Thresholds:**
- SOC < 0.5% → CRITICAL, 0.5–1% → POOR, 1–2% → MODERATE, >2% → GOOD
- Rainfall < 250mm → HYPER-ARID, 250–400mm → ARID, 400–700mm → SEMI-ARID
- pH < 5.5 → ACIDIC, 5.5–7.0 → OPTIMAL, >7.0 → ALKALINE
- Temperature > 25°C → HOT (heat stress risk)

**Cross-Variable Interactions Detected:**
1. LOW SOC + MONOCULTURE → soil food web collapse warning
2. LOW RAINFALL + LOW SOC → water retention critically impaired
3. HIGH TEMP + LOW RAINFALL → heat-drought compound stress
4. MONOCULTURE + DRY CLIMATE → erosion risk elevated

---

## Local Setup

### Prerequisites
- Python 3.10+
- Google Gemini API key (free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/darukaa-biodiversity-chatbot.git
cd darukaa-biodiversity-chatbot

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
copy .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Run Web UI
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

### Run CLI
```bash
python main.py
```

---

## Environment Variables

```
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

---

## Example Usage

**Text input:**
```
Biodiversity is declining on my land
```
→ System asks for soil, rainfall, land use data

**JSON input:**
```json
{"soil_organic_carbon_pct": 0.3, "rainfall_mm_year": 280, "land_use": "monoculture", "temperature_c": 28}
```
→ System gives 3 structured recommendations with citations, metrics, time horizons

**Sample Output:**
```
RECOMMENDATION [1]: Legume Cover Cropping + Mulching
Action      : Sow vetch/clover between crop rows immediately post-harvest
Mechanism   : N-fixation adds root biomass → SOC rises 15–25% over 2–3 years
Metrics     : SOC ↑, invertebrate richness ↑ 2–3×, VWC ↑ 5–10%
Time Horizon: Medium (2–3 years)
Source      : FAO, 2017 – Soil Organic Carbon: the hidden potential
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini (via LangChain) |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Web UI | Streamlit |
| Memory | LangChain ChatMessageHistory |
| Language | Python 3.12 |

---

## CI/CD

No CI/CD pipeline configured. Run locally using the setup instructions above.

---

## Evaluation Criteria Coverage

| Criteria | Implementation |
|---|---|
| Depth of Reasoning (30%) | 4 cross-variable interaction detectors in Python code |
| Scientific Grounding (25%) | 16 chunks with real citations from FAO, IPCC, Nature, Science |
| Knowledge System Design (20%) | ChromaDB + sentence-transformers semantic retrieval pipeline |
| Conversational Intelligence (15%) | Memory + clarifying questions + context adaptation |
| Output Clarity (10%) | Structured format enforced by system prompt on every response |
