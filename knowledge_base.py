"""
knowledge_base.py
Builds and queries a ChromaDB vector store from structured biodiversity knowledge chunks.
Each chunk is a scientific finding linking environmental variables to biodiversity outcomes.
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict

# ---------------------------------------------------------------------------
# Raw knowledge corpus – each entry is a citable scientific finding
# ---------------------------------------------------------------------------
KNOWLEDGE_CHUNKS: List[Dict] = [
    # ── Soil Health ──────────────────────────────────────────────────────────
    {
        "id": "soil_001",
        "text": (
            "Soil organic carbon (SOC) below 1% severely limits microbial biomass and "
            "enzymatic activity, reducing decomposer diversity by up to 40%. "
            "Legume cover crops (e.g., clover, vetch) raise SOC by 15–25% over 2–3 years "
            "by fixing atmospheric nitrogen and adding root biomass. "
            "(FAO, 2017 – Soil Organic Carbon: the hidden potential)"
        ),
        "tags": ["soil", "SOC", "cover_crops", "microbial_diversity", "nitrogen_fixation"],
    },
    {
        "id": "soil_002",
        "text": (
            "Soil pH outside the 5.5–7.0 range reduces plant-available phosphorus and "
            "suppresses mycorrhizal fungal networks, which support 80% of terrestrial plant "
            "species. Lime application on acidic soils (pH < 5.5) restores mycorrhizal "
            "colonisation within 1–2 growing seasons. "
            "(Rillig et al., 2019 – Nature Reviews Microbiology)"
        ),
        "tags": ["soil", "pH", "mycorrhizae", "plant_diversity", "phosphorus"],
    },
    {
        "id": "soil_003",
        "text": (
            "Soil moisture deficit below 20% volumetric water content (VWC) collapses "
            "invertebrate communities (earthworms, beetles) that drive nutrient cycling. "
            "Mulching with crop residues maintains VWC 5–10% higher than bare soil, "
            "supporting 2–3× greater invertebrate species richness. "
            "(Bardgett & van der Putten, 2014 – Nature)"
        ),
        "tags": ["soil", "moisture", "invertebrates", "mulching", "nutrient_cycling"],
    },
    # ── Land Use / Land Cover ────────────────────────────────────────────────
    {
        "id": "land_001",
        "text": (
            "Monoculture agriculture reduces plant species richness by 70–90% compared to "
            "native grasslands and eliminates structural habitat heterogeneity needed by "
            "pollinators and ground-nesting birds. Intercropping with 3+ species increases "
            "arthropod diversity by 30–50% and reduces pest pressure by 20–30%. "
            "(Tscharntke et al., 2012 – Trends in Ecology & Evolution)"
        ),
        "tags": ["land_use", "monoculture", "intercropping", "pollinators", "arthropods"],
    },
    {
        "id": "land_002",
        "text": (
            "Habitat fragmentation below 10 ha patch size creates edge effects that reduce "
            "interior forest species (amphibians, forest birds) by 50–80%. "
            "Establishing wildlife corridors ≥ 50 m wide between fragments restores "
            "gene flow and increases species richness by 20–40% within 5–10 years. "
            "(Haddad et al., 2015 – Science Advances)"
        ),
        "tags": ["land_use", "fragmentation", "corridors", "edge_effects", "connectivity"],
    },
    {
        "id": "land_003",
        "text": (
            "Agroforestry systems integrating trees with crops increase above-ground carbon "
            "stocks by 20–50 Mg C/ha and support 2–5× more bird and insect species than "
            "open cropland by providing vertical structural complexity. "
            "(IPCC AR6 WGIII, 2022 – Chapter 7: Agriculture, Forestry and Land Use)"
        ),
        "tags": ["land_use", "agroforestry", "carbon_stocks", "birds", "insects", "IPCC"],
    },
    # ── Climate Factors ──────────────────────────────────────────────────────
    {
        "id": "climate_001",
        "text": (
            "In semi-arid regions (annual rainfall < 400 mm), water harvesting techniques "
            "(contour bunds, half-moon catchments) increase effective soil moisture by "
            "30–60%, enabling establishment of drought-tolerant native shrubs that serve "
            "as keystone habitat for reptiles and small mammals. "
            "(Reij et al., 2009 – World Resources Institute)"
        ),
        "tags": ["climate", "rainfall", "semi_arid", "water_harvesting", "native_shrubs"],
    },
    {
        "id": "climate_002",
        "text": (
            "Temperature increases of 1–2 °C above historical mean shift plant phenology "
            "by 5–10 days, creating phenological mismatches between flowering plants and "
            "specialist pollinators, reducing pollination success by 15–30%. "
            "Planting phenologically diverse native species buffers this mismatch. "
            "(Memmott et al., 2007 – Proceedings of the Royal Society B)"
        ),
        "tags": ["climate", "temperature", "phenology", "pollinators", "native_plants"],
    },
    {
        "id": "climate_003",
        "text": (
            "Riparian buffer strips (10–30 m wide) along water bodies reduce agricultural "
            "runoff nitrogen by 50–80%, maintaining aquatic macroinvertebrate diversity "
            "and supporting amphibian breeding habitat. They also moderate local "
            "microclimate by 1–3 °C. "
            "(Mayer et al., 2007 – Journal of Environmental Quality)"
        ),
        "tags": ["climate", "water", "riparian", "nitrogen_runoff", "amphibians", "microclimate"],
    },
    # ── Biodiversity Indicators ──────────────────────────────────────────────
    {
        "id": "bio_001",
        "text": (
            "Species richness follows a species-area relationship (SAR): doubling habitat "
            "area increases species richness by ~20% (z ≈ 0.25–0.35). Below minimum viable "
            "habitat thresholds, local extinction debt accumulates even without further "
            "habitat loss. Habitat restoration of ≥ 30% of degraded land is the CBD "
            "Kunming-Montreal Global Biodiversity Framework target. "
            "(MacArthur & Wilson, 1967; CBD GBF Target 2, 2022)"
        ),
        "tags": ["biodiversity", "species_richness", "habitat_area", "SAR", "restoration"],
    },
    {
        "id": "bio_002",
        "text": (
            "Functional diversity (trait diversity) predicts ecosystem resilience better "
            "than species richness alone. Systems with high functional redundancy maintain "
            "nutrient cycling and primary productivity under disturbance. "
            "Introducing native plant species from ≥ 3 functional groups (grasses, forbs, "
            "shrubs) increases functional diversity index by 40–60%. "
            "(Tilman et al., 2014 – PNAS)"
        ),
        "tags": ["biodiversity", "functional_diversity", "resilience", "native_plants", "ecosystem_services"],
    },
    # ── Human Impact ─────────────────────────────────────────────────────────
    {
        "id": "human_001",
        "text": (
            "Pesticide application (especially neonicotinoids) at field-standard doses "
            "reduces wild bee species richness by 30–50% and impairs navigation in "
            "surviving colonies. Transitioning to Integrated Pest Management (IPM) with "
            "biological controls reduces pesticide load by 50–70% while maintaining "
            "crop yields within 5–10% of conventional systems. "
            "(Woodcock et al., 2017 – Science; FAO IPM guidelines)"
        ),
        "tags": ["human_impact", "pesticides", "bees", "IPM", "pollinators"],
    },
    {
        "id": "human_002",
        "text": (
            "Deforestation for agriculture releases 1.5–2.5 Pg C/year globally and "
            "eliminates 135 species/day (estimated). Forest edges created by deforestation "
            "experience 2–8× higher tree mortality due to wind and desiccation. "
            "Reforestation with native species achieves 80% of old-growth biodiversity "
            "within 40–60 years. "
            "(Hansen et al., 2013 – Science; IPBES Global Assessment, 2019)"
        ),
        "tags": ["human_impact", "deforestation", "carbon", "reforestation", "native_species"],
    },
    {
        "id": "human_003",
        "text": (
            "Light and noise pollution in peri-urban areas disrupt nocturnal species "
            "(bats, moths, owls) activity by 30–70%, reducing insect population control "
            "and seed dispersal services. Installing shielded, amber-spectrum lighting "
            "and noise barriers reduces these impacts by 50–60%. "
            "(Gaston et al., 2013 – Philosophical Transactions of the Royal Society B)"
        ),
        "tags": ["human_impact", "light_pollution", "noise_pollution", "nocturnal_species", "urban"],
    },
    # ── Multi-variable interactions ──────────────────────────────────────────
    {
        "id": "multi_001",
        "text": (
            "The soil-biodiversity nexus: SOC > 2% supports 3–5× more soil invertebrate "
            "species than SOC < 0.5%. Invertebrates drive nutrient mineralisation, "
            "supporting plant diversity, which in turn feeds higher trophic levels. "
            "This trophic cascade means improving SOC has a multiplier effect across "
            "the entire food web. "
            "(Bardgett & van der Putten, 2014 – Nature)"
        ),
        "tags": ["multi_variable", "SOC", "invertebrates", "trophic_cascade", "food_web"],
    },
    {
        "id": "multi_002",
        "text": (
            "Water-species survival interaction: In arid systems, a 10% increase in "
            "plant cover (achieved via water harvesting) reduces soil surface temperature "
            "by 3–5 °C, enabling colonisation by heat-sensitive invertebrates and "
            "increasing overall species richness by 25–35%. "
            "(Whitford & Duval, 2009 – Ecology of Desert Systems)"
        ),
        "tags": ["multi_variable", "water", "plant_cover", "temperature", "arid", "invertebrates"],
    },
    {
        "id": "multi_003",
        "text": (
            "Land use × habitat fragmentation × climate interaction: Fragmented landscapes "
            "under climate warming show 2–3× faster local extinction rates than connected "
            "landscapes, because species cannot track shifting climate envelopes. "
            "Landscape-scale connectivity planning (corridors + stepping stones) is the "
            "highest-leverage intervention in warming scenarios. "
            "(Urban, 2015 – Science; Thomas et al., 2004 – Nature)"
        ),
        "tags": ["multi_variable", "fragmentation", "climate_change", "connectivity", "extinction"],
    },
]

# ---------------------------------------------------------------------------
# ChromaDB setup
# ---------------------------------------------------------------------------
_EF = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def _get_or_build_collection() -> chromadb.Collection:
    client = chromadb.Client()  # in-memory; swap to PersistentClient for disk persistence
    try:
        col = client.get_collection("biodiversity_kb", embedding_function=_EF)
    except Exception:
        col = client.create_collection("biodiversity_kb", embedding_function=_EF)
        col.add(
            ids=[c["id"] for c in KNOWLEDGE_CHUNKS],
            documents=[c["text"] for c in KNOWLEDGE_CHUNKS],
            metadatas=[{"tags": ",".join(c["tags"])} for c in KNOWLEDGE_CHUNKS],
        )
    return col


_COLLECTION = _get_or_build_collection()


def retrieve(query: str, n_results: int = 4) -> List[str]:
    """Return the top-n most relevant knowledge chunks for a query."""
    results = _COLLECTION.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]  # list of strings
