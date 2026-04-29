"""
backend/species_db.py
----------------------
Static metadata dictionary for 22 common Indian medicinal plants.

Each entry is keyed by the lowercase species slug (matches YOLOv8 class name
and the GLB filename convention).

Schema per entry:
    common_name     : English common name
    scientific_name : Binomial Latin name
    family          : Botanical family
    uses            : Comma-separated medicinal uses
    local_name      : Hindi / regional name
    description     : Short descriptive sentence
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Species database
# ---------------------------------------------------------------------------

SPECIES_DB: dict[str, dict[str, str]] = {

    "tulsi": {
        "common_name":     "Tulsi",
        "scientific_name": "Ocimum tenuiflorum",
        "family":          "Lamiaceae",
        "uses":            "Immunity booster, anti-inflammatory, cough and cold relief, stress reducer, fever management",
        "local_name":      "Tulsi / Holy Basil",
        "description":     "Considered the queen of herbs in Ayurveda; revered as sacred in Hindu households.",
    },

    "neem": {
        "common_name":     "Neem",
        "scientific_name": "Azadirachta indica",
        "family":          "Meliaceae",
        "uses":            "Antibacterial, antifungal, blood purifier, skin disorders, dental care, pesticide",
        "local_name":      "Neem / Nimba",
        "description":     "Known as the 'village pharmacy'; nearly every part of the tree has medicinal value.",
    },

    "aloe_vera": {
        "common_name":     "Aloe Vera",
        "scientific_name": "Aloe barbadensis miller",
        "family":          "Asphodelaceae",
        "uses":            "Skin soothing, burn relief, digestive aid, anti-inflammatory, wound healing",
        "local_name":      "Ghritkumari / Gavarpadha",
        "description":     "A succulent plant whose gel is widely used in cosmetics and traditional medicine.",
    },

    "turmeric": {
        "common_name":     "Turmeric",
        "scientific_name": "Curcuma longa",
        "family":          "Zingiberaceae",
        "uses":            "Anti-inflammatory, antioxidant, joint pain relief, digestive health, wound healing",
        "local_name":      "Haldi",
        "description":     "Contains curcumin, one of the most studied phytochemicals for anti-inflammatory effects.",
    },

    "ashwagandha": {
        "common_name":     "Ashwagandha",
        "scientific_name": "Withania somnifera",
        "family":          "Solanaceae",
        "uses":            "Adaptogen, stress and anxiety relief, energy booster, cognitive support, thyroid health",
        "local_name":      "Ashwagandha / Indian Ginseng",
        "description":     "One of the most important herbs in Ayurveda for building physical and mental endurance.",
    },

    "brahmi": {
        "common_name":     "Brahmi",
        "scientific_name": "Bacopa monnieri",
        "family":          "Plantaginaceae",
        "uses":            "Memory enhancer, anxiety reducer, epilepsy support, ADHD aid, antioxidant",
        "local_name":      "Brahmi / Jalanimba",
        "description":     "A creeping wetland herb traditionally used to enhance cognition and calm the nervous system.",
    },

    "giloy": {
        "common_name":     "Giloy",
        "scientific_name": "Tinospora cordifolia",
        "family":          "Menispermaceae",
        "uses":            "Immunity booster, antipyretic, dengue management, diabetes support, detoxification",
        "local_name":      "Giloy / Guduchi / Amrita",
        "description":     "Called 'Amrita' (nectar of immortality) in Sanskrit; a major Ayurvedic tonic herb.",
    },

    "mint": {
        "common_name":     "Mint",
        "scientific_name": "Mentha spicata",
        "family":          "Lamiaceae",
        "uses":            "Digestive aid, nausea relief, headache, freshens breath, irritable bowel syndrome",
        "local_name":      "Pudina",
        "description":     "A cooling aromatic herb used extensively in cooking and traditional remedies across India.",
    },

    "lemongrass": {
        "common_name":     "Lemongrass",
        "scientific_name": "Cymbopogon citratus",
        "family":          "Poaceae",
        "uses":            "Antifungal, cholesterol management, anxiety relief, digestive health, fever reduction",
        "local_name":      "Lemon Grass / Bhustrina",
        "description":     "A tropical grass whose citrus-scented leaves are used in teas and aromatherapy.",
    },

    "curry_leaf": {
        "common_name":     "Curry Leaf",
        "scientific_name": "Murraya koenigii",
        "family":          "Rutaceae",
        "uses":            "Hair growth, diabetes management, antioxidant, digestive health, cholesterol control",
        "local_name":      "Kadi Patta / Meetha Neem",
        "description":     "Fragrant leaves essential to South Indian cooking with notable antidiabetic properties.",
    },

    "moringa": {
        "common_name":     "Moringa",
        "scientific_name": "Moringa oleifera",
        "family":          "Moringaceae",
        "uses":            "Nutritional supplement, anti-inflammatory, blood sugar control, antioxidant, bone health",
        "local_name":      "Sahjan / Drumstick Tree",
        "description":     "Called the 'miracle tree'; leaves are among the most nutritionally dense of any plant.",
    },

    "amla": {
        "common_name":     "Amla",
        "scientific_name": "Phyllanthus emblica",
        "family":          "Phyllanthaceae",
        "uses":            "Vitamin C source, hair care, digestive health, anti-aging, immune boost",
        "local_name":      "Amla / Indian Gooseberry",
        "description":     "One of the richest natural sources of Vitamin C; a cornerstone of Chyawanprash.",
    },

    "noni": {
        "common_name":     "Noni",
        "scientific_name": "Morinda citrifolia",
        "family":          "Rubiaceae",
        "uses":            "Antioxidant, anti-cancer properties, pain relief, immune support, anti-inflammatory",
        "local_name":      "Noni / Ach",
        "description":     "A tropical fruit tree with bitter-tasting fruit traditionally used across Pacific and Asian cultures.",
    },

    "hibiscus": {
        "common_name":     "Hibiscus",
        "scientific_name": "Hibiscus rosa-sinensis",
        "family":          "Malvaceae",
        "uses":            "Blood pressure reduction, hair growth, cholesterol control, antioxidant, liver protection",
        "local_name":      "Gurhal / Jaswand",
        "description":     "Vibrant red flowers used in herbal teas and Ayurvedic hair oils across South Asia.",
    },

    "lavender": {
        "common_name":     "Lavender",
        "scientific_name": "Lavandula angustifolia",
        "family":          "Lamiaceae",
        "uses":            "Anxiety relief, sleep aid, antiseptic, headache, skin care, aromatherapy",
        "local_name":      "Lavender",
        "description":     "Prized for its calming fragrance; oil is a staple in aromatherapy and natural skincare.",
    },

    "rosemary": {
        "common_name":     "Rosemary",
        "scientific_name": "Salvia rosmarinus",
        "family":          "Lamiaceae",
        "uses":            "Memory enhancement, hair growth, antioxidant, digestive health, anti-inflammatory",
        "local_name":      "Rosemary",
        "description":     "A Mediterranean herb widely used for culinary and medicinal purposes; boosts circulation.",
    },

    "basil": {
        "common_name":     "Basil",
        "scientific_name": "Ocimum basilicum",
        "family":          "Lamiaceae",
        "uses":            "Antibacterial, stress relief, digestive aid, anti-inflammatory, blood sugar control",
        "local_name":      "Sabja / Sweet Basil",
        "description":     "A fragrant culinary herb with powerful antibacterial and adaptogenic properties.",
    },

    "chamomile": {
        "common_name":     "Chamomile",
        "scientific_name": "Matricaria chamomilla",
        "family":          "Asteraceae",
        "uses":            "Sleep aid, anxiety reducer, digestive soothing, anti-inflammatory, wound healing",
        "local_name":      "Babune Ka Paudha",
        "description":     "One of the most consumed medicinal herbs globally; famous for its calming daisy-like flowers.",
    },

    "coriander": {
        "common_name":     "Coriander",
        "scientific_name": "Coriandrum sativum",
        "family":          "Apiaceae",
        "uses":            "Digestive health, blood sugar regulation, cholesterol reduction, antioxidant, detox",
        "local_name":      "Dhaniya",
        "description":     "A staple in Indian kitchens; seeds and leaves both carry strong medicinal properties.",
    },

    "fenugreek": {
        "common_name":     "Fenugreek",
        "scientific_name": "Trigonella foenum-graecum",
        "family":          "Fabaceae",
        "uses":            "Blood sugar control, milk production boost, cholesterol reduction, hair care, digestion",
        "local_name":      "Methi",
        "description":     "Seeds and leaves are used in cooking and medicine across South Asia and the Middle East.",
    },

    "ginger": {
        "common_name":     "Ginger",
        "scientific_name": "Zingiber officinale",
        "family":          "Zingiberaceae",
        "uses":            "Nausea relief, anti-inflammatory, digestive aid, cold and flu, pain management",
        "local_name":      "Adrak / Sounth",
        "description":     "A rhizome with potent bioactive compounds; one of the most widely used spice-medicines.",
    },

    "cardamom": {
        "common_name":     "Cardamom",
        "scientific_name": "Elettaria cardamomum",
        "family":          "Zingiberaceae",
        "uses":            "Digestive health, bad breath, blood pressure reduction, anti-nausea, antioxidant",
        "local_name":      "Elaichi",
        "description":     "The 'queen of spices'; pods are used in Ayurveda to treat digestive and respiratory ailments.",
    },
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_species_info(species_slug: str) -> dict[str, str] | None:
    """Return metadata for a species slug, or None if not in the database.

    Args:
        species_slug: Lowercase slug, e.g. 'tulsi', 'aloe_vera'.

    Returns:
        Metadata dict or None.
    """
    return SPECIES_DB.get(species_slug.lower().replace(" ", "_"))


def list_all_species() -> list[dict[str, str]]:
    """Return a list of all species entries, each enriched with its slug key."""
    return [
        {"slug": slug, **data}
        for slug, data in SPECIES_DB.items()
    ]
