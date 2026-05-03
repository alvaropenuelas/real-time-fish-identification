"""
Species index → display name mapping for both datasets.
Fish4Knowledge classes use their original folder names (fish_01 … fish_23).
Mediterranean classes use scientific names from iNaturalist.
"""

# Fish4Knowledge — 23 Indo-Pacific reef species
F4K_SPECIES = {
    "fish_01": "Dascyllus reticulatus",
    "fish_02": "Plectroglyphidodon dickii",
    "fish_03": "Chromis chrysura",
    "fish_04": "Amphiprion clarkii",
    "fish_05": "Chaetodon lunulatus",
    "fish_06": "Chaetodon trifascialis",
    "fish_07": "Myripristis kuntee",
    "fish_08": "Acanthurus nigrofuscus",
    "fish_09": "Hemigymnus fasciatus",
    "fish_10": "Neoniphon sammara",
    "fish_11": "Abudefduf vaigiensis",
    "fish_12": "Canthigaster valentini",
    "fish_13": "Pomacentrus moluccensis",
    "fish_14": "Zebrasoma scopas",
    "fish_15": "Hemigymnus melapterus",
    "fish_16": "Lutjanus fulvus",
    "fish_17": "Scolopsis bilineata",
    "fish_18": "Scaridae",
    "fish_19": "Pempheris vanicolensis",
    "fish_20": "Zanclus cornutus",
    "fish_21": "Neoglyphidodon nigroris",
    "fish_22": "Balistapus undulatus",
    "fish_23": "Siganus fuscescens",
}

# Mediterranean — 10 species downloaded from iNaturalist
MED_SPECIES = {
    "Sparus_aurata":          {"common": "Gilthead seabream",   "taxon_id": 1494783},
    "Diplodus_sargus":        {"common": "White seabream",      "taxon_id": 118669},
    "Scorpaena_scrofa":       {"common": "Red scorpionfish",    "taxon_id": 84861},
    "Mullus_surmuletus":      {"common": "Striped red mullet",  "taxon_id": 118619},
    "Epinephelus_marginatus": {"common": "Dusky grouper",       "taxon_id": 100119},
    "Coris_julis":            {"common": "Rainbow wrasse",      "taxon_id": 50968},
    "Oblada_melanura":        {"common": "Saddled seabream",    "taxon_id": 118662},
    "Thalassoma_pavo":        {"common": "Ornate wrasse",       "taxon_id": 50972},
    "Sarpa_salpa":            {"common": "Salema porgy",        "taxon_id": 118663},
    "Muraena_helena":         {"common": "Mediterranean moray", "taxon_id": 118590},
}

# Common display name lookup: folder_name → display string
DISPLAY_NAMES = {k: v for k, v in F4K_SPECIES.items()}
DISPLAY_NAMES.update({k: v["common"] for k, v in MED_SPECIES.items()})
