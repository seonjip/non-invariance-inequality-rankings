"""Paths, indicator lists, covariate sets and random seeds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "HNY7_OE_16_18.sav"
OUT = ROOT / "output"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
for p in (OUT, TABLES, FIGURES):
    p.mkdir(parents=True, exist_ok=True)

# eleven indicators, ordered as in the manuscript tables (ascending prevalence is applied later)
INDICATORS = ["caries_exp", "no_checkup", "chew_disc", "poor_sroh", "perio", "untreated",
              "toothpain", "lt20teeth", "unmet", "denture_need", "edentulous"]

# five alternative thresholds of the same constructs
THRESHOLD_VARIANTS = ["lt25teeth", "sroh_broad", "cpi4", "untreated2", "chew_broad"]

# indicators in the threshold-comparison table, paired construct by construct
THRESHOLD_SET = ["lt20teeth", "lt25teeth", "poor_sroh", "sroh_broad", "perio", "cpi4",
                 "untreated", "untreated2", "chew_disc", "chew_broad"]

# clinically examined conditions only
CLINICAL = ["edentulous", "denture_need", "lt20teeth", "untreated", "perio", "caries_exp"]

# covariate sets: model 1 is the main specification
COVSETS = {"M0": (), "M1": ("age", "age2", "sex"), "M2": ("age", "age2", "sex", "town", "apt")}

# socioeconomic rank definitions: variable name and category order from most to least advantaged
SEP = {
    "income_q5": ("ho_incm5", [5, 4, 3, 2, 1], 19),
    "income_q4": ("ho_incm", [4, 3, 2, 1], 19),
    "income_d10": ("inc_dec", list(range(10, 0, -1)), 19),
    "income_individual_q5": ("incm5", [5, 4, 3, 2, 1], 19),
    "education_4": ("edu", [4, 3, 2, 1], 30),
    "education_3": ("edu3", [3, 2, 1], 30),
}

AGE_GROUPS = [("35-44", 35, 44), ("45-54", 45, 54), ("55-64", 55, 64),
              ("65-74", 65, 74), ("75+", 75, 200)]

N_BOOT = 2000
SEEDS = {"income_main": 901, "education_main": 902, "favourable_coding": 903,
         "age_groups": 301, "covsets": 401, "sep_variants": 501,
         "complete_case": 904, "quadratic": 601}

LABELS = {
    "edentulous": "Complete tooth loss (no remaining natural teeth)",
    "denture_need": "Need for a removable denture (partial or complete)",
    "lt20teeth": "Fewer than 20 remaining natural teeth",
    "chew_disc": "Chewing difficulty (uncomfortable or very uncomfortable)",
    "unmet": "Unmet need for dental care in the past 12 months",
    "untreated": "Untreated dental caries (one or more decayed permanent teeth)",
    "perio": "Periodontal pockets (community periodontal index score of 3 or 4)",
    "toothpain": "Toothache in the past 12 months",
    "poor_sroh": "Poor or very poor self-rated oral health",
    "no_checkup": "No dental check-up in the past 12 months",
    "caries_exp": "Any caries experience in the permanent teeth (DMFT of one or more)",
    "lt25teeth": "Fewer than 25 remaining natural teeth",
    "sroh_broad": "Fair, poor or very poor self-rated oral health",
    "cpi4": "Periodontal pockets (community periodontal index score of 4)",
    "untreated2": "Untreated dental caries (two or more decayed permanent teeth)",
    "chew_broad": "Chewing difficulty (fair, uncomfortable or very uncomfortable)",
}
