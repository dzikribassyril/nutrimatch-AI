"""Clean and enrich NutriMatch food data with meal pairing metadata.

Run from the project root:
    python Data/clean_enrich_food_dataset.py
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
INPUT_PATH = DATA_DIR / "train_ready_dataset_v2.csv"
OUTPUT_PATH = DATA_DIR / "train_ready_dataset_v3.csv"
AUDIT_PATH = DATA_DIR / "train_ready_dataset_v3_audit.csv"


BOOL_MAP = {
    "true": True,
    "false": False,
    "1": True,
    "0": False,
    "1.0": True,
    "0.0": False,
    "yes": True,
    "no": False,
}


VEGETABLE_KEYWORDS = {
    "bayam", "kangkung", "sawi", "kol", "kubis", "wortel", "brokoli",
    "buncis", "kacang panjang", "tauge", "toge", "labu", "pare",
    "terong", "oyong", "genjer", "pakis", "daun", "rebung", "jantung pisang",
    "jamur", "tomat", "timun", "mentimun", "selada", "gudeg", "cap cai",
    "capcay", "urap", "lalap", "karedok", "pecel sayur", "sayur",
}

FRUIT_KEYWORDS = {
    "pisang", "apel", "jeruk", "mangga", "pepaya", "alpukat", "anggur",
    "nanas", "semangka", "melon", "duku", "duwet", "jambu", "kedondong",
    "salak", "sirsak", "belimbing", "nangka", "rambutan", "buah naga",
    "sawo", "markisa", "manggis", "carica",
}

STAPLE_KEYWORDS = {
    "nasi", "beras", "mie", "mi ", "bihun", "kwetiau", "lontong",
    "ketupat", "roti", "kentang", "singkong", "ubi", "talas", "sagu",
    "jagung", "oat", "pasta", "makaroni", "spaghetti", "ketan",
}

RICE_NOODLE_KEYWORDS = {
    "nasi", "beras", "mie", "mi ", "bihun", "kwetiau", "lontong",
    "ketupat", "soun", "nasi tim",
}

SWEET_SNACK_KEYWORDS = {
    "kue", "dodol", "bagea", "ledre", "bakpia", "biskuit", "permen",
    "coklat", "agar-agar", "selai", "jam ", "semprong", "talam",
    "apem", "putu", "pia", "mangkok", "lumpur", "kue sus", "tambang",
    "serbuk coklat", "noga", "enting",
}

SAVORY_SNACK_KEYWORDS = {
    "keripik", "kripik", "kerupuk", "rempeyek", "sukro", "atom", "emping",
    "lanting", "kecimpring", "ceriping", "serimping",
}

PROTEIN_KEYWORDS = {
    "ayam", "ikan", "telur", "daging", "sapi", "kambing", "bebek",
    "udang", "cumi", "kepiting", "kerang", "tahu", "tempe", "oncom",
    "kedelai", "bandeng", "lele", "tongkol", "tuna", "mujair", "mujahir", "patin",
    "teri", "abon", "rendang", "sate", "opor", "semur", "dendeng",
}

ANIMAL_PROTEIN_KEYWORDS = {
    "ayam", "ikan", "telur", "daging", "sapi", "kambing", "bebek",
    "udang", "cumi", "kepiting", "kerang", "bandeng", "lele", "tongkol",
    "tuna", "mujair", "mujahir", "patin", "teri",
}

PLANT_PROTEIN_KEYWORDS = {"tahu", "tempe", "oncom", "kedelai", "kacang hijau", "kacang merah"}

DAIRY_KEYWORDS = {
    "susu", "keju", "yoghurt", "yogurt", "kepala susu", "krim",
}

COMPLETE_MENU_KEYWORDS = {
    "nasi goreng", "nasi uduk", "nasi kuning", "nasi campur", "nasi rames",
    "nasi padang", "mie ayam", "mie bakso", "mie aceh", "mie goreng",
    "bakso", "soto", "sup ", "rawon", "gulai", "opor", "rendang",
    "semur", "tongseng", "gado-gado", "gado gado", "pecel", "karedok",
    "lontong sayur", "ketupat sayur", "bubur", "cap cai", "capcay",
}

INGREDIENT_ONLY_KEYWORDS = {
    "minyak", "margarin", "mentega", "garam", "gula pasir", "tepung",
    "pati", "kanji", "sagu kering", "sirup", "setrup", "kaldu",
    "bumbu", "terasi", "kecap", "saus", "sambal", "ragi", "cuka",
}

ODD_OR_LOW_VALUE_KEYWORDS = {
    "ampas", "dideh", "bagian yang larut", "kulit ", "tulang",
}

NON_HALAL_KEYWORDS = {"babi", "ham", "bacon", "arak", "tuak", "alkohol"}

READY_PROTEIN_KEYWORDS = {
    "goreng", "rebus", "kukus", "pepes", "bakar", "panggang", "asap",
    "dendeng", "abon", "rendang", "semur", "sate", "opor", "gulai",
    "masakan", "ceplok", "dadar", "teriyaki", "bakso", "soto", "coto",
}

RAW_STAPLE_PATTERNS = {
    "beras pecah kulit", "beras giling", "beras ketan", "mie basah",
    "mie bendo", "mie celon", "mie pangsit basah", "mie sagu", "bihun",
}


def normalize_bool(value, default=False) -> bool:
    if pd.isna(value):
        return default
    return BOOL_MAP.get(str(value).strip().lower(), default)


def contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def has_word(text: str, word: str) -> bool:
    return re.search(rf"(^|[^a-z0-9]){re.escape(word)}([^a-z0-9]|$)", text) is not None


def contains_any_word(text: str, keywords: set[str]) -> bool:
    for keyword in keywords:
        if " " in keyword or "-" in keyword:
            if keyword in text:
                return True
        elif has_word(text, keyword):
            return True
    return False


def classify_main_ingredient(name: str, current: str) -> str:
    if contains_any(name, DAIRY_KEYWORDS):
        return "susu"
    if contains_any_word(name, {"tempe", "tahu", "oncom", "kedelai"}):
        return "kedelai"
    if contains_any_word(name, {"telur"}):
        return "telur"
    if contains_any_word(name, {"ayam"}):
        return "ayam"
    if contains_any_word(name, {"ikan", "bandeng", "lele", "tongkol", "tuna", "mujair", "mujahir", "patin", "teri", "haruan", "haruwan"}):
        return "ikan"
    if contains_any_word(name, {"udang", "cumi", "kepiting", "kerang"}):
        return "seafood"
    if contains_any_word(name, {"sapi", "daging", "rendang", "dendeng"}):
        return "sapi"
    if contains_any_word(name, {"kambing"}):
        return "kambing"
    if contains_any_word(name, {"babi"}):
        return "babi"
    if contains_any(name, PLANT_PROTEIN_KEYWORDS | {"kacang tanah"}):
        return "kacang"
    if contains_any(name, VEGETABLE_KEYWORDS):
        return "sayuran"
    if contains_any(name, FRUIT_KEYWORDS):
        return "buah"
    if contains_any(name, {"beras", "nasi"}):
        return "beras"
    if contains_any(name, {"mie", "mi ", "roti", "terigu", "gandum"}):
        return "terigu"
    if contains_any(name, {"singkong", "ubi", "talas"}):
        return "singkong"
    return current if current and current != "nan" else "other"


def classify_cooking(name: str, current: str) -> str:
    if contains_any(name, {"tumis", "oseng"}):
        return "tumis"
    if contains_any(name, {"kuah", "sup", "soto", "rawon", "gulai", "opor", "berkuah"}):
        return "berkuah"
    if contains_any(name, {"rebus", "kukus", "pepes", "tim"}):
        return "rebus_kukus"
    if contains_any(name, {"goreng", "ceplok", "dadar"}):
        return "gorengan"
    if contains_any(name, {"bakar", "panggang"}):
        return "bakar"
    return current


def classify_category_and_type(row: pd.Series, name: str, main_ing: str, cooking: str) -> tuple[str, str]:
    cat = str(row.get("food_category", "")).strip().lower()
    item_type = str(row.get("recommendation_item_type", "")).strip().lower()

    if contains_any(name, INGREDIENT_ONLY_KEYWORDS):
        return cat or "lainnya", "ingredient"
    if "keju kacang" in name:
        return "snack_dessert", "snack"
    if contains_any(name, DAIRY_KEYWORDS):
        if "susu" in name or "krim" in name:
            return "minuman", "drink"
        return "lainnya", "dairy"
    if contains_any(name, SWEET_SNACK_KEYWORDS):
        return "snack_dessert", "snack"
    if contains_any(name, SAVORY_SNACK_KEYWORDS):
        return "gorengan" if "goreng" in name else "snack_dessert", "snack"
    if "roti" in name:
        return "karbohidrat_pokok", "staple"
    if contains_any(name, COMPLETE_MENU_KEYWORDS) or cat == "berkuah":
        return "berkuah" if cooking == "berkuah" else cat, "menu"
    if contains_any(name, VEGETABLE_KEYWORDS):
        return "sayuran", "vegetable" if cooking in ("mentah_segar", "other") else "menu"
    if contains_any(name, FRUIT_KEYWORDS):
        return "buah", "fruit"
    if contains_any(name, PROTEIN_KEYWORDS):
        if contains_any(name, PLANT_PROTEIN_KEYWORDS) or main_ing in {"kedelai", "kacang"}:
            return "lauk_nabati", "plant_protein" if cooking in ("other", "mentah_segar") else "menu"
        return "lauk_hewani", "protein_dish" if cooking in ("other", "mentah_segar") else "menu"
    if contains_any(name, STAPLE_KEYWORDS):
        return "karbohidrat_pokok", "staple" if cooking in ("other", "mentah_segar") else "menu"
    return cat, item_type


def classify_pairing(row: pd.Series, name: str) -> tuple[str, str, str]:
    cat = str(row.get("food_category", "")).lower()
    item_type = str(row.get("recommendation_item_type", "")).lower()
    cooking = str(row.get("cooking_category", "")).lower()
    main_ing = str(row.get("main_ingredient", "")).lower()

    if item_type == "ingredient":
        return "invalid", "invalid", "not_ready_to_eat"
    if "keju kacang" in name:
        return "sweet_snack", "sweet_snack", "sweet_snack_needs_dairy_pairing"
    if contains_any(name, DAIRY_KEYWORDS) or item_type == "dairy":
        return "dairy_pairing", "dairy", "dairy_or_cheese_pairing"
    if item_type == "drink" or cat == "minuman":
        return "drink", "drink", "drink_not_main_meal"
    if contains_any(name, SAVORY_SNACK_KEYWORDS):
        return "savory_snack", "snack", "snack_not_main_meal"
    if contains_any(name, SWEET_SNACK_KEYWORDS) or (item_type == "snack" and cat == "snack_dessert"):
        return "sweet_snack", "sweet_snack", "sweet_snack_needs_dairy_pairing"
    if item_type == "snack":
        return "savory_snack", "snack", "snack_not_main_meal"
    if contains_any(name, COMPLETE_MENU_KEYWORDS) or item_type == "menu" and cat in {"berkuah"}:
        return "complete_menu", "complete", "standalone_complete_menu"
    if cat == "karbohidrat_pokok" or item_type == "staple":
        if any(pattern == name for pattern in RAW_STAPLE_PATTERNS):
            return "raw_staple", "invalid", "raw_staple_requires_cooking"
        if contains_any(name, RICE_NOODLE_KEYWORDS):
            return "rice_noodle_staple", "staple", "needs_traditional_protein_and_vegetable"
        if "roti" in name:
            return "bread_staple", "staple", "bread_needs_dairy_or_light_pairing"
        return "savory_staple", "staple", "needs_traditional_protein_and_vegetable"
    if cat in {"lauk_hewani", "lauk_nabati"} or item_type in {"protein_dish", "plant_protein"}:
        if cooking in {"other", "mentah_segar"} and not contains_any(name, READY_PROTEIN_KEYWORDS):
            return "protein_ingredient", "invalid", "protein_not_ready_as_dish"
        if cooking in {"gorengan", "bakar", "rebus_kukus", "berkuah", "tumis", "olahan"} or main_ing in {
            "ayam", "ikan", "seafood", "sapi", "kambing", "telur", "kedelai", "kacang"
        }:
            return "traditional_protein", "protein", "traditional_protein_for_rice_noodle"
        return "protein_ingredient", "invalid", "protein_not_ready_as_dish"
    if cat == "sayuran" or item_type == "vegetable":
        if cooking in {"tumis", "berkuah", "rebus_kukus", "olahan"} or contains_any(name, {"sayur", "tumis", "kuah", "sup", "cap cai", "capcay", "gudeg", "pecel", "urap", "karedok"}):
            return "cooked_or_ready_vegetable", "vegetable", "traditional_vegetable_pairing"
        return "raw_vegetable", "invalid", "raw_vegetable_requires_preparation"
    if cat == "buah" or item_type == "fruit":
        return "fruit_side", "fruit", "fruit_side_or_snack"
    return "unknown", "invalid", "unclear_food_context"


def recommendability(row: pd.Series, name: str) -> tuple[bool, str]:
    cal = float(row.get("calories_100g", 0) or 0)
    cat = str(row.get("food_category", "")).lower()
    item_type = str(row.get("recommendation_item_type", "")).lower()
    cooking = str(row.get("cooking_category", "")).lower()
    pairing_role = str(row.get("pairing_role", "")).lower()
    pairing_group = str(row.get("pairing_group", "")).lower()

    reasons = []
    if cal <= 0:
        reasons.append("zero_or_invalid_calorie")
    if contains_any(name, INGREDIENT_ONLY_KEYWORDS):
        reasons.append("ingredient_only")
    if contains_any(name, ODD_OR_LOW_VALUE_KEYWORDS):
        reasons.append("not_normal_serving_food")
    if pairing_role == "invalid":
        reasons.append(str(row.get("pairing_notes", "invalid_context")))
    if pairing_group == "raw_staple":
        reasons.append("raw_staple_requires_cooking")
    if cooking == "mentah_segar" and cat != "buah":
        reasons.append("raw_item_requires_preparation")
    if item_type == "drink" and not contains_any(name, DAIRY_KEYWORDS):
        reasons.append("non_dairy_drink_not_used_in_meal_combo")
    if "kacang babi" in name or "tempe kacang babi" in name:
        reasons.append("ambiguous_babi_term_review")

    if reasons:
        return False, "|".join(dict.fromkeys(reasons))
    return True, ""


def enrich_allergen_flags(df: pd.DataFrame, idx: int, name: str) -> None:
    base_text = " ".join(
        str(df.at[idx, col]).lower()
        for col in ["main_ingredient", "base_ingredient", "base_ingredient_tags", "recipe_ingredients_reference"]
        if col in df.columns and not pd.isna(df.at[idx, col])
    )
    combined = f"{name} {base_text}"

    allergen_rules = {
        "contains_dairy": DAIRY_KEYWORDS,
        "contains_nuts": {"kacang", "kenari", "mete", "wijen"},
        "contains_peanut": {"kacang tanah"},
        "contains_seafood": {"ikan", "udang", "cumi", "kepiting", "kerang", "bandeng", "lele", "tongkol", "tuna", "mujair", "mujahir", "patin", "teri", "haruan", "haruwan", "belut"},
        "contains_egg": {"telur"},
        "contains_soy": {"kedelai", "tempe", "tahu", "oncom", "tauco"},
        "contains_gluten": {"terigu", "gandum", "roti", "mie", "bihun", "pasta", "makaroni", "spaghetti", "biskuit", "kue"},
    }
    for col, keywords in allergen_rules.items():
        if col in df.columns and contains_any(combined, keywords):
            df.at[idx, col] = True


def meal_suitability(row: pd.Series, name: str) -> tuple[bool, bool, bool, str]:
    pairing_role = str(row.get("pairing_role", "")).lower()
    cat = str(row.get("food_category", "")).lower()
    cooking = str(row.get("cooking_category", "")).lower()

    if not normalize_bool(row.get("is_recommendable_food"), True):
        return False, False, False, "not_recommendable"
    if pairing_role in {"sweet_snack", "dairy", "fruit"}:
        return True, False, False, "breakfast_or_light_meal_only"
    if pairing_role == "drink":
        return False, False, False, "drink_not_combo_component"
    if pairing_role == "complete":
        return True, True, True, "complete_menu_all_meals"
    if pairing_role in {"staple", "protein", "vegetable"}:
        if cat == "sayuran" and cooking == "mentah_segar":
            return False, False, False, "raw_vegetable"
        return True, True, True, "main_meal_component"
    return False, False, False, "unclear_suitability"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    audit_rows = []

    for text_col in ["recommendation_exclusion_reason", "halal_status"]:
        if text_col not in df.columns:
            df[text_col] = ""
        df[text_col] = df[text_col].fillna("").astype(str)

    bool_cols = [
        "contains_gluten", "contains_dairy", "contains_nuts", "contains_peanut",
        "contains_seafood", "contains_egg", "contains_soy", "contains_celery",
        "ingredient_only_flag", "raw_ingredient_flag", "is_recommendable_food",
        "is_halal_candidate", "contains_non_halal_ingredient",
        "suitable_breakfast", "suitable_lunch", "suitable_dinner",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda value: normalize_bool(value, False))

    for idx, row in df.copy().iterrows():
        name = str(row.get("food_name_clean") or row.get("food_name") or "").strip().lower()
        old = row.to_dict()
        changes = []

        cooking = classify_cooking(name, str(row.get("cooking_category", "other")).strip().lower())
        main_ing = classify_main_ingredient(name, str(row.get("main_ingredient", "other")).strip().lower())
        cat, item_type = classify_category_and_type(row, name, main_ing, cooking)

        df.at[idx, "cooking_category"] = cooking
        df.at[idx, "main_ingredient"] = main_ing
        df.at[idx, "food_category"] = cat
        df.at[idx, "recommendation_item_type"] = item_type

        enrich_allergen_flags(df, idx, name)

        temp_row = df.loc[idx]
        pairing_group, pairing_role, pairing_notes = classify_pairing(temp_row, name)
        df.at[idx, "pairing_group"] = pairing_group
        df.at[idx, "pairing_role"] = pairing_role
        df.at[idx, "pairing_notes"] = pairing_notes

        ingredient_only = normalize_bool(df.at[idx, "ingredient_only_flag"], False) or item_type == "ingredient"
        raw_ingredient = (
            normalize_bool(df.at[idx, "raw_ingredient_flag"], False)
            or (cooking == "mentah_segar" and cat != "buah")
        )
        df.at[idx, "ingredient_only_flag"] = ingredient_only
        df.at[idx, "raw_ingredient_flag"] = raw_ingredient

        non_halal = normalize_bool(df.at[idx, "contains_non_halal_ingredient"], False) or contains_any(name, NON_HALAL_KEYWORDS)
        df.at[idx, "contains_non_halal_ingredient"] = non_halal
        df.at[idx, "is_halal_candidate"] = not non_halal
        if non_halal:
            df.at[idx, "halal_status"] = "non_halal_or_review"

        is_rec, exclusion_reason = recommendability(df.loc[idx], name)
        df.at[idx, "is_recommendable_food"] = is_rec
        df.at[idx, "recommendation_exclusion_reason"] = exclusion_reason

        breakfast, lunch, dinner, suitability_notes = meal_suitability(df.loc[idx], name)
        df.at[idx, "suitable_breakfast"] = breakfast
        df.at[idx, "suitable_lunch"] = lunch
        df.at[idx, "suitable_dinner"] = dinner
        df.at[idx, "pairing_suitability_notes"] = suitability_notes
        meal_tags = "|".join(
            label for flag, label in [(breakfast, "breakfast"), (lunch, "lunch"), (dinner, "dinner")] if flag
        )
        df.at[idx, "meal_time_tags"] = meal_tags
        df.at[idx, "primary_meal_time"] = meal_tags.split("|")[0] if meal_tags else "excluded"

        for col in [
            "cooking_category", "main_ingredient", "food_category", "recommendation_item_type",
            "ingredient_only_flag", "raw_ingredient_flag", "is_recommendable_food",
            "suitable_breakfast", "suitable_lunch", "suitable_dinner",
            "is_halal_candidate", "contains_non_halal_ingredient",
        ]:
            if str(old.get(col, "")) != str(df.at[idx, col]):
                changes.append(f"{col}:{old.get(col, '')}->{df.at[idx, col]}")

        if changes or not is_rec:
            audit_rows.append({
                "food_id": row.get("food_id"),
                "food_name": row.get("food_name"),
                "changes": "; ".join(changes),
                "pairing_group": pairing_group,
                "pairing_role": pairing_role,
                "is_recommendable_food": is_rec,
                "exclusion_reason": exclusion_reason,
            })

    df["feature_rule_version"] = "food_context_pairing_rules_v3"
    df["menu_ready_rule_version"] = "menu_ready_pairing_filter_v3"

    df.to_csv(OUTPUT_PATH, index=False)
    pd.DataFrame(audit_rows).to_csv(AUDIT_PATH, index=False)

    print(f"Wrote {OUTPUT_PATH} ({len(df)} rows)")
    print(f"Wrote {AUDIT_PATH} ({len(audit_rows)} audited rows)")
    print("Recommendable rows:", int(df["is_recommendable_food"].sum()))
    print("Pairing roles:")
    print(df["pairing_role"].value_counts(dropna=False).to_string())
    print("Pairing groups:")
    print(df["pairing_group"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
