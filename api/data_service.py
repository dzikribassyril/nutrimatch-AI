import pandas as pd
import numpy as np

# Load Data Once
FOOD_DATA_PATH = '../Data/train_ready_dataset.csv'
USER_DATA_PATH = '../Data/user_profile_features_schema.csv'

try:
    food_df = pd.read_csv(FOOD_DATA_PATH)
    user_df = pd.read_csv(USER_DATA_PATH)
except Exception as e:
    print(f"Error loading CSV files: {e}")
    food_df = pd.DataFrame()
    user_df = pd.DataFrame()

def get_food_data():
    return food_df

def filter_foods_by_preferences(categories: list, ingredients: list, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    TODO for Data Science Team:
    Once the 'food_category' and 'main_ingredient' columns are added to train_ready_dataset.csv, 
    uncomment the logic below to perform precise filtering!
    """
    
    # === FUTURE LOGIC (COMMENTED OUT FOR NOW) ===
    # filtered_df = dataset.copy()
    # 
    # if categories:
    #     cat_lower = [c.lower() for c in categories]
    #     filtered_df = filtered_df[filtered_df['food_category'].str.lower().isin(cat_lower)]
    #
    # if ingredients:
    #     ing_lower = [i.lower() for i in ingredients]
    #     filtered_df = filtered_df[filtered_df['main_ingredient'].str.lower().isin(ing_lower)]
    # 
    # return filtered_df
    
    # === CURRENT FALLBACK LOGIC (MVP) ===
    # Combine both lists and do a substring search on the food name
    all_keywords = categories + ingredients
    if not all_keywords:
        return dataset
        
    all_keywords = [k.lower().strip() for k in all_keywords if k.strip()]
    
    mask = dataset['food_name_clean'].apply(
        lambda name: any(kw in str(name).lower() for kw in all_keywords)
    )
    
    filtered_df = dataset[mask].copy()
    if filtered_df.empty:
        print(f"Warning: No foods found matching {all_keywords}. Falling back to full dataset.")
        return dataset
        
    return filtered_df

# Macro normalization factors based on the training data limits
# We approximate these from the dataset
user_macro_cols = ['target_calorie', 'protein_target_g', 'fat_target_g', 'carb_target_g']
if not user_df.empty:
    user_max = np.max(user_df[user_macro_cols].fillna(0).values, axis=0) + 1e-9
else:
    user_max = np.array([3000.0, 200.0, 100.0, 400.0]) # Fallback sensible max values

def get_user_max():
    return user_max
