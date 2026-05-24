import os
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List, Dict

# Custom objects required for loading the keras model
class InteractionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InteractionLayer, self).__init__(**kwargs)
    def call(self, user_embed, food_embed):
        return tf.multiply(user_embed, food_embed)

def asymmetric_allergy_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1, 1])
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    y_true_bool = tf.cast(y_true, tf.bool)
    y_pred_safe = tf.cast(y_pred < 0.5, tf.bool)
    false_negative = tf.logical_and(y_true_bool, y_pred_safe)
    penalty = tf.where(false_negative, 100.0, 1.0)
    return tf.reduce_mean(bce * penalty)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'nutrimatch_model.keras')

try:
    loaded_model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={'InteractionLayer': InteractionLayer, 'asymmetric_allergy_loss': asymmetric_allergy_loss}
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None

def get_ai_recommendations(user_macros: list, user_allergies: list, filtered_food_df: pd.DataFrame, user_max: np.ndarray, top_k: int = 5) -> List[Dict]:
    """
    Predicts and ranks the foods using the TensorFlow model, applies allergy guardrails, and calculates ideal portion size.
    """
    if loaded_model is None or filtered_food_df.empty:
        return []

    num_foods = len(filtered_food_df)
    
    # Inputs Normalization
    u_mac = np.array(user_macros, dtype=np.float32) / user_max
    u_all = np.array(user_allergies, dtype=np.float32)
    
    u_mac_batch = np.tile(u_mac, (num_foods, 1))
    u_all_batch = np.tile(u_all, (num_foods, 1))
    
    # Extract Food Macros and Allergies
    food_macro_cols = ['calories_100g', 'protein_100g', 'fat_100g', 'carbohydrate_100g']
    allergy_cols_food = [
        'contains_gluten', 'contains_dairy', 'contains_nuts', 'contains_peanut', 
        'contains_seafood', 'contains_egg', 'contains_soy', 'contains_celery'
    ]
    
    f_mac_batch = filtered_food_df[food_macro_cols].fillna(0).values.astype(np.float32)
    
    # Re-normalize food macros based on full training max (ideally saved, here we compute from subset max for simplicity, 
    # but in real life we load the scalar)
    # Using fallback normalization for foods (rough estimate based on dataset)
    food_max = np.array([900.0, 100.0, 100.0, 100.0]) # fallback
    f_mac_batch = f_mac_batch / food_max
    
    f_all_batch = filtered_food_df[allergy_cols_food].astype(bool).astype(np.float32).values
    
    # Predict
    match_scores, _ = loaded_model.predict(
        [u_mac_batch, u_all_batch, f_mac_batch, f_all_batch], 
        batch_size=256, 
        verbose=0
    )
    
    # Absolute Guardrail
    has_allergy = np.any(np.logical_and(u_all_batch == 1, f_all_batch == 1), axis=1)
    
    results = []
    target_cal = user_macros[0]
    
    for i in range(num_foods):
        if not has_allergy[i]:
            cal_100g = filtered_food_df.iloc[i]['calories_100g']
            if cal_100g > 0:
                ideal_grams = (target_cal / cal_100g) * 100
                
                results.append({
                    'food_name': str(filtered_food_df.iloc[i]['food_name']),
                    'calories_100g': float(cal_100g),
                    'ideal_grams': float(ideal_grams),
                    'ideal_calories': float(target_cal),
                    'match_score': float(match_scores[i][0])
                })
                
    # Sort and return top K
    results = sorted(results, key=lambda x: x['match_score'], reverse=True)
    return results[:top_k]
