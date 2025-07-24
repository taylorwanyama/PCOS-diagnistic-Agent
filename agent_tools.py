from typing import TypedDict, Annotated, Dict
from langchain_core.tools import tool
import joblib
import pandas as pd
from functools import lru_cache
import json

# Define the PCOS input schema
class PCOSInput(TypedDict):
    age: Annotated[int, "Age in years"]
    weight: Annotated[float, "Weight in kilograms"]
    height: Annotated[float, "Height in centimeters"]
    bmi: Annotated[float, "Body Mass Index"]
    blood_group: Annotated[int, "Encoded Blood Group (11 to 18)"]
    pulse_rate: Annotated[float, "Pulse rate in bpm"]
    rr: Annotated[float, "Respiratory rate (breaths/min)"]
    hb: Annotated[float, "Hemoglobin level (g/dl)"]
    cycle_type: Annotated[str, "Cycle type: 'R' (Regular) or 'I' (Irregular)"]
    cycle_length: Annotated[int, "Cycle length in days"]
    marriage_years: Annotated[int, "Years married"]
    pregnant: Annotated[int, "1=Yes, 0=No"]
    number_of_abortions: Annotated[int, "Number of abortions"]
    beta_hcg_1: Annotated[float, "I beta-HCG (mIU/mL)"]
    beta_hcg_2: Annotated[float, "II beta-HCG (mIU/mL)"]
    fsh: Annotated[float, "FSH hormone level (mIU/mL)"]
    lh: Annotated[float, "LH hormone level (mIU/mL)"]
    fsh_lh_ratio: Annotated[float, "FSH/LH ratio"]
    hip: Annotated[float, "Hip circumference (inch)"]
    waist: Annotated[float, "Waist circumference (inch)"]
    waist_hip_ratio: Annotated[float, "Waist to Hip ratio"]
    tsh: Annotated[float, "TSH (mIU/L)"]
    amh: Annotated[float, "AMH (ng/mL)"]
    prl: Annotated[float, "Prolactin (ng/mL)"]
    vit_d3: Annotated[float, "Vitamin D3 (ng/mL)"]
    prg: Annotated[float, "Progesterone (ng/mL)"]
    rbs: Annotated[float, "Random Blood Sugar (mg/dL)"]
    weight_gain: Annotated[int, "1=Yes, 0=No"]
    hair_growth: Annotated[int, "1=Yes, 0=No"]
    skin_darkening: Annotated[int, "1=Yes, 0=No"]
    hair_loss: Annotated[int, "1=Yes, 0=No"]
    pimples: Annotated[int, "1=Yes, 0=No"]
    fast_food: Annotated[int, "1=Yes, 0=No"]
    reg_exercise: Annotated[int, "1=Yes, 0=No"]
    bp_systolic: Annotated[int, "Systolic blood pressure (mmHg)"]
    bp_diastolic: Annotated[int, "Diastolic blood pressure (mmHg)"]
    follicles_left: Annotated[int, "Number of follicles in left ovary"]
    follicles_right: Annotated[int, "Number of follicles in right ovary"]
    avg_size_left: Annotated[float, "Average follicle size in left ovary (mm)"]
    avg_size_right: Annotated[float, "Average follicle size in right ovary (mm)"]
    endometrium: Annotated[float, "Endometrial thickness (mm)"]

@lru_cache(maxsize=1)
def get_model():
    data = joblib.load("pcos_model_revised.joblib")
    return data["model"], data["required_cols"]

@tool
def predict_pcos(
    # These parameter names now match PCOSInput fields
    age: int,
    weight: float,
    height: float,
    bmi: float,
    blood_group: int,
    pulse_rate: float,
    rr: float,
    hb: float,
    cycle_type: str,
    cycle_length: int,
    marriage_years: int,
    pregnant: int,
    number_of_abortions: int,
    beta_hcg_1: float,
    beta_hcg_2: float,
    fsh: float,
    lh: float,
    fsh_lh_ratio: float,
    hip: float,
    waist: float,
    waist_hip_ratio: float,
    tsh: float,
    amh: float,
    prl: float,
    vit_d3: float,
    prg: float,
    rbs: float,
    weight_gain: int,
    hair_growth: int,
    skin_darkening: int,
    hair_loss: int,
    pimples: int,
    fast_food: int,
    reg_exercise: int,
    bp_systolic: int,
    bp_diastolic: int,
    follicles_left: int,
    follicles_right: int,
    avg_size_left: float,
    avg_size_right: float,
    endometrium: float
):
    """Predicts whether an individual has PCOS or not based on input features."""
    print("🔍 predict_pcos tool called")

    data = locals()

    # Convert cycle_type from string to binary
    data["cycle_type"] = 1 if data["cycle_type"].upper() == "R" else 0

    try:
        model, required_cols = get_model()

        # UPDATED field_to_column_map with EXTREMELY precise matching based on observed 0-valued keys
        field_to_column_map = {
            "age": " Age (yrs)",
            "weight": "Weight (Kg)",
            "height": "Height(Cm) ",
            "bmi": "BMI",
            "blood_group": "Blood Group",
            "pulse_rate": "Pulse rate(bpm) ",
            "rr": "RR (breaths/min)",
            "hb": "Hb(g/dl)",
            "cycle_type": "Cycle(R/I)",
            "cycle_length": "Cycle length(days)",
            "marriage_years": "Marraige Status (Yrs)",
            "pregnant": "Pregnant(Y/N)",
            "number_of_abortions": " No. of abortions",
            "beta_hcg_1": " I beta-HCG(mIU/mL)",
            "beta_hcg_2": "II beta-HCG(mIU/mL)",
            "fsh": "FSH(mIU/mL)",
            "lh": "LH(mIU/mL)",
            "fsh_lh_ratio": "FSH/LH",
            "hip": "Hip(inch)",
            "waist": "Waist(inch)",
            "waist_hip_ratio": "Waist:Hip Ratio",
            "tsh": "TSH (mIU/L)",
            "amh": "AMH(ng/mL)",
            "prl": "PRL(ng/mL)",
            "vit_d3": "Vit D3 (ng/mL)",
            "prg": "PRG(ng/mL)",
            "rbs": "RBS(mg/dl)",
            "weight_gain": "Weight gain(Y/N)",
            "hair_growth": "hair growth(Y/N)",
            "skin_darkening": "Skin darkening (Y/N)",
            "hair_loss": "Hair loss(Y/N)",
            "pimples": "Pimples(Y/N)",
            "fast_food": "Fast food (Y/N)",
            "reg_exercise": "Reg.Exercise(Y/N)",
            "bp_systolic": "BP _Systolic (mmHg)",
            "bp_diastolic": "BP _Diastolic (mmHg)",
            "follicles_left": "Follicle No. (L)",
            "follicles_right": "Follicle No. (R)",
            "avg_size_left": "Avg. F size (L) (mm)",
            "avg_size_right": "Avg. F size (R) (mm)",
            "endometrium": "Endometrium (mm)"
        }

        # This part of the code correctly maps your function arguments to the model's column names
        mapped_data = {
            field_to_column_map[k]: data[k]
            for k in data if k in field_to_column_map
        }

        # Ensure all required_cols are present, defaulting to 0 if not provided
        final_features_for_model = {}
        for col in required_cols:
            if col in mapped_data:
                final_features_for_model[col] = mapped_data[col]
            else:
                # This 'else' block will catch any required_cols that still don't find a match,
                # ensuring all model-expected columns are present, even if with 0.
                final_features_for_model[col] = 0


        X = pd.DataFrame([final_features_for_model])[required_cols]
        proba = model.predict_proba(X)[0][1]

        return {
            "pcos_prediction": "Likely PCOS" if proba > 0.5 else "Unlikely PCOS",
            "prediction_probability": round(float(proba), 2),
            "features": final_features_for_model
        }

    except Exception as e:
        return {
            "error": str(e)
        }
        
@tool
def explain_pcos_results(
    pcos_prediction: Annotated[str, "Result from the prediction, e.g. 'Likely PCOS'"],
    features: Annotated[Dict[str, float], "Dictionary of relevant medical features"]
) -> str:
    """Explains the predicted results from PCOS prediction."""
    print("🔍 explain_pcos_results tool called")
    
    explanation = "" 
    reasons = []

    if pcos_prediction == "Likely PCOS":
        
        # Ensure these keys match the FINAL_FEATURES_FOR_MODEL keys
        if features.get("Cycle(R/I)") == 0: 
            reasons.append("irregular menstrual cycles")
        if features.get("hair growth(Y/N)") == 1:
            reasons.append("excessive hair growth")
        if features.get("Skin darkening (Y/N)") == 1:
            reasons.append("skin darkening")
        if features.get("Pimples(Y/N)") == 1:
            reasons.append("presence of acne")
        if features.get("AMH(ng/mL)", 0) > 4.5:
            reasons.append(f"elevated AMH ({features['AMH(ng/mL)']} ng/mL)")
        if features.get("FSH/LH", 1.1) < 1.0: 
            reasons.append(f"low FSH/LH ratio ({features['FSH/LH']})")
        
        explanation = ("Based on: " + ", ".join(reasons)) if reasons else "No significant PCOS indicators detected for 'Likely PCOS' explanation."
        
    elif pcos_prediction == "Unlikely PCOS":
        
        explanation = "According to the model's prediction, you do not have PCOS."
        
    else:
        explanation = "No specific explanation available due to an unknown prediction."

    return explanation

    
@tool
def recommend_next_steps(
    pcos_prediction: Annotated[str, "Prediction result"],
    explanation_text: Annotated[str, "Explanation from previous analysis"]
) -> str:
    """Provides recommendations based on the predicted diagnosis and explanation."""
    if pcos_prediction == "Likely PCOS":
        return (
            "Consider consulting a gynecologist for further testing and treatment. "
            "Adopt a healthier lifestyle including regular physical activity, balanced diet, and stress management."
        )
    else:
        return (
            "Currently, no immediate intervention is necessary. "
            "However, maintaining a healthy lifestyle and monitoring symptoms is recommended."
        )

tools = [predict_pcos, explain_pcos_results, recommend_next_steps]