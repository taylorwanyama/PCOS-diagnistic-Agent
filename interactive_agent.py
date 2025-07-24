from ai_agent import app 
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from pydantic import BaseModel, ValidationError, conint, confloat
from memory import save_to_db
import uuid
import json


user_id = f"user_{uuid.uuid4().hex[:8]}"
thread_id = "cli-session"

print("\n🤖: Hello. Welcome to the PCOS Diagnostic Assistant!")
print("Please respond using numbers. For Yes/No fields, use 0 = No and 1 = Yes.\n")

class PCOSInput(BaseModel):
    age_in_years: conint(ge=10, le=100)
    weight_kgs: confloat(gt=30, lt=200)
    height_cm: confloat(gt=100, lt=250)
    bmi: confloat(gt=10, lt=50)
    blood_group: conint(ge=11, le=18)
    pulse_rate_bpm: confloat(gt=30, le=150)
    rr_breaths_min: confloat(gt=8, le=40)
    hb_g_dl: confloat(gt=5, le=20)
    cycle_type: str
    cycle_length_days: conint(gt=0, le=100)
    marriage_years: conint(ge=0, le=50)
    pregnant: conint(ge=0, le=1)
    number_of_abortions: conint(ge=0, le=10)
    beta_hcg_1: confloat(gt=0)
    beta_hcg_2: confloat(gt=0)
    fsh_miu_ml: confloat(gt=0)
    lh_miu_ml: confloat(gt=0)
    fsh_lh_ratio: confloat(ge=0)
    tsh_miu_l: confloat(gt=0)
    prl_ng_ml: confloat(gt=0)
    vit_d3_ng_ml: confloat(gt=0)
    prg_ng_ml: confloat(gt=0)
    rbs_mg_dl: confloat(gt=0)
    amh_ng_ml: confloat(gt=0)
    hair_growth: conint(ge=0, le=1)
    skin_darkening: conint(ge=0, le=1)
    pimples: conint(ge=0, le=1)
    hair_loss: conint(ge=0, le=1)
    weight_gain: conint(ge=0, le=1)
    fast_food: conint(ge=0, le=1)
    reg_exercise: conint(ge=0, le=1)
    bp_systolic_mmHg: confloat(gt=50, le=200)
    bp_diastolic_mmHg: confloat(gt=30, le=150)
    hip_inch: confloat(gt=20, le=80)
    waist_inch: confloat(gt=20, le=80)
    waist_hip_ratio: confloat(ge=0.5, le=1.5)
    follicles_left: conint(ge=0, le=50)
    follicles_right: conint(ge=0, le=50)
    avg_size_left_mm: confloat(ge=0, le=20)
    avg_size_right_mm: confloat(ge=0, le=20)
    endometrium_mm: confloat(ge=0, le=20)


data = {}
for field_name, field_info in PCOSInput.model_fields.items():
    while True:
        try:
            input_message = f"Enter {field_name.replace('_', ' ')} ({field_info.annotation.__name__}): "
            value = input(input_message)

            if field_name == "cycle_type":
                if value.upper() not in ["R", "I"]:
                    raise ValueError("Cycle type must be 'R' (Regular) or 'I' (Irregular).")
                data[field_name] = value.upper()
            elif field_info.annotation is int:
                data[field_name] = int(value)
            elif field_info.annotation is float:
                data[field_name] = float(value)
            else:
                data[field_name] = value 

            
            if hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra is not None:
                constraints = field_info.json_schema_extra
                current_value = data[field_name] 

                if 'ge' in constraints and current_value < constraints['ge']:
                    raise ValueError(f"Value too low (minimum: {constraints['ge']})")
                if 'le' in constraints and current_value > constraints['le']:
                    raise ValueError(f"Value too high (maximum: {constraints['le']})")
                if 'gt' in constraints and current_value <= constraints['gt']:
                    raise ValueError(f"Value too low (must be greater than: {constraints['gt']})")
                if 'lt' in constraints and current_value >= constraints['lt']:
                    raise ValueError(f"Value too high (must be less than: {constraints['lt']})")

            break 
        except ValueError as e:
            print(f"Invalid input for {field_name}: {e}. Please try again.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Please try again.")

try:
    user_data = PCOSInput(**data)
    # DEBUG PRINT: Verify user_data content immediately after creation
    print(f"DEBUG: User Data after Pydantic validation: {user_data.model_dump()}")
except ValidationError as e:
    print(f"Invalid input for PCOSInput: {e}")
    exit() 


state = {
    "messages": [HumanMessage(content="Start PCOS diagnosis with provided patient data.")],
    "features": user_data.model_dump(), # Store original features in state
    "user_id": user_id,
    "thread_id": thread_id
}

# DEBUG PRINT: Verify state features before invoking app
print(f"DEBUG: State features before invoke: {state.get('features')}")


print("\n🤖: Analyzing the provided information...")


final_state = app.invoke(state, config={"configurable": {"thread_id": thread_id}})

prediction = final_state.get("pcos_prediction", "Unknown")
probability = final_state.get("prediction_probability", 0.0)


raw_predict_pcos_tool_output = {}
raw_explanation_text = ""
raw_recommendation_text = ""


for msg in final_state["messages"]:
    if isinstance(msg, ToolMessage):
        if msg.name == "predict_pcos":
            if isinstance(msg.content, str):
                try:
                    parsed_content = json.loads(msg.content)
                    if isinstance(parsed_content, dict):
                        raw_predict_pcos_tool_output = parsed_content
                    else:
                        print(f"Warning: predict_pcos tool returned a JSON string that is not a dictionary: {msg.content}")
                except json.JSONDecodeError:
                    print(f"Warning: predict_pcos tool returned a non-JSON string (likely an error message): {msg.content}")
            elif isinstance(msg.content, dict):
                raw_predict_pcos_tool_output = msg.content
            else:
                print(f"Warning: predict_pcos tool returned content of unexpected type ({type(msg.content)}): {msg.content}")
        elif msg.name == "explain_pcos_results":
            raw_explanation_text = msg.content
        elif msg.name == "recommend_next_steps":
            raw_recommendation_text = msg.content

# Prioritize features from tool output if successful, else use original input from state
features_for_logging = raw_predict_pcos_tool_output.get("features", final_state.get("features", user_data.model_dump()))

explanation = raw_explanation_text if raw_explanation_text else "No explanation provided."
recommendation = raw_recommendation_text if raw_recommendation_text else "No recommendation provided."


print(f"\n--- User Data Provided (as used by model) ---\n")
for key, value in features_for_logging.items(): 
    print(f"{key.replace('_', ' ').title()}: {value}")
print(f"\n-------------------------\n")


print(f"➡️ Model Prediction: **{prediction}** with probability **{round(probability * 100, 2)}%**\n")
print(f"📖 Explanation: {explanation}")
print(f"🩺 Recommendation: {recommendation}")
print("-" * 60)


try:
    save_to_db(
        user_input=features_for_logging, 
        prediction=prediction,
        probability=probability,
        explanation=explanation,
        recommendation=recommendation,
        user_id=user_id,
        thread_id=thread_id
    )
    print("Log: Prediction and explanation saved to database.")
except Exception as e:
    print(f"Error saving to database: {e}")