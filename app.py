
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("disease_prediction_model.pkl")

# Feature names expected by the model
feature_names = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14', 'Symptom_15', 'Symptom_16', 'Symptom_17', 'symptom_count', 'symptom_count_std', 'symptom_count_minmax', 'Disease_encoded', ' redness_of_eyes', ' irregular_sugar_level', ' diarrhoea', ' back_pain', ' stomach_bleeding', ' bruising', ' swollen_blood_vessels', ' sweating', ' anxiety', ' altered_sensorium', ' red_sore_around_nose', ' swelled_lymph_nodes', ' history_of_alcohol_consumption', ' blurred_and_distorted_vision', ' swollen_extremeties', ' dehydration', ' sinus_pressure', ' passage_of_gases', ' extra_marital_contacts', ' weakness_in_limbs', ' restlessness', ' weight_loss', ' continuous_feel_of_urine', ' skin_rash', ' bloody_stool', ' dischromic _patches', ' weakness_of_one_body_side', ' unsteadiness', ' muscle_pain', ' headache', ' irritability', ' pain_during_bowel_movements', ' stomach_pain', ' knee_pain', ' vomiting', ' palpitations', ' abdominal_pain', ' swollen_legs', ' dark_urine', ' dizziness', ' slurred_speech', ' nausea', ' polyuria', ' chest_pain', ' continuous_sneezing', ' constipation', ' puffy_face_and_eyes', ' cold_hands_and_feets', ' drying_and_tingling_lips', ' visual_disturbances', ' spotting_ urination', ' fluid_overload', ' patches_in_throat', ' nodal_skin_eruptions', ' yellowish_skin', ' scurring', ' cramps', ' joint_pain', ' loss_of_smell', ' belly_pain', ' pain_behind_the_eyes', ' family_history', ' weight_gain', ' brittle_nails', ' malaise', ' neck_pain', ' blackheads', ' congestion', ' receiving_unsterile_injections', ' lack_of_concentration', ' toxic_look_(typhos)', ' pus_filled_pimples', ' mucoid_sputum', ' yellow_urine', ' swelling_joints', ' high_fever', ' increased_appetite', ' mood_swings', ' hip_joint_pain', ' foul_smell_of urine', ' obesity', ' silver_like_dusting', ' swelling_of_stomach', ' mild_fever', ' chills', ' watering_from_eyes', ' small_dents_in_nails', ' acute_liver_failure', ' stiff_neck', ' blister', ' fast_heart_rate', ' loss_of_balance', ' inflammatory_nails', ' rusty_sputum', ' yellowing_of_eyes', ' excessive_hunger', ' shivering', ' prominent_veins_on_calf', ' enlarged_thyroid', ' internal_itching', ' distention_of_abdomen', ' breathlessness', ' receiving_blood_transfusion', ' lethargy', ' acidity', ' fatigue', ' depression', ' muscle_wasting', ' abnormal_menstruation', ' throat_irritation', ' ulcers_on_tongue', ' skin_peeling', ' indigestion', ' burning_micturition', ' cough', ' coma', 'itching', ' red_spots_over_body', ' irritation_in_anus', ' blood_in_sputum', ' pain_in_anal_region', ' movement_stiffness', ' phlegm', ' muscle_weakness', ' sunken_eyes', ' yellow_crust_ooze', ' runny_nose', ' bladder_discomfort', ' loss_of_appetite', ' spinning_movements', ' painful_walking', 'performance_category', 'symptom_bin']

# Streamlit UI
st.title("Disease Prediction App")
st.write("Enter the patient details below:")

user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"Enter {feature}:", value=0.0)

# Predict button
if st.button("Predict Disease"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Disease: {prediction}")
