# import joblib
# import numpy as np
# import pandas as pd
# import gradio as gr

# # 1. Load the trained Random Forest model
# MODEL_PATH = "saved_model/random_forest.joblib"
# model = joblib.load(MODEL_PATH)

# # 2. Load the training data to get the symptom columns (feature names)
# DATA_PATH = "dataset/training_data.csv"
# df_train = pd.read_csv(DATA_PATH)

# # Last column is the target (prognosis), all others are symptoms/features
# symptom_columns = df_train.columns[:-1]  # all columns except last

# # 3. Prediction function for Gradio
# def predict_disease(selected_symptoms):
#     """
#     selected_symptoms: list of symptom names chosen from the checkbox group.
#     We build a 0/1 vector for all symptoms and run the model.
#     """
#     # Start with all zeros
#     x = np.zeros(len(symptom_columns), dtype=int)

#     # Set 1 for the symptoms that user selected
#     col_index = {name: i for i, name in enumerate(symptom_columns)}
#     if selected_symptoms:
#         for s in selected_symptoms:
#             if s in col_index:
#                 x[col_index[s]] = 1

#     # Model expects 2D array
#     x_2d = x.reshape(1, -1)

#     # Predict
#     prediction = model.predict(x_2d)[0]
#     return str(prediction)

# # 4. Build Gradio interface
# description_text = """
# # Disease Prediction from Symptoms

# **Note:** This is a demo project for learning / academic purposes only.  
# It must *not* be used for real medical diagnosis. Always consult a doctor.
# """

# with gr.Blocks() as demo:
#     gr.Markdown(description_text)

#     gr.Markdown("### Select your symptoms:")

#     symptoms_input = gr.CheckboxGroup(
#         choices=list(symptom_columns),
#         label="Symptoms"
#     )

#     predict_button = gr.Button("Predict Disease")
#     output_text = gr.Textbox(label="Predicted Disease")

#     predict_button.click(
#         fn=predict_disease,
#         inputs=symptoms_input,
#         outputs=output_text
#     )

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
MODEL_PATH = "saved_model/random_forest.joblib"
model = joblib.load(MODEL_PATH)

# Load dataset
DATA_PATH = "dataset/training_data.csv"
df_train = pd.read_csv(DATA_PATH)

symptom_columns = df_train.columns[:-1]

st.title("Disease Prediction from Symptoms")
st.write("⚠️ For educational use only")

selected_symptoms = st.multiselect(
    "Select your symptoms:",
    symptom_columns
)

if st.button("Predict Disease"):
    x = np.zeros(len(symptom_columns))
    for s in selected_symptoms:
        if s in symptom_columns:
            x[list(symptom_columns).index(s)] = 1

    prediction = model.predict([x])[0]
    st.success(f"Predicted Disease: {prediction}")


# # 5. Launch the app
# if __name__ == "__main__":
#     demo.launch()
