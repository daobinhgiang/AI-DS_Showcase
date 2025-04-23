import streamlit as st
from transformers import pipeline

# Load emotion classifier
@st.cache(allow_output_mutation=True)
def load_model():
    return pipeline("text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True)

model = load_model()

# UI
st.title("Emotion Classifier")
user_input = st.text_area("How are you feeling right now?")

if user_input:
    results = model(user_input)[0]
    st.write("## Emotion Scores:")
    for result in results:
        st.write(f"**{result['label']}**: {result['score']:.3f}")
