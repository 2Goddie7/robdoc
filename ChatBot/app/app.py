import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import random

@st.cache_resource(show_spinner=True)
def load_model():
    with open('ChatBot/modelo/model.pkl', 'rb') as f:
        model = pickle.load(f)
    embed_model = SentenceTransformer(model['embedding_model_name'])
    return model, embed_model

try:
    model, embed_model = load_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

st.title("¿Tienes preguntas?")
st.subheader("RobDoc te ayudará!")

option = st.radio("Selecciona una categoría:", 
                  ["Contactos", "Horarios", "Preguntas Frecuentes", "+Especialidades"])

if option == "Contactos":
    st.markdown("""
    **Contactos:**
    
    Puedes contactarnos al número **1800-MediCom** o mediante nuestras redes sociales (Facebook, Instagram, TikTok, etc).
    """)

elif option == "Horarios":
    st.markdown("""
    **Horarios de atención:**
    
    Nuestros horarios de atención son de **Lunes a Sábado de 7:00 am a 18:00 pm**  
    y los días **Domingos de 9:00 am a 14:00 pm**
    """)

elif option == "+Especialidades":
    st.markdown("""
    **Próximas especialidades:**
    
    Estamos trabajando para poder incorporar las especialidades:  
    Optometría - Audiología - Ginecología - Odontología
    """)

elif option == "Preguntas Frecuentes":
    st.markdown("**Preguntas frecuentes:**")
    
    if "faq_indices" not in st.session_state:
        if len(model['questions']) > 4:
            st.session_state.faq_indices = random.sample(range(len(model['questions'])), 4)
        else:
            st.session_state.faq_indices = list(range(len(model['questions'])))
        st.session_state.selected_faq = None

    for idx in st.session_state.faq_indices:
        if st.button(model['questions'][idx], key=f"faq_btn_{idx}"):
            st.session_state.selected_faq = idx

    if st.session_state.selected_faq is not None:
        st.info(model['answers'][st.session_state.selected_faq])

    st.markdown("---")
    st.markdown("**¿No encuentras tu pregunta? Escríbela aquí:**")
    user_question = st.text_input("Tu pregunta:")

    if user_question:
        user_vec = embed_model.encode([user_question])
        similarities = cosine_similarity(user_vec, model['matrix']).flatten()
        
        threshold = 0.7  # o usar un slider para ajustar dinámicamente
        max_sim = np.max(similarities)
        max_idx = np.argmax(similarities)

        if max_sim >= threshold:
            st.success(f"Respuesta: {model['answers'][max_idx]}")
        else:
            st.warning("Lo siento, no encontré una respuesta adecuada. Intenta reformular tu pregunta o consulta una de las preguntas frecuentes.")
