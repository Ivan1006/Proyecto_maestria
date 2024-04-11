import streamlit as st
import pandas as pd
from models.models import load_model, MODELS
from text_processing.text_processing import preprocess_text
from sentiment_analysis.setiment_analysis import analyze_sentiment, plot_sentiment_distribution

def main():
    st.set_page_config(page_title="Detector de Sentimientos en Texto", page_icon=":speech_balloon:")
    st.title("Detector de Sentimientos en Texto")

    model_name = st.sidebar.selectbox("Seleccionar modelo de análisis de sentimientos", list(MODELS.keys()))

    temperature = st.sidebar.slider("Temperatura", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

    option = st.sidebar.selectbox("Elegir opción", ["Analizar un texto", "Cargar un archivo"])

    if "tokenizer" not in st.session_state or "model" not in st.session_state:
        tokenizer, model = load_model(MODELS[model_name])
        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
    else:
        tokenizer = st.session_state.tokenizer
        model = st.session_state.model

    if option == "Analizar un texto":
        text = st.text_area("Ingrese el texto que desea analizar")

        if st.button("Analizar Sentimiento - Texto", key=None) and text:
            text = preprocess_text(text)
            sentiment_class, probabilities = analyze_sentiment(text, tokenizer, model, temperature=temperature)
            sentiment_label = ["Negativo", "Neutro", "Positivo"][sentiment_class]
            st.write("Sentimiento detectado:", sentiment_label)
            st.write("Probabilidades:", {label: prob for label, prob in zip(["Negativo", "Neutro", "Positivo"], probabilities)})
            plot_sentiment_distribution([sentiment_class])
        elif not text:
            st.warning("Por favor ingrese un texto para analizar.")

    elif option == "Cargar un archivo":
        uploaded_file = st.file_uploader("Cargar archivo", type=["xlsx", "csv"])

        if uploaded_file is not None:
            if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)

            # Obtener el nombre de la primera columna del DataFrame
            text_column = df.columns[0]

            # Aplicar preprocesamiento de texto
            df[text_column] = df[text_column].apply(preprocess_text)

            st.write(df)
            sentiments = []
            for text in df[text_column]:
                sentiment_class, _ = analyze_sentiment(text, tokenizer, model, temperature=temperature)
                sentiments.append(sentiment_class)

            st.write("Distribución de Sentimientos:")
            plot_sentiment_distribution(sentiments)

if __name__ == "__main__":
    main()
