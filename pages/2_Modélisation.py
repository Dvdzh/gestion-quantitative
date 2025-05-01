import streamlit as st
import streamlit.components.v1 as components
import pandas as pd 

import onnxruntime as ort
import numpy as np
import json 

import plotly.figure_factory as ff

def predict_onnx(model_path, X_test):
    # Load the ONNX model
    onnx_session = ort.InferenceSession(model_path)

    # Prepare input data for inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    # Perform inference
    predictions = onnx_session.run([output_name], {input_name: X_test})

    return predictions[0]

# Wide layout 
st.set_page_config(layout="wide")

with st.sidebar:
    # Présentation pour chaque modèle des paramètres utilisés pour l'optimisation
    # Avec un bouton pour afficher/masquer les détails  
    st.markdown("### Optimisation des hyperparamètres")
    st.write("Régression logistique")
    # Bouton dérouler pour plus d'informations 
    with st.expander("Détails", expanded=False):
        st.write("""
            - **Pénalité** : L2
            - **C** : 0.1, 1, 10
            - **Solveur** : lbfgs, liblinear
            - **Max itérations** : 100, 500
            """
        )
    st.write("Arbre de décision")
    with st.expander("Détails", expanded=False):
        st.write(
            """
            - **Profondeur maximale** : 3, 5, 8
            - **Nombre minimum d'échantillons par feuille** : 1, 3
            - **Critère** : gini, entropy
            """
        )
    st.write("Forêt aléatoire")
    with st.expander("Détails", expanded=False):
        st.write(
            """
            - **Nombre d'estimateurs** : 200, 500
            - **Profondeur maximale** : Default, 10
            - **Nombre minimum d'échantillons par feuille** : 1, 3
            - **Bootstrap** : True, False
            """
        )
    st.write("Gradient boosting")
    with st.expander("Détails", expanded=False):
        st.write(
            """
            - **Nombre d'estimateurs** : 100, 300, 500
            - **Taux d'apprentissage** : 0.05, 0.1, 0.2
            - **Profondeur maximale** : 2, 3, 4
            """
        )

def display_tab(method, TICKER): 
    
    # Model name 
    if method == "Logistic":
        st.latex(r"P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \beta_4 X_4)}}")
    elif method == "DecisionTree":
        st.latex(r"Y = \begin{cases} 1 & \text{if } X_1 \leq t_1 \\ 0 & \text{otherwise} \end{cases}")
    elif method == "RandomForest":  
        st.latex(r"Y = \frac{1}{N} \sum_{i=1}^{N} Y_i")
    elif method == "GradientBoosting":
        st.latex(r"Y = \sum_{m=1}^{M} \alpha_m h_m(X) + \beta")

    # 2 - Show confusion matrix
    with open(f"data/modelisation/results/{TICKER}_{method}_results.json", 'r') as f:
        data = json.load(f)
        confusion_matrix = data["confusion_matrix"]
        accuracy = float(data["accuracy"])
        precision = float(data["precision"])
        recall = float(data["recall"])
        f1_score = float(data["f1_score"])
        

        # Plot confusion matrix using Plotly
        z = confusion_matrix
        x = ["Predicted Negative", "Predicted Positive"]
        y = ["Actual Negative", "Actual Positive"]
        fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale="Viridis", showscale=True)

        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Accuracy", f"{accuracy:.2%}")
            st.metric("Precision", f"{precision:.2%}")
        with col3:
            st.metric("Recall", f"{recall:.2%}")
            st.metric("F1 Score", f"{f1_score:.2%}")

    # # 3 - Show the model
    # # Add sliders for user input
    # col1, col2, col3, col4 = st.columns(4)
    # with col1:
    #     ret1 = st.slider("Return 1D", min_value=-0.1, max_value=0.1, value=0.01880945, step=0.0001, key=f"{TICKER}-ret1d")
    # with col2:
    #     ret5 = st.slider("Return 5D", min_value=-0.1, max_value=0.1, value=0.01880945, step=0.0001, key=f"{TICKER}-ret5d")
    # with col3:
    #     sma5 = st.slider("SMA 5D", min_value=0.0, max_value=0.1, value=0.01880945, step=0.0001, key=f"{TICKER}-sma5d")
    # with col4:
    #     vol5 = st.slider("Volatility 5D", min_value=0.0, max_value=0.1, value=0.01880945, step=0.0001, key=f"{TICKER}-vol5d")

    # # Prepare input array
    # X_test = np.array([[ret1, ret5, sma5, vol5]], dtype=np.float32)

    # # Perform prediction
    # predict = predict_onnx(f"models/{method}.onnx", X_test)
    # st.write(f"Prediction: {predict}")
    # # predict = predict_onnx(f"models/{method}.onnx", X_test)
    # # st.write(f"Prediction : {predict}")


    # 1 - Show the dataframe 
    prepared_df = pd.read_csv(f"data/modelisation/prepared_data/{TICKER}_prepared.csv", index_col=0)
    # changer nom par le dictionnaire
    name_dict = dict(
        target="Target",
        ret1="Return 1 Day",
        ret3="Return 3 Days",
        sma5="SMA 5 Days",
        vol5="Volatility 5 Days",
    )
    prepared_df.rename(columns=name_dict, inplace=True)
    st.dataframe(prepared_df, use_container_width=True)  # streamlit pandas dataframe
    
def display_content(TICKER):

    # Présentation des données d'entraînement
    st.subheader("Aperçu des données d'entraînement")
    st.write(
        f"Le modèle a été entraîné sur les données de l'action {TICKER} (80% du dataset). "
        "Les données comprennent les rendements sur 1 jour, 3 jours, la moyenne mobile sur 5 jours et la volatilité sur 5 jours. "
        "Le modèle prédit si le prix de l'action augmentera (1) ou non (0)."
    )
    # Charger le tableau des données préparées
    df = pd.read_csv(f"data/modelisation/prepared_data/{TICKER}_prepared.csv")
    # Changer la couleur de la colonne target en rouge 
    df = df.style.apply(lambda x: ['background: red' if x.name == 'target' else '' for i in x], axis=1)
    st.dataframe(df, use_container_width=True)  # streamlit pandas dataframe

    # Présentation des résultats
    st.subheader("Aperçu des résultats")
    st.write(
        "Le modèle a été évalué sur un ensemble de test (20% du dataset). "
        "La matrice de confusion, l'exactitude, la précision, le rappel et le score F1 sont affichés ci-dessous."
    )
    # Charger les résultats 
    # Charger le tableau des résultats
    df = pd.read_csv("data/modelisation/results/results.csv")
    df = df[df["ticker"] == TICKER]
    st.dataframe(df, use_container_width=True)  # streamlit pandas dataframe

    # Afficher les 4 matrices de confusion 
    figs = {}
    for method in ["Logistic", "DecisionTree", "RandomForest", "GradientBoosting"]:
        with open(f"data/modelisation/results/{TICKER}_{method}_results.json", 'r') as f:
            data_temp = json.load(f)
            z = data_temp["confusion_matrix"]
            x = ["Predicted Negative", "Predicted Positive"]
            y = ["Actual Negative", "Actual Positive"]
            fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale="Viridis", showscale=True)
            figs[method] = fig
            # fig tight layout
            fig.update_layout(
                title=f"Matrice de confusion pour le modèle {method}",
                xaxis_title="Prédiction",
                yaxis_title="Vérité terrain",
                width=500,
                height=400,
                # Centrer le titre
                title_x=0.25,
                title_y=0.95,
                # Ajuster les bords 
                margin=dict(l=20, r=20, t=100, b=20),
            )
    col1, col2 = st.columns(2)
    with col1:
        # st.legend("Matrice de confusion")
        st.plotly_chart(figs["Logistic"], use_container_width=True)
        st.plotly_chart(figs["DecisionTree"], use_container_width=True)
    with col2:
        st.plotly_chart(figs["RandomForest"], use_container_width=True)
        st.plotly_chart(figs["GradientBoosting"], use_container_width=True)
    
    # Constat : 
    # - une très bon recall : proche de 0.9, en effet on a presque toutes les jours de prédiction où le prix monte
    # - mais un très mauvais precision : proche de 0.5, en effet on a beaucoup de faux positifs, la moitié ne sont pas des jours où le prix monte
    # Pareil pour les trois modèles, on le modèle prédit souvent que le prix va monter, mais il se trompe souvent
    # On pourrait faire une étude de l'influences des hyperparamètres pour identifier la source du problème

    st.subheader("Analyse des résultats")
    st.write(
        "L'analyse des résultats montre que le modèle a une bonne capacité à prédire les jours où le prix de l'action augmente, "
        "mais il a également un taux élevé de faux positifs. "
        "Cela signifie que le modèle prédit souvent que le prix va augmenter, mais il se trompe fréquemment. "
        "Cela peut être dû à un déséquilibre dans les données d'entraînement, où il y a plus de jours où le prix augmente que de jours où il diminue. "
    )

    st.subheader("Optimisation des hyperparamètres")
    st.write(
        "L'optimisation des hyperparamètres a été effectuée pour tenter d'améliorer les performances du modèle. "
        "La configuration d'optimisation a été choisie et noté dans la side bar. "
        "Les résultats de l'optimisation sont affichés ci-dessous, et on été réalisé pour maximiser l'accuracy."
    )

    
tab1, tab2, tab3 = st.tabs(["AAPL", "GOOG", "MSFT"])
with tab1:
    TICKER = "AAPL"
    display_content(TICKER)
with tab2:
    TICKER = "GOOGL"
    display_content(TICKER)
with tab3:
    TICKER = "MSFT"
    display_content(TICKER)




