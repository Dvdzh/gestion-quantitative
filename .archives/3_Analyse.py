import streamlit as st

# Wide layout 
st.set_page_config(layout="wide")

st.subheader("Partie 1 - Stratégies de trading")
st.write("""
    Dans cette partie, on va résumé notre approche pour les stratégies de trading. \n
    Nous avons utilisé un backtester backtrading pour évaluer les performances de nos stratégies de trading. \n
    Nous avons implémenté les différentes stratégies de trading proposées dans le devoir. \n
    Nous avons utilisé un backtester backtrading pour évaluer les performances de nos stratégies de trading. \n
    Et nous avons plot tout cela dans un dashboard streamlit. \n
""")

st.subheader("Partie 2 - Modélisation")
st.write("""
    Dans cette partie, on va résumé notre approche de modélisation. \n
    La pipeline était la suivante : \n
    1. On a téléchargé les données et on les a séparées en ensembles d'entraînement et de test. \n
    2. On a entraîné quatre modèles de machine learning sur l'ensemble d'entraînement : régression logistique, arbre de décision, forêt aléatoire et gradient boosting. \n
    3. On a évalué les modèles sur l'ensemble de test. \n
    4. On a optimisé les hyperparamètres pour chaque modèle. \n
    5. On a évalué les modèles sur l'ensemble de test. \n
""")
st.write("""
    Analyse des résultats : \n
    - On a observé un overfitting sur l'ensemble d'entraînement. \n
    - On a constaté une tendance à prédire la classe 1 pour chaque modèle. \n
""")
st.write("""
    Analyse par rapport aux stratégies de trading : \n
    - On a obtenu des résultats différents avec le trading. \n
    - On aurait pu inclure la prédiction du prix comme signal tous les jours. \n
    - Cependant, on peut s'attendre à des résultats pas terribles étant donné que les données sur l'ensemble de test ont une précision de 0.5. \n
""")
