import streamlit as st

st.set_page_config(
    page_title="Gestion Quantitative",
    page_icon="📈",
)

st.title("Bienvenue dans le Projet de Gestion Quantitative")
st.sidebar.success("Naviguez entre les pages ci-dessus.")

st.markdown(
    """
    ## À propos du projet
    Ce projet vise à fournir des outils interactifs pour l'analyse et la gestion quantitative. 
    Vous pouvez explorer différentes fonctionnalités via les pages disponibles dans la barre latérale.

    ### Fonctionnalités principales :
    - Analyse de stratégie de trading 📊
    - Classification de série temporelle 📈

    ### Comment commencer ?
    Utilisez la barre latérale pour naviguer entre les différentes pages et découvrir les outils disponibles.

    ---

    ### Cadre de travail
    IND8123 Technologies Financières pour ingénieurs, Polytechnique Montréal \n

    
    """
)