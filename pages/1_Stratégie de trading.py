import streamlit as st
import streamlit.components.v1 as components
import pandas as pd 

# Wide layout 
st.set_page_config(layout="wide")

# Two columns for dropdowns
col1, col2 = st.columns(2)

# Create tabs for different stocks
tab1, tab2, tab3 = st.tabs(["AAPL", "GOOG", "MSFT"])

with col1:
    # Strategy dropdown
    strategy = st.selectbox("Select Strategy", ["Bollinger", 
                                                "Momentum1Day", 
                                                "Momentum5Days", 
                                                "RSI", 
                                                "TDI",
                                                "StochasticR",],
                                                index=3)
with col2:
    # Holding date dropdown
    holding_date = st.selectbox("Select Holding Date", ["1", "5"])

def display_html(TICKER, strategy, holding_date):
    try:
        with open(f"data/html_results/{TICKER}_{strategy}Strategy_holdbars{holding_date}.html", 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=700, scrolling=True)
    except FileNotFoundError:
        st.error(f"Error: {TICKER} backtest results file not found.")

def display_info(TICKER, strategy, holding_date):
    try:
        # read csv 
        stats_df = pd.read_csv(f"data/csv_results/{TICKER}_{strategy}Strategy_holdbars{holding_date}_stats.csv", index_col=0)

        start = stats_df.loc["Start"].iloc[0][:-9]
        end = stats_df.loc["End"].iloc[0][:-9]
        duration = stats_df.loc["Duration"].iloc[0][:-9]

        return_info = round(float(stats_df.loc["Return [%]"].iloc[0]), 2)
        return_ann = round(float(stats_df.loc["Return (Ann.) [%]"].iloc[0]), 2)
        volatility_ann = round(float(stats_df.loc["Volatility (Ann.) [%]"].iloc[0]), 2)
        sharpe_ratio = round(float(stats_df.loc["Sharpe Ratio"].iloc[0]), 2)
        sortino_ratio = round(float(stats_df.loc["Sortino Ratio"].iloc[0]), 2)

        max_drawdown = round(float(stats_df.loc["Max. Drawdown [%]"].iloc[0]), 2)
        trades = int(stats_df.loc["# Trades"].iloc[0])  # Si c'est un entier, on le garde comme tel
        win_rate = round(float(stats_df.loc["Win Rate [%]"].iloc[0]), 2)
        profit_factor = round(float(stats_df.loc["Profit Factor"].iloc[0]), 2)
        expectancy = round(float(stats_df.loc["Expectancy [%]"].iloc[0]), 2)

        st.info(
            f"""
            **Start :** {start} \n
            **End :** {end} \n
            **Duration :** {duration}
            """
        )
    
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            # return, return ann, volatility ann, sharpe ratio, sortino ratio
            st.metric(label="Return [%]", value=return_info, delta="-0.01%")
            st.metric(label="Return (Ann.) [%]", value=return_ann, delta="-0.01%")
            st.metric(label="Volatility (Ann.) [%]", value=volatility_ann, delta="0.01%")
            st.metric(label="Sharpe Ratio", value=sharpe_ratio, delta="0.01")
            st.metric(label="Sortino Ratio", value=sortino_ratio, delta="0.01")
        with subcol2:
            # max drawdown, trades, win rate, profit factor, expectancy
            st.metric(label="Max Drawdown [%]", value=max_drawdown, delta="-0.01%")
            st.metric(label="Trades", value=trades, delta="0")
            st.metric(label="Win Rate [%]", value=win_rate, delta="0.01%")
            st.metric(label="Profit Factor", value=profit_factor, delta="0.01")
            st.metric(label="Expectancy [%]", value=expectancy, delta="0.01%")

    except FileNotFoundError:
        st.error(f"Error: {TICKER} backtest results file not found.")

def display_trades(TICKER, strategy, holding_date):
    try:
        trades_df = pd.read_csv(f"data/csv_results/{TICKER}_{strategy}Strategy_holdbars{holding_date}_trades.csv", index_col=0)
        # drop SL, TP, Tag
        trades_df = trades_df.drop(columns=["SL", "TP", "Tag"])
        st.dataframe(trades_df, use_container_width=True)
    except FileNotFoundError:
        st.error(f"Error: {TICKER} trades file not found.")

def display_dashboard(TICKER, strategy, holding_date):
    col1, col2 = st.columns([4, 1])
    with col1:
        display_html(TICKER, strategy, holding_date)
    with col2:
        display_info(TICKER, strategy, holding_date)
    display_trades(TICKER, strategy, holding_date)

# GOOG tab content
with tab1:
    TICKER = "AAPL"
    display_dashboard(TICKER, strategy, holding_date)

with tab2:
    TICKER = "GOOGL"
    display_dashboard(TICKER, strategy, holding_date)

with tab3:
    TICKER = "MSFT"
    display_dashboard(TICKER, strategy, holding_date)

# Rajouter une legende
# Ecrire que on voit que pour hold 1 jour, habituellement entre 1 et 3 jours (probablement à cause jour ouvrés)
# Pour 5 jours de la meme manière la position peut rester ouverte juste qu'à 10 jours pcq jours fériés
# Statistique dashboard droite : calculé par rapport à la moyenne pour ce titre pour chaque stratégie et hold date
st.markdown(
    """
    ### Remarque 
    
    On peut noter que pour la stratégie de holding d'un jour, la plupart des trades sont fermés entre 1 et 3 jours.
    Pour la stratégie de holding de 5 jours, la plupart des trades sont fermés entre 5 et 10 jours.
    Ce qui est logique car il y a des jours fériés et des week-ends.

    Les statistiques sur le tableau de droite sont calculées par rapport à la moyenne pour ce titre pour chaque stratégie et date de holding.
    """
)

