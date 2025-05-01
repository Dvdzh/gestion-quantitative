import streamlit as st
import streamlit.components.v1 as components
import pandas as pd 

# Wide layout 
st.set_page_config(layout="wide")

# # Two columns for dropdowns
# col1, col2 = st.columns(2)

# Create tabs for different stocks
tab1, tab2, tab3 = st.tabs(["AAPL", "GOOG", "MSFT"])

with st.sidebar:
    strategy = st.selectbox("Select Strategy", 
                            [
                                                "RSI",
                                                "Bollinger",
                                                "ChaikinMoneyFlow",
                                                "StochasticR",
                                                "TDI",
                                                "Momentum60Day",
                                                "VXN",
                                                "YieldCurve",
                                                "CreditSpread",
                                                "WilliamsR"
                                                ],
                                                index=0)
    holding_date = st.selectbox("Select Holding Date", ["1", "5"])

    if strategy == "RSI":
        with st.expander("Explications de la stratégie", expanded=True):
            st.write("Achete quand le RSI est inférieur à 40 et vend quand il est supérieur à 60.")
            st.write(f"Vend les positions aux bout de {holding_date} jours.")
    elif strategy == "Bollinger":
        with st.expander("Explications de la stratégie", expanded=True):
            st.write("Achete quand le prix est à l'intérieur des bandes de Bollinger (0 ≤ pctB ≤ 1)")
            st.write("Vend quand il est au-dessus de la bande supérieure (pctB > 1).")
            st.write(f"Vend les positions aux bout de {holding_date} jours.")
    elif strategy == "ChaikinMoneyFlow":
        with st.expander("Explications de la stratégie", expanded=True):
            st.write("Achete quand le CMF est supérieur à 0.20 et vend quand il est inférieur à -0.20.")
            st.write(f"Vend les positions aux bout de {holding_date} jours.")
    elif strategy == "StochasticR":
        with st.expander("Explications de la stratégie", expanded=True):
            st.write("Achete quand le %R est inférieur à 50 et vend quand il est supérieur à 90.")
            st.write(f"Vend les positions aux bout de {holding_date} jours.")
    elif strategy == "TDI":
        with st.expander("Explications de la stratégie", expanded=True):
            st.write("Achete quand le TDI est supérieur à 0 et le DI est supérieur à 0.")
            st.write("Vend quand le TDI est supérieur à 0 et le DI est inférieur à 0.")
            st.write(f"Vend les positions aux bout de {holding_date} jours.")
    elif strategy == "Momentum60Day":
        with st.expander("Explications de la stratégie", expanded=True):
            st.write("Achete quand le prix est supérieur à celui d'il y a 60 jours.")
            st.write(f"Vend les positions aux bout de {holding_date} jours.")
    elif strategy == "VXN":
        with st.expander("Explications de la stratégie", expanded=True):
            st.write("Achete quand le VXN est supérieur à 30.")
            st.write(f"Vend les positions aux bout de {holding_date} jours.")
    elif strategy == "YieldCurve":
        with st.expander("Explications de la stratégie", expanded=True):
            st.write("Achete quand le Yield Curve est supérieur à 0.20.")
            st.write(f"Vend les positions aux bout de {holding_date} jours.")
    elif strategy == "CreditSpread":
        with st.expander("Explications de la stratégie", expanded=True):
            st.write("Achete quand le Credit Spread est supérieur à 0.20.")
            st.write(f"Vend les positions aux bout de {holding_date} jours.")


def display_html(TICKER, strategy, holding_date):
    try:
        with open(f"data/strategies/html_results/{TICKER}_{strategy}Strategy_holdbars{holding_date}.html", 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=750, scrolling=True)
    except FileNotFoundError:
        st.error(f"Error: {TICKER} backtest results file not found.")

def display_info(TICKER, strategy, holding_date):
    try:
        # read csv 
        stats_df = pd.read_csv(f"data/strategies/csv_results/{TICKER}_{strategy}Strategy_holdbars{holding_date}_stats.csv", index_col=0)

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

        stats_mean_df = pd.read_csv(f"data/strategies/csv_results/{TICKER}_mean_holdbars{holding_date}.csv", index_col=0)
        return_info_mean = round(float(stats_mean_df.loc["Return [%]"].iloc[0]), 2)
        return_ann_mean = round(float(stats_mean_df.loc["Return (Ann.) [%]"].iloc[0]), 2)
        volatility_ann_mean = round(float(stats_mean_df.loc["Volatility (Ann.) [%]"].iloc[0]), 2)
        sharpe_ratio_mean = round(float(stats_mean_df.loc["Sharpe Ratio"].iloc[0]), 2)
        sortino_ratio_mean = round(float(stats_mean_df.loc["Sortino Ratio"].iloc[0]), 2)

        max_drawdown_mean = round(float(stats_mean_df.loc["Max. Drawdown [%]"].iloc[0]), 2)
        trades_mean = int(stats_mean_df.loc["# Trades"].iloc[0])  # Si c'est un entier, on le garde comme tel
        win_rate_mean = round(float(stats_mean_df.loc["Win Rate [%]"].iloc[0]), 2)
        profit_factor_mean = round(float(stats_mean_df.loc["Profit Factor"].iloc[0]), 2)
        expectancy_mean = round(float(stats_mean_df.loc["Expectancy [%]"].iloc[0]), 2)
    
        with st.expander("Détails des résultats", expanded=True):
            st.markdown(
                f"""
                **Start :** {start} \n
                **End :** {end} \n
                **Duration :** {duration}
                """
            )

        subcol1, subcol2 = st.columns(2)
        with subcol1:
            # return, return ann, volatility ann, sharpe ratio, sortino ratio
            st.metric(label="Return [%]", value=return_info, delta=round(float(return_info-return_info_mean), 2))
            st.metric(label="Return (Ann.) [%]", value=return_ann, delta=round(float(return_ann-return_ann_mean), 2))
            st.metric(label="Volatility (Ann.) [%]", value=volatility_ann, delta=round(float(volatility_ann-volatility_ann_mean), 2))
            st.metric(label="Sharpe Ratio", value=sharpe_ratio, delta=round(float(sharpe_ratio-sharpe_ratio_mean), 2))
            st.metric(label="Sortino Ratio", value=sortino_ratio, delta=round(float(sortino_ratio-sortino_ratio_mean), 2))
        with subcol2:
            # max drawdown, trades, win rate, profit factor, expectancy
            st.metric(label="Max Drawdown [%]", value=max_drawdown, delta=round(float(max_drawdown-max_drawdown_mean), 2))
            st.metric(label="Trades", value=trades, delta=trades - trades_mean)
            st.metric(label="Win Rate [%]", value=win_rate, delta=round(float(win_rate-win_rate_mean), 2))
            st.metric(label="Profit Factor", value=profit_factor, delta=round(float(profit_factor-profit_factor_mean), 2))
            st.metric(label="Expectancy [%]", value=expectancy, delta=round(float(expectancy-expectancy_mean), 2))

        
        with st.expander("Détails sur les deltas", expanded=False):
            st.markdown(
                f"""
                Deltas calculés par rapport à la moyenne pour ce titre pour chaque stratégie et date de holding. \n
                """
            )

    except FileNotFoundError:
        st.error(f"Error: {TICKER} backtest results file not found.")

def display_trades(TICKER, strategy, holding_date):
    try:
        trades_df = pd.read_csv(f"data/strategies/csv_results/{TICKER}_{strategy}Strategy_holdbars{holding_date}_trades.csv", index_col=0)
        # drop SL, TP, Tag
        trades_df = trades_df.drop(columns=["SL", "TP", "Tag"])
        st.dataframe(trades_df, use_container_width=True)
    except FileNotFoundError:
        st.error(f"Error: {TICKER} trades file not found.")

def display_dashboard(TICKER, strategy, holding_date):
    # Présentation du dashboard
    st.subheader(f"Dashboard {TICKER} - {strategy} - Holding {holding_date} days")
    st.markdown(
        """
        Ce dashboard présente les résultats du backtest pour la stratégie de trading sélectionnée.
        Il inclut un graphique interactif, des statistiques clés et un tableau des trades.
        """
    )
    col1, col2 = st.columns([4, 1])
    with col1:
        display_html(TICKER, strategy, holding_date)
    with col2:
        display_info(TICKER, strategy, holding_date)
    # Présentation du tableau des trades
    st.subheader("Tableau des trades")
    st.markdown(
        """
        Ce tableau présente les trades réalisés pendant la période de backtest.
        Il inclut des informations sur le prix d'entrée, le prix de sortie, le profit réalisé, etc.
        """
    )
    display_trades(TICKER, strategy, holding_date)

# GOOG tab content
with tab1:
    TICKER = "AAPL"
    display_dashboard(TICKER, strategy, holding_date)

    # Recapitulatif pour 

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
    ### Remarques 
    - On peut noter que pour la stratégie de holding d'un jour, la plupart des trades sont fermés entre 1 et 3 jours.
    Pour la stratégie de holding de 5 jours, la plupart des trades sont fermés entre 5 et 10 jours.
    Ce qui est logique car il y a des jours fériés et des week-ends.
    - On a implementé la stratégie de **WilliamsR** de la manière suivante : 
    Achete quand le %R est inférieur à -20 et vend quand il est inférieur à -80.

    """
)

