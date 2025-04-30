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

# Two columns for dropdowns
col1, col2 = st.columns(2)

# # Create tabs for different stocks
# tab1, tab2, tab3 = st.tabs(["AAPL", "GOOG", "MSFT"])


# # dropdown method 
# method = st.selectbox("Select Method", ["DecisionTree", 
#                                         "GradientBoosting",
#                                         "RandomForest",
#                                         "Logistic"], 
#                                     key=f"{TICKER}_method")

tab1, tab2, tab3, tab4 = st.tabs(["Decision Tree", "Gradient Boosting", "Random Forest", "Logistic Regression"])

def display_tab(method, TICKER): 
    
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

TICKER = "AAPL"
with tab1:
    method = "Logistic"
    # write logistic regression model 
    st.latex(r"P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \beta_4 X_4)}}")
    display_tab(method, TICKER)
with tab2:
    # streamlit add svg 
    method = "DecisionTree"
    st.image("data/modelisation/images/output_19_0.svg")
    display_tab(method, TICKER)
with tab3:
    method = "RandomForest"
    st.image("data/modelisation/images/output_19_0.svg")
    display_tab(method, TICKER)
with tab4:
    method = "GradientBoosting"
    display_tab(method, TICKER)

