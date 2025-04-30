"""
walk_forward_example.py
-----------------------
Télécharge les prix (yfinance), crée un label binaire "UP +5 jours",
entraîne 4 modèles SK-Learn et affiche la matrice de confusion sur
un split chronologique 80 % / 20 %.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ------------------------------------------------------------------
# 1. Paramètres généraux
# ------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "GOOGL"]
LOOK_AHEAD = 5          # horizon de prédiction (jours)
TRAIN_RATIO = 0.80      # fraction chronologique pour le train-set
SEED = 42

# ------------------------------------------------------------------
# 2. Fonction utilitaire : features + label
# ------------------------------------------------------------------
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Label : prix dans 5 jours supérieur au prix courant ?
    df["future_close"] = df["Close"].shift(-LOOK_AHEAD)
    df["target"] = (df["future_close"] > df["Close"]).astype(int)

    # Features (exemples simples, extensibles) ────────────────
    df["ret1"]  = np.log(df["Close"] / df["Close"].shift(1))
    df["ret3"]  = np.log(df["Close"] / df["Close"].shift(3))
    df["sma5"]  = df["Close"].rolling(5).mean() / df["Close"] - 1
    df["vol5"]  = df["ret1"].rolling(5).std()

    df = df.dropna().reset_index(drop=True)  # retirer NaN
    return df[["Date", "target", "ret1", "ret3", "sma5", "vol5"]]

# ------------------------------------------------------------------
# 3. Modèles à tester
#    (la logistique bénéficie d’une standardisation automatique)
# ------------------------------------------------------------------
MODELS = {
    "Logistic": make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1000, random_state=SEED)),
    "DecisionTree": DecisionTreeClassifier(random_state=SEED),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
}

import os 
prepared_data_path = os.path.dirname(os.getcwd()) + "/data/modelisation/prepared_data"
os.makedirs(prepared_data_path, exist_ok=True)

# ------------------------------------------------------------------
# 4. Boucle principale par ticker
# ------------------------------------------------------------------
for tic in TICKERS:
    print(f"\n========================  {tic}  ========================\n")
    # 4-a. Télécharger et préparer
    price = yf.download(tic, start="2015-01-01", progress=False)
    price = price.reset_index()                # yfinance retourne DatetimeIndex
    price = price.droplevel(1, axis=1)  # supprimer le multi-index
    data  = prepare_data(price)

    # Save data to CSV for later use

    data.to_csv(f"{prepared_data_path}/{tic}_prepared.csv", index=False)
    X = data[["ret1", "ret3", "sma5", "vol5"]]
    y = data["target"].values

    # 4-b. Split chronologique
    cut = int(len(data) * TRAIN_RATIO)
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y[:cut], y[cut:]

    # 4-c. Entraînement + évaluation pour chaque modèle
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        acc  = (tp + tn) / cm.sum()
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec  = tp / (tp + fn) if (tp + fn) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        print(f"--- {name} ---")
        print("Matrice de confusion\n", cm)
        print(f"Accuracy  : {acc:.3f}")
        print(f"Precision : {prec:.3f} | Recall : {rec:.3f} | F1 : {f1:.3f}\n")

# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType
# import os

# # Répertoire pour sauvegarder les modèles ONNX
# onnx_dir = os.path.dirname(os.getcwd()) + "/onnx_models"
# onnx_dir = "onnx_models"
# os.makedirs(onnx_dir, exist_ok=True)

# # Entraîner et sauvegarder chaque modèle
# for name, model in MODELS.items():
#     print(f"Training and saving model: {name}")
#     model.fit(X_train, y_train)  # Entraînement du modèle

#     # Conversion du modèle en ONNX
#     initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
#     onnx_model = convert_sklearn(model, initial_types=initial_type)

#     # Sauvegarde du modèle ONNX
#     onnx_path = os.path.join(onnx_dir, f"{name}.onnx")
#     with open(onnx_path, "wb") as f:
#         f.write(onnx_model.SerializeToString())

#     print(f"Model {name} saved to {onnx_path}")


# import onnxruntime as ort
# import numpy as np

# # Load the ONNX model
# onnx_path = onnx_dir + "/Logistic.onnx"  # Change to the desired model
# onnx_session = ort.InferenceSession(onnx_path)

# # Prepare input data for inference
# # Assuming `X_test` is the input data for inference
# input_name = onnx_session.get_inputs()[0].name
# output_name = onnx_session.get_outputs()[0].name

# # Perform inference
# predictions = onnx_session.run([output_name], {input_name: X_test.to_numpy().astype(np.float32)})

# # Display predictions
# print("Predictions:", predictions[0])