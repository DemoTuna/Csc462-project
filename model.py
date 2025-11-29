import joblib
import numpy as np
import pandas as pd

def predict(test_csv_path, output_csv_path):
    model = joblib.load("weights.joblib")
    test = pd.read_csv(test_csv_path)
    feature_cols = [f"p{i}" for i in range(1, 43)] + ["turn"]
    X_test = test[feature_cols].values.astype(np.float32)
    proba = model.predict_proba(X_test)
    preds = np.argmax(proba, axis=1)
    pd.DataFrame({"id": test["id"], "label_move_col": preds}).to_csv(output_csv_path, index=False)
