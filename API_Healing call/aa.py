import joblib
from sentence_transformers import SentenceTransformer
import pandas as pd


#joblib.dump(grid, 'myproject_grid2_model_trained.joblib')



grid = joblib.load(r"C:\Users\User\Downloads\myproject_grid2_model_trained.joblib")
model_test = SentenceTransformer('all-MiniLM-L6-v2')


df=pd.read_csv(r"C:\Users\User\Downloads\new_dataframe_myproject_stemmed_and_labeled.csv")
df_1=pd.read_csv(r"C:\Users\User\OneDrive\Desktop\my_project\Symptom2Disease.csv")
label_count_dict = dict(zip(df["label"].value_counts().index, df_1["label"].value_counts().index))

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    # Embedding generation and prediction logic (ensure model_test and grid are defined)
    test_embeddings = model_test.encode(text)
    output = grid.predict(test_embeddings)
    predicted_label = output[0]
    print(f"Predicted Label: {label_count_dict[predicted_label]}")
    result = label_count_dict[predicted_label]

    return jsonify({'prediction': result})


def run_app():
    # Bind to localhost (127.0.0.1) and port 5001
    app.run(host='127.0.0.1', port=5000, threaded=True)


if __name__ == '__main__':
    run_app()
