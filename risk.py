from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
dataset = pd.read_csv("dataset.csv")

with open("stroke_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def index():
    return render_template("index.html",
        genders=sorted(dataset["gender"].unique()),
        ages=sorted(dataset["age"].unique()),
        hypertensions=sorted(dataset["hypertension"].unique()),
        heart_diseases=sorted(dataset["heart_disease"].unique()),
        ever_marrieds=sorted(dataset["ever_married"].unique()),
        work_types=sorted(dataset["work_type"].unique()),
        Residence_types=sorted(dataset["Residence_type"].unique()),
        avg_glucose_levels=sorted(dataset["avg_glucose_level"].unique()),
        smoking_statuss=sorted(dataset["smoking_status"].unique())
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "gender": request.form["gender"],
        "age": float(request.form["age"]),
        "hypertension": int(request.form["hypertension"]),
        "heart_disease": int(request.form["heart_disease"]),
        "ever_married": request.form["ever_married"],
        "work_type": request.form["work_type"],
        "Residence_type": request.form["Residence_type"],
        "avg_glucose_level": float(request.form["avg_glucose_level"]),
        "smoking_status": request.form["smoking_status"]
    }

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    result = "Stroke Risk" if prediction == 1 else "No Stroke Risk"
    return str(result)


if __name__ == "__main__":
    app.run(debug=True, port=5001)



