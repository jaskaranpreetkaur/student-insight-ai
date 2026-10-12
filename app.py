from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load the trained model from the project folder
MODEL_PATH = Path(__file__).parent / "student_performance_model (1).pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

FEATURES = [
    "sleep_hours",
    "screen_time_hours",
    "study_hours",
    "attendance_pct",
    "stress_level"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Please submit valid JSON data."}), 400

        # Read and validate all model inputs
        values = {
            "sleep_hours": float(data["sleep_hours"]),
            "screen_time_hours": float(data["screen_time_hours"]),
            "study_hours": float(data["study_hours"]),
            "attendance_pct": float(data["attendance_pct"]),
            "stress_level": float(data["stress_level"])
        }

        if not all(np.isfinite(v) for v in values.values()):
            return jsonify({"error": "Inputs must be finite numbers."}), 400

        limits = {
            "sleep_hours": (3, 11),
            "screen_time_hours": (0, 13),
            "study_hours": (0, 9),
            "attendance_pct": (40, 100),
            "stress_level": (1, 10)
        }

        for name, value in values.items():
            low, high = limits[name]
            if not low <= value <= high:
                return jsonify({
                    "error": f"{name} must be between {low} and {high}."
                }), 400

        # Preserve the exact feature names and order used in training
        input_df = pd.DataFrame([values], columns=FEATURES)

        score = float(model.predict(input_df)[0])
        score = float(np.clip(score, 0, 100))

        if score >= 80:
            category = "Strong predicted performance"
        elif score >= 60:
            category = "Moderate predicted performance"
        else:
            category = "Needs additional academic support"

        suggestions = []

        if values["sleep_hours"] < 7:
            suggestions.append("Try maintaining a consistent sleep routine.")
        if values["screen_time_hours"] > 6:
            suggestions.append("Review recreational screen-time habits.")
        if values["study_hours"] < 2:
            suggestions.append("Consider a regular, focused study routine.")
        if values["attendance_pct"] < 75:
            suggestions.append("Look for ways to improve class attendance.")
        if values["stress_level"] >= 8:
            suggestions.append("Consider healthy stress-management strategies.")

        if not suggestions:
            suggestions.append(
                "Continue monitoring your habits and maintain a balanced routine."
            )

        return jsonify({
            "predicted_score": round(score, 2),
            "category": category,
            "suggestions": suggestions
        })

    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Enter valid numeric values for every field."
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the model and server logs."
        }), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)