from flask import Flask, render_template, request
import joblib


app = Flask(__name__)


model = joblib.load("sales_prediction\model.pkl")

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    tv_advertising_value = float(request.form["tv_advertising"])
    predicted = model.predict([[tv_advertising_value]])
    predicted_sales=round(predicted[0][0])
    print("Predicted Sales:",predicted_sales)
    return render_template("index.html",result=predicted_sales)
    
    
if __name__ == "__main__":
    app.run(debug=True)
    
    