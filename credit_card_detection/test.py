import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_df = request.form['input_df']
        input_df_lst = input_df.split(',')
        features = np.array(input_df_lst, dtype=np.float64)
        prediction = model.predict(features.reshape(1, -1))
        if prediction[0] == 0:
            result = "Legitimate transaction"
        else:
            result = "Fraudulent transaction"
        return render_template('index.html', result=result)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
