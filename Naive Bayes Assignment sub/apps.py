from flask import Flask, render_template, request
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, confusion_matrix

# creating Engine which connect to MySQL
user = 'root'
pw = '98486816'
db = 'Clustering'
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Load the saved model
model = joblib.load("processed1")

# Define Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        data = pd.read_csv(f)
        
        # Data preprocessing
        data.drop(['maritalstatus','native','workclass','race'], axis=1, inplace=True)
        data = pd.get_dummies(data, columns=['education','sex','relationship','occupation'])
        data['Salary_numerical'] = data['Salary'].map({' <=50K': 0, ' >50K': 1})
        x1 = data.drop(['Salary','Salary_numerical'], axis=1)

        # Model prediction
        test_pred_nb = pd.DataFrame(model.predict(x1), columns=["Salary_pred"])
        
        # Evaluation metrics
        confusion_matrix_nb = confusion_matrix(data['Salary_numerical'], test_pred_nb)
        accuracy_nb = accuracy_score(data['Salary_numerical'], test_pred_nb)

        # Combine predictions with original data
        final = pd.concat([data, test_pred_nb], axis=1)

        # Save to MySQL database
        final.to_sql("Salary_Naive_Bayes", con=engine, if_exists='replace', index=False)
        
        return render_template("new.html", Y=final.to_html(justify='center'), 
                               confusion_matrix=confusion_matrix_nb, accuracy=accuracy_nb)

if __name__ == '__main__':
    app.run(debug=True)
