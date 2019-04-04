from flask import Flask,redirect,url_for,request,render_template
import pandas as pd
import sklearn
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
app = Flask(__name__)


@app.route('/success/<int:result>')
def success(result):
    if result==0:
            return render_template(r"indexi.html")
    else:
            return render_template(r"indexj.html")
@app.route('/login',methods=['POST','GET'])
def login():
        age=int(request.form['age'])
        sex=int(request.form['sex'])
        chest_pain_type=int(request.form['chest_pain_type'])
        resting_blood_pressure=int(request.form['resting_blood_pressure'])
        serum_cholestrol=int(request.form['serum_cholestrol'])
        fasting_blood_sugar=int(request.form['fasting_blood_sugar'])
        resting_ecg=int(request.form['resting_ecg'])
        heart_rate=int(request.form['heart_rate'])
        exercise_induced_angina=int(request.form['exercise_induced_angina'])
        Depression=int(request.form['depression'])
        heartDisease_train=pd.read_csv(r"C:\Users\aman\Desktop\ml project\tubes2_HeartDisease_train.csv")
        heartDisease_train['Column4'] = pd.to_numeric(heartDisease_train['Column4'], errors='coerce')
        heartDisease_train['Column5'] = pd.to_numeric(heartDisease_train['Column5'], errors='coerce')
        heartDisease_train['Column6'] = pd.to_numeric(heartDisease_train['Column6'], errors='coerce')
        heartDisease_train['Column7'] = pd.to_numeric(heartDisease_train['Column7'], errors='coerce')
        heartDisease_train['Column8'] = pd.to_numeric(heartDisease_train['Column8'], errors='coerce')
        heartDisease_train['Column9'] = pd.to_numeric(heartDisease_train['Column9'], errors='coerce')
        heartDisease_train['Column10'] = pd.to_numeric(heartDisease_train['Column10'], errors='coerce')
        heartDisease_train['Column11'] = pd.to_numeric(heartDisease_train['Column11'], errors='coerce')
        heartDisease_train['Column12'] = pd.to_numeric(heartDisease_train['Column12'], errors='coerce')
        heartDisease_train['Column13'] = pd.to_numeric(heartDisease_train['Column13'], errors='coerce')
        heartDisease_train.rename(columns={'Column1': 'age', 'Column2': 'sex', 'Column3': 'chest_pain_type', 'Column4': 'resting_bp',
                         'Column5': 'ser_chol', 'Column6': 'fast_glucose', 'Column7': 'rest_ecg',
                         'Column8': 'heart_rate', 'Column9': 'exc_angina', 'Column10': 'depression',
                         'Column11': 'peak_exc', 'Column12': 'maj_vessels', 'Column13': 'thal',
                         'Column14': 'heart_disease'}, inplace=True)
        heartDisease_train.drop(columns=['peak_exc', 'maj_vessels', 'thal'], inplace=True)
        heartDisease_train.dropna(inplace=True)
        heartDisease_train['heart_disease'] = (heartDisease_train['heart_disease'] >= 1).astype(int)
        heartDisease_train = heartDisease_train.astype(int)
        x1 = heartDisease_train.iloc[:, :-1]
        y1 = heartDisease_train.iloc[:, -1]
        x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.25, random_state=3)
        classifier = SVC(kernel='linear', random_state=3)
        classifier.fit(x_train1, y_train1)
        x = np.array([[age, sex, chest_pain_type, resting_blood_pressure, serum_cholestrol, fasting_blood_sugar, resting_ecg, heart_rate,
                       exercise_induced_angina, Depression]])
        r = classifier.predict(x)
        return redirect(url_for('success',result=r))


if __name__ == '__main__':
    app.run(debug=True)
