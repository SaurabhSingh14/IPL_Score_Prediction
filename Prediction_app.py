from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/', methods=['GET', 'POST'])
def home():
    feature_list = []
    model = load('IPL Score Prediction.joblib')

    if request.method == 'POST':
        batting_team = request.form.get('batteam')
        bowling_team = request.form.get('bowlteam')
        runs = int(request.form.get('runs'))
        feature_list.append(runs)
        wickets = int(request.form.get('wickets'))
        feature_list.append(wickets)
        overs = int(request.form.get('overs'))
        feature_list.append(overs)
        run_5 = int(request.form.get('run_5'))
        feature_list.append(run_5)
        wickets_5 = int(request.form.get('wickets_5'))
        feature_list.append(wickets_5)

        if(batting_team=='Royal Challengers Bangalore'):
            bat_lst = [0, 0, 0, 0, 0, 0, 1, 0]
            team1 = 'RCB'
        elif (batting_team=='Chennai Super Kings'):
            bat_lst = [1, 0, 0, 0, 0, 0, 0, 0]
            team1 = 'CSK'
        elif (batting_team=='Mumbai Indians'):
            bat_lst = [0, 0, 0, 0, 1, 0, 0, 0]
            team1 = 'MI'
        elif (batting_team=='Kings XI Punjab'):
            bat_lst = [0, 0, 1, 0, 0, 0, 0, 0]
            team1 = 'KXIP'
        elif (batting_team=='Delhi Capitals'):
            bat_lst = [0, 1, 0, 0, 0, 0, 0, 0]
            team1 = 'DC'
        elif (batting_team=='Sunrisers Hyderabad'):
            bat_lst = [0, 0, 0, 0, 0, 0, 0, 1]
            team1 = 'SRH'
        elif (batting_team=='Kolkata Knight Riders'):
            bat_lst = [0, 0, 0, 1, 0, 0, 0, 0]
            team1 = 'KKR'
        else:
            bat_lst = [0, 0, 0, 0, 0, 1, 0, 0]
            team1 = 'RR'

        if(bowling_team=='Royal Challengers Bangalore'):
            bowl_lst = [0, 0, 0, 0, 0, 0, 1, 0]
            team2 = 'RCB'
        elif (bowling_team=='Chennai Super Kings'):
            bowl_lst = [1, 0, 0, 0, 0, 0, 0, 0]
            team2 = 'CSK'
        elif (bowling_team=='Mumbai Indians'):
            bowl_lst = [0, 0, 0, 0, 1, 0, 0, 0]
            team2 = 'MI'
        elif (bowling_team=='Kings XI Punjab'):
            bowl_lst = [0, 0, 1, 0, 0, 0, 0, 0]
            team2 = 'KXIP'
        elif (bowling_team=='Delhi Capitals'):
            bowl_lst = [0, 1, 0, 0, 0, 0, 0, 0]
            team2 = 'DC'
        elif (bowling_team=='Sunrisers Hyderabad'):
            bowl_lst = [0, 0, 0, 0, 0, 0, 0, 1]
            team2 = 'SRH'
        elif (bowling_team=='Kolkata Knight Riders'):
            bowl_lst = [0, 0, 0, 1, 0, 0, 0, 0]
            team2 = 'KKR'
        else:
            bowl_lst = [0, 0, 0, 0, 0, 1, 0, 0]
            team2 = 'RR'

        for x in bat_lst:
            feature_list.append(x)
        for x in bowl_lst:
            feature_list.append(x)

        prediction = int(np.round(model.predict([feature_list])))

        return render_template('result.html', result_1=prediction-5, result_2=prediction+5, team1=team1,
                               team2=team2)

    return render_template('index.html')

if __name__ == '__main__':
    app.run()
