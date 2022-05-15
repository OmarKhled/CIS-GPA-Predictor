from flask import Flask, request, abort, jsonify
from flask_cors import CORS
import json
import pickle
import numpy as np

# load the model from disk
loaded_carry = pickle.load(open('finalized_carry.sav', 'rb'))
loaded_cgpa = pickle.load(open('finalized_cgpaUnOP.sav', 'rb'))

app = Flask(__name__)

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers',
                         'Content-Type,Authorization,true')
    response.headers.add('Access-Control-Allow-Methods',
                         'POST')
    return response        

@app.route("/api/ml", methods=["POST"])
def run_ML():
    body = request.get_json()
    data = eval(body["data"])
    print("body", body)
    e1 = str(data.get('interactionLevel', '1'))
    e2 = str(data.get('offlineDays', '1'))
    e3 = str(data.get('academicLifeImportantance', '1'))
    e4 = str(data.get('age', '0'))
    e5 = str(data.get('year', '1'))
    e6 = str(data.get('transportationTime', '0'))
    e7 = str(data.get('school', '0'))
    e8 = str(data.get('credits', '0'))
    e9 = str(data.get('friends', '0'))
    e10 = str(data.get('studentActivities', '0'))

    print(e1, e2, e3, e4, e5, e6, e7, e8, e9, e10)

    if e4 == "0":
        age_list = [1,0,0]
    elif e4=="1":
        age_list = [0,1,0]
    elif e4=="2":
        age_list = [0,0,1]
    year_list = [0]*5
    year_list[int(e5)-1] = 1
    living_list = [0]*5
    living_list[int(e6)] = 1
    school_list= [0]*4
    school_list[int(e7)] = 1
    credit_list= [0]*3
    credit_list[int(e8)] = 1
    friends_list= [0]*5
    friends_list[int(e9)] = 1
    activities_list = [0]*4
    activities_list[int(e10)] = 1
    listers = [int(e1), int(e2), int(e3)]

    k = [age_list, year_list, living_list, school_list, credit_list, friends_list, activities_list]

    for l in k:
        listers.extend(l)
    if loaded_carry.predict(np.array(listers).reshape(1,32)):
        carries = "Yes"
    else:
        carries = "No"
    response = {
        'success': True,
        'PCGPA': str(loaded_cgpa.predict(np.array(listers).reshape(1,32))),
        'Carries': carries,
        'Inputs': body["data"]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)
