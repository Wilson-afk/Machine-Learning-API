import flask
from flask import Flask, jsonify, request
import json
import joblib
import numpy as np
import traceback
import requests

app = Flask(__name__)

model = joblib.load('models/Final-Model.sav')
print('Model loaded')

model_columns = joblib.load('models/Final-Model-Columns.sav')
print('Model columns loaded')

@app.route('/predict', methods=['GET','POST'])
def predict():
    
    if model:
        try:

            content = request.get_json(force=True)
            
            arr = []
            lab = []
            diction = []

            for i in range(14):
                model_i = joblib.load('labels/Label_'+str(i)+'.sav')
                lab.append(model_i)
                
            main = joblib.load('labels/Label_main.sav')

            for i in range(len(lab)):
                model_i = dict(zip(lab[i].transform(lab[i].classes_), lab[i].classes_))
                diction.append(model_i)
              
            main = dict(zip(main.transform(main.classes_), main.classes_))

            for i in range(len(content)):
                drr = list(dict(content[i]).values())
                
                lij = []
                for k in range(len(diction)):
                    lij.append(diction[k])

                for i in range(len(lij)):
                    for r in range(len(drr)):
                        for j in range(len(lij[i])):
                            if drr[r] == lij[i][j]:
                            
                                drw = list(lij[i].keys())
                                drr[r] = drw[j]
                 
                x_in = np.array(drr).reshape(1,-1)
                prediction = int(model.predict(x_in)[0])
                arr.append(prediction)
            
            for j in range(len(arr)):
                for k in range(len(main)):
                    if arr[j] == list(main.keys())[k]:
                    
                        arr[j] = list(main.values())[k]
            
            response = json.dumps({'response': arr})
            return response, 200
        
        except:
            return jsonify({'trace': traceback.format_exc()})
    
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    app.run(port=5000, debug=True)