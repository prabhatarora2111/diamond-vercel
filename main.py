#flask,scikit-learn,pandas,pickle-mixin
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import csv

app = Flask(__name__)
data = pd.read_csv('gemstone.csv')

with open('LassoModelDiamond.pkl', 'rb') as file:
    pipe = pickle.load(file)

@app.route("/")
def index():

    cut = sorted(data['cut'].unique())
    color = sorted(data['color'].unique())
    clarity = sorted(data['clarity'].unique())
    return render_template('index.html',cut = cut, clarity=clarity, color=color)

    # color = sorted(data['color'].unique())
    # return render_template('index.html',color = color)
    
    # clarity = sorted(data['clarity'].unique())
    # return render_template('index.html',clarity = clarity)

    # locations = sorted(data['location'].unique())
    # return render_template('index.html',locations=locations)

@app.route("/predict", methods = ['POST'])
def predict():

    carat = request.form.get('carat')
    cut = request.form.get('cut')
    color = request.form.get('color')
    clarity = request.form.get('clarity')
    depth = request.form.get('depth')
    table = request.form.get('table')
    x = request.form.get('x')
    y = request.form.get('y')
    z = request.form.get('z')


    # location = request.form.get('Location')
    # bhk = request.form.get('bhk')
    # bathroom = request.form.get('bathroom')
    # square_feet = request.form.get('square_feet')

    print(carat,cut,color,clarity,depth,table,x,y,z)
    input = pd.DataFrame([[carat,cut,color,clarity,depth,table,x,y,z]],columns = ['carat','cut','color','clarity','depth','table','x','y','z'])
    prediction = pipe.predict(input)[0]


    return str(np.round(prediction)*10)

# # Function to get user input
# def get_user_input():
#     location = location
#     bhk = bhk
#     bathroom = bathroom
#     square_feet = square_feet
#     return location, bhk, bathroom, square_feet

# # Function to add data to CSV file
# def add_data_to_csv(data, csv_file):
#     with open(csv_file, 'a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(data)

# # Path to the CSV file
# csv_file_path = 'user_data.csv'

# # Main function
# def main():
#     # Get user input
#     location, bhk, bathroom, square_feet = get_user_input()
    
#     # Data entered by the user
#     user_data = [location, bhk, bathroom, square_feet]
    
#     # Add data to CSV file
#     add_data_to_csv(user_data, csv_file_path)
    
#     print("Data added to CSV file successfully.")

if __name__=="__main__":
    app.run(debug=True,port=5002)
