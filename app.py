from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract specific data from form by name to match the Model's order
        # Model expects 8 features: 
        # ['CGPA', 'Internships', 'Projects', 'Workshops', 'ExtraCurricular', 'PlacementTraining', 'SSC', 'HSC']
        
        features = [
            float(request.form['cgpa']),
            int(request.form['internships']),
            int(request.form['projects']),
            int(request.form['workshops']),
            int(request.form['extra_curricular']),
            int(request.form['placement_training']),
            float(request.form['ssc_marks']),
            float(request.form['hsc_marks'])
        ]
        
        # Convert to numpy array
        final_features = [np.array(features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        
        output = 'Placed' if prediction[0] == 1 else 'Not Placed'
        
        return render_template('index.html', prediction_text='Status: {}'.format(output))

    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(e))

if __name__ == "__main__":
    app.run(debug=True)