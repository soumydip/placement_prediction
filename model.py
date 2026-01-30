import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Data Load
df = pd.read_csv('aiml/placementdata.csv')

# 2. Data Preprocessing
df['PlacementStatus'] = df['PlacementStatus'].replace({'NotPlaced': 0, 'Placed': 1})
df['ExtracurricularActivities'] = df['ExtracurricularActivities'].replace({'No': 0, 'Yes': 1})
df['PlacementTraining'] = df['PlacementTraining'].replace({'No': 0, 'Yes': 1})

# Feature Selection (Total 8 features)
x = df[['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 'ExtracurricularActivities', 'PlacementTraining', 'SSC_Marks', 'HSC_Marks']]
y = df['PlacementStatus']

# 3. Train Test Split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# 4. Model Training
model = RandomForestClassifier(n_estimators=100)
model.fit(xtrain, ytrain)

print("Model Training Complete!")

# 5. Save the MODEL (Not the prediction variable)


with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("model.pkl saved successfully!")