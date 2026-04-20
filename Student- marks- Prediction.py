import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "study_hours":[1,2,3,4,5],
    "marks":[30,40,50,60,70]
}

df = pd.DataFrame(data)

X = df[['study_hours']]
y = df['marks']

model = LinearRegression()
model.fit(X,y)

print(model.predict([[6]]))
