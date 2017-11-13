import pandas as pd
# Further imports
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import sgd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

feature_names = ['party', 'handicapped-infants', 'water-project-cost-sharing',
                 'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                 'el-salvador-aid', 'religious-groups-in-schools',
                 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                 'mx-missle', 'immigration', 'synfuels-corporation-cutback',
                 'education-spending', 'superfund-right-to-sue', 'crime',
                 'duty-free-exports', 'export-administration-act-south-africa']

voting_data = pd.read_csv('house-votes-84.data.txt', na_values=['?'],
                          names=feature_names)
voting_data.head()

voting_data.describe()

voting_data.dropna(inplace=True)
voting_data.describe()

voting_data.replace(('y','n'), (1,0), inplace=True)
voting_data.replace(('democrat','republican'), (1,0), inplace=True)

voting_data.head()

all_features = voting_data[feature_names].values
all_classes = voting_data['party'].values

#   Done setting up inputs - now to implement ML
def create_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', kernel_initializer='normal',  input_dim=17))
    model.add(Dropout(0.5))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=create_model, nb_epoch=100, verbose=0)

cv_scores = cross_val_score(estimator, all_features, all_classes, cv=10)
print(cv_scores.mean())