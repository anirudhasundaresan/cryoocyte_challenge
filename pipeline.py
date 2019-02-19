import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV

# Reading in the data
train_csv = pd.read_csv("train_.csv")
# train_csv_dims = train_csv.shape
# print("Shape of the train csv file is: ", train_csv_dims)

# Check which cols are numeric/ categorical
cols = train_csv.columns # all cols
num_cols = train_csv._get_numeric_data().columns # only numeric cols

'''
print("Number of numeric cols: ", len(num_cols))
print("Number of categorical cols: ", len(list(set(cols) - set(num_cols))))
print("Categorical columns are: ", list(set(cols) - set(num_cols)))
'''

# we need count of unique values for all cols just to make sure any of the numeric values are not categorical
'''
for col in train_csv:
    if len(train_csv[col].unique()) < 8000:
        print(col, len(train_csv[col].unique()))
        # verifies that x13, x68 and x91 are indeed the only categorical values
'''

# get total number of rows with nans in them
'''
inds = pd.isnull(train_csv).any(1).nonzero()[0]
print("Number of rows in which NaNs are present: ", len(inds)) - there are only 73/8000 rows where nans are present; better to remove them since 73 is a small portion and also because I do not have domain knowledge
# else, I could have tried imputation techniques. For now, best to remove them so that the model that is developed is generalizable as well.
'''
train_csv.dropna(inplace=True)
print("Modifying training set to remove rows with NaNs, so training set now with", set(train_csv.count().tolist()), 'rows.') # there are now 7927 rows for the training set in train_csv.

# we need to check for outliers in the data for each column
# let us keep the categorical columns separate and work on the numeric data
cat_train_csv = pd.concat([train_csv['x13'], train_csv['x68'], train_csv['x91']], axis=1)
cat_train_csv.reset_index(inplace=True, drop=True)
label_train_csv = train_csv['y']
train_csv.drop(['x13', 'x68', 'x91', 'y'], axis=1, inplace=True)

# y = train_csv[train_csv.apply(lambda x :(x-x.mean()).abs()<(3*x.std()) ).all(1)] # we can do this to check for outliers since this is now scaled data (looks like a lot of outliers)
# removing outliers from the data is questionable, so I will leave it as it is.

# Explore correlations between features, and between features and targets
corr = train_csv.corr()

def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

# print("Top Absolute Correlations")
corr_stats = get_top_abs_correlations(train_csv, 10) # print(corr_stats) - to see which all features are correlated with one another the most
# looks like 'x25' is most correlated with y and there are some features that are correlated with one another.

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
# plt.show() - a bit dense, not much information can be gained; there are some highly correlated variables; will need PCA fit_transform.

# main preprocessing before building model
# let us scale data in all columns; regression anyway requires scaled data; RF does not require scaling. PCA must be done only after scaling
scaler = StandardScaler()
train_csv[train_csv.columns] = scaler.fit_transform(train_csv[train_csv.columns])
pca = PCA(n_components=0.99, svd_solver='full') # make sure we have 95% variance explained when we transform
train_csv_transformed = pca.fit_transform(train_csv)
dummy_train = pd.get_dummies(data=cat_train_csv, drop_first=True)

### Training phase
# final mx for training
train_csv_final = pd.concat([pd.DataFrame(train_csv_transformed), dummy_train], axis=1)

print("Training scores: ")
# trying elastic net regression since there is correlation between features. We might not want to fully get rid of transformed PCA features
regr = ElasticNetCV(cv=10, random_state=0)
regr.fit(train_csv_final, label_train_csv)
print("ElasticNet R^2 score: ", regr.score(train_csv_final, label_train_csv))

ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(train_csv_final, label_train_csv)
print("Ridge R^2 score: ", ridge.score(train_csv_final, label_train_csv))

mlp_regr = MLPRegressor(hidden_layer_sizes=(13, 13, 13))
mlp_regr.fit(train_csv_final, label_train_csv)
print("MLP R^2 score: ", mlp_regr.score(train_csv_final, label_train_csv))
print()


### Validation phase (to be commented out when finalizing model - this phase used only for tuning)
valid_csv = pd.read_csv("train_anirudha.csv")
valid_csv.dropna(inplace=True)
valid_labels = valid_csv['y']
cat_valid_csv = pd.concat([valid_csv['x13'], valid_csv['x68'], valid_csv['x91']], axis=1)
cat_valid_csv.reset_index(inplace=True, drop=True)
valid_csv.drop(['x13', 'x68', 'x91', 'y'], axis=1, inplace=True)

# preprocessing validation data sccording to the training data
valid_csv[valid_csv.columns] = scaler.transform(valid_csv[valid_csv.columns])
valid_csv_transformed = pca.transform(valid_csv)

dummy_valid = pd.get_dummies(data=cat_valid_csv, drop_first=True)
valid_csv_final = pd.concat([pd.DataFrame(valid_csv_transformed), dummy_valid], axis=1)

print("Validation scores: ")
elastic_predict = regr.predict(valid_csv_final)
ridge_predict = ridge.predict(valid_csv_final)
mlp_predict = mlp_regr.predict(valid_csv_final)
print("ElasticNet R^2 score: ", regr.score(valid_csv_final, valid_labels))
print("Ridge R^2 score: ", ridge.score(valid_csv_final, valid_labels))
print("MLP R^2 score: ", mlp_regr.score(valid_csv_final, valid_labels))
print()

### Testing phase (to be used for final submission)
test_csv = pd.read_csv("test_.csv")
'''
# none of the categoricals are missing, I guess it is safe to impute with mean for the other missing rows
>>> test_csv = pd.read_csv("test_.csv")
>>> test_csv.count()['x13']
2000
>>> test_csv.count()['x68']
2000
>>> test_csv.count()['x91']
2000
'''
# preprocessing test data according to the training set
test_csv.fillna(test_csv.mean(), inplace=True) # imputing for test data only - not imputing for train since not many rows with NaNs.
cat_test_csv = pd.concat([test_csv['x13'], test_csv['x68'], test_csv['x91']], axis=1)
cat_test_csv.reset_index(inplace=True, drop=True)
test_csv.drop(['x13', 'x68', 'x91'], axis=1, inplace=True)
test_csv[test_csv.columns] = scaler.transform(test_csv[test_csv.columns])
test_csv_transformed = pca.transform(test_csv)

dummy_test = pd.get_dummies(data=cat_test_csv, drop_first=True)
test_csv_final = pd.concat([pd.DataFrame(test_csv_transformed), dummy_test], axis=1)

print("Test results: ")
elastic_predict = regr.predict(test_csv_final)
ridge_predict = ridge.predict(test_csv_final)
mlp_predict = mlp_regr.predict(test_csv_final)
print("ElasticNet results: ", elastic_predict)
print("Ridge results: ", ridge_predict)
print("MLP results: ", mlp_predict)
print()
