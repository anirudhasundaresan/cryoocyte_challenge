from sklearn.linear_model import ElasticNetCV
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

train_csv = pd.read_csv("train_.csv")
train_csv_dims = train_csv.shape
print("Shape of the train csv file is: ", train_csv_dims) # 8000 x 101
# x0        x1        x2        x3        x4        x5        x6            x7        x8    ...          x92       x93       x94       x95       x96       x97       x98       x99         y

# Check which cols are numeric/ categorical
cols = train_csv.columns # all cols
num_cols = train_csv._get_numeric_data().columns # only numeric cols
print("Number of numeric cols: ", len(num_cols))
print("Number of categorical cols: ", len(list(set(cols) - set(num_cols))))
print("Categorical columns are: ", list(set(cols) - set(num_cols)))

# we need count of unique values for all cols just to make sure any of the numeric values are not categorical
for col in train_csv:
    if len(train_csv[col].unique()) < 7500:
        print(col, len(train_csv[col].unique()))
        # verifies that x13, x68 and x91 are indeed the only categorical values
# some cols have <=8000 unique values, hence this means some of them has nans

# get total number of rows with nans in them
inds = pd.isnull(train_csv).any(1).nonzero()[0]
print("Number of rows in which NaNs are present: ", len(inds)) # there are only 73/8000 rows where nans are present; better to remove them since 73 is a small portion and also because I do not have domain knowledge
# else, I could have tried imputation techniques. For now, best to remove them so that the model that is developed is generalizable as well.
# print('Number of NaNs for all the columns together: ', sum(8000-x for x in train_csv.count().tolist() if x<8000)) # just verifying

train_csv.dropna(inplace=True)
print("Modifying training set to remove rows with NaNs, so training set now with", set(train_csv.count().tolist()), 'rows.') # there are now 7927 rows for the training set in train_csv.

# we need to check for outliers in the data for each column
# let us keep the categorical columns separate and work on the numeric data
train_all = train_csv.copy()
cat_train_csv = pd.concat([train_csv['x13'], train_csv['x68'], train_csv['x91']], axis=1)
cat_train_csv.reset_index(inplace=True, drop=True)

label_train_csv = train_csv['y']

del train_csv['x13']
del train_csv['x68']
del train_csv['x91']
del train_csv['y']

# let us scale data in all columns; regression anyway requires scaled data; RF does not require scaling. PCA must be done only after scaling
# train_csv has only numerical values now; remember to use cat_train_csv while building the model

scaler = StandardScaler()
train_csv[train_csv.columns] = scaler.fit_transform(train_csv[train_csv.columns])
# X_test = scaler.transform(X_test)
# y = train_csv[train_csv.apply(lambda x :(x-x.mean()).abs()<(3*x.std()) ).all(1)] # we can do this to check for outliers since this is now scaled data
# len(y) = 6712x98 which means that there are some outlier data - not sure if we should remove outliers since they might be important

# explore the dataset, see correlations and get PCA curves, see how much variance is explained in each column.

print("Correlation Matrix")
corr = train_csv.corr()
print(corr)
print()

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

print("Top Absolute Correlations")
corr_stats = get_top_abs_correlations(train_csv, 10)
print(corr_stats)
# looks like x25 is most correlated with y and there are some features that are correlated with one another.

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
# plt.show() # a bit dense, not much information can be gained; there are some highly correlated variables


# it would help to model our task by using PCA, and to decide the number of components for PCA, we need the PCA curve with the explained variance
pca = PCA(n_components=0.95, svd_solver='full') # make sure we have 95% variance explained when we transform
train_csv_transformed = pca.fit_transform(train_csv)


X = pd.get_dummies(data=cat_train_csv, drop_first=True)
train_csv_final = pd.concat([pd.DataFrame(train_csv_transformed), X], axis=1)

# let us try regression; do cross validation
regr = ElasticNetCV(cv=10, random_state=0)
regr.fit(train_csv_final, label_train_csv)
print(regr.score(train_csv_final, label_train_csv))

'''
lm = LinearRegression()
mse_scores = -cross_val_score(lm, train_csv_transformed, label_train_csv, cv=10, scoring='neg_mean_squared_error')
# fix the sign of MSE scores
rmse_scores = np.sqrt(mse_scores)
print(rmse_scores)
# calculate the average RMSE
print(rmse_scores.mean())
# do elastic since we want feature selection and regularization
'''


# print(x)
# print(y)
