#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
import re

st.set_page_config(layout="centered")
# Read the CSV file into a pandas dataframe
df = pd.read_csv(r"Copper_Set.csv")
df.head(2)
print(len(df['item_date'].unique())) #
print(len(df['customer'].unique())) #
print(len(df['material_ref'].unique()))
print(len(df['product_ref'].unique())) #
print(len(df['delivery date'].unique())) #

###null values, shape and data types before dealing with incorrect data

df.shape

# missing values
missing_values_count = df.isnull().sum()
print(missing_values_count)

df.info()

#Dealing with data in wrong format

# dealing with data in wrong format
# for categorical variables, this step is ignored
# df = df[df['status'].isin(['Won', 'Lost'])]
df['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce').dt.date
df['quantity_tons'] = pd.to_numeric(df['quantity_tons'], errors='coerce')
df['customer'] = pd.to_numeric(df['customer'], errors='coerce')
df['country'] = pd.to_numeric(df['country'], errors='coerce')
df['application'] = pd.to_numeric(df['application'], errors='coerce')
df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce')
df['width'] = pd.to_numeric(df['width'], errors='coerce')
df['material_ref'] = df['material_ref'].str.lstrip('0')
df['product_ref'] = pd.to_numeric(df['product_ref'], errors='coerce')
df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce').dt.date
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')

##Nan values and shape after dealing with data in wrong format

missing_values_count = df.isnull().sum()
print(missing_values_count)
print(df.shape)
df.info()

##Dealing with Missing Values

# material_ref has large set of null values, so replacing them with unknown
df['material_ref'].fillna('unknown', inplace=True)
# deleting the remaining null values as they are less than 1% of data which can be neglected
df = df.dropna()

missing_values_count = df.isnull().sum()
print(missing_values_count)
print(df.shape)

df_p=df.copy()

####PLOTTING

# ['quantity_tons','selling_price','application','thickness','width', 'country']
# ['status','item type']

df_p['quantity_tons']   # skewed
df_p['country']
df_p['application']
df_p['thickness']       # skewed
df_p['width']
df_p['selling_price']   # skewed

import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(df_p['quantity_tons'])
plt.show()
sns.distplot(df_p['country'])
plt.show()
sns.distplot(df_p['application'])
plt.show()
sns.distplot(df_p['thickness'])
plt.show()
sns.distplot(df_p['width'])
plt.show()
sns.distplot(df_p['selling_price'])
plt.show()


import numpy as np
mask1 = df_p['selling_price'] <= 0
print(mask1.sum())
df_p.loc[mask1, 'selling_price'] = np.nan

mask1 = df_p['quantity_tons'] <= 0
print(mask1.sum())
df_p.loc[mask1, 'quantity_tons'] = np.nan

mask1 = df_p['thickness'] <= 0
print(mask1.sum())

df_p.isnull().sum()

df_p.dropna(inplace=True)
len(df_p)

import numpy as np
df_p['selling_price_log'] = np.log(df_p['selling_price'])
sns.distplot(df_p['selling_price_log'])
plt.show()

df_p['quantity_tons_log'] = np.log(df_p['quantity_tons'])
sns.distplot(df_p['quantity_tons_log'])
plt.show()

df_p['thickness_log'] = np.log(df_p['thickness'])
sns.distplot(df_p['thickness_log'])
plt.show()
# reverts log
# df_p['reverted_values'] = np.exp(df_p['selling_price_log'])
# sns.distplot(df_p['reverted_values'])
# plt.show()


df_p.head()

x=df_p[['quantity_tons_log','application','thickness_log','width','selling_price_log','country','customer',
        'product_ref']].corr()


sns.heatmap(x, annot=True, cmap="YlGnBu")

#DECISION TREE REGRESSOR
#1 Encode categorical features
#2 Test and Train split
#3 Define Hyperparameters for GridSearchCV
#4 Train the DecisionTreeRegressor with best parameters
#5 Evalution metrics
#6 Predict the selling price for new values
#7 Save the DecisionTreeRegressor model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

X=df_p[['quantity_tons_log','status','item type','application','thickness_log','width','country','customer',
        'product_ref']]
y=df_p['selling_price_log']
# encoding categorical variables
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(X[['item type']])
X_ohe = ohe.fit_transform(X[['item type']]).toarray()
ohe2 = OneHotEncoder(handle_unknown='ignore')
ohe2.fit(X[['status']])
X_be = ohe2.fit_transform(X[['status']]).toarray()
# independent features after encoding
X = np.concatenate((X[['quantity_tons_log', 'application', 'thickness_log', 'width','country','customer',
                       'product_ref']].values, X_ohe, X_be), axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
# test and train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# decision tree
dtr = DecisionTreeRegressor()
# hyperparameters
param_grid = {'max_depth': [2, 5, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'max_features': ['auto', 'sqrt', 'log2']}
# gridsearchcv
grid_search = GridSearchCV(estimator=dtr, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
# evalution metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean squared error:', mse)
print('R-squared:', r2)

# ['quantity_tons_log', 'application', 'thickness_log', 'width','country','customer','product_ref']].values, X_ohe, X_be
new_sample = np.array([[np.log(40), 10, np.log(250), 0, 28,30202938,1670798778,'PL','Won']])
new_sample_ohe = ohe.transform(new_sample[:, [7]]).toarray()
new_sample_be = ohe2.transform(new_sample[:, [8]]).toarray()
new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
new_sample1 = scaler.transform(new_sample)
new_pred = best_model.predict(new_sample1)
print('Predicted selling price:', np.exp(new_pred))


# Saving the model
import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('t.pkl', 'wb') as f:
    pickle.dump(ohe, f)
with open('s.pkl', 'wb') as f:
    pickle.dump(ohe2, f)

print(len(df_p))
df_p.head(3)

df_c = df_p[df_p['status'].isin(['Won', 'Lost'])]
len(df_c)

#DECISION TREE CLASSIFIER

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelBinarizer

Y = df_c['status']
X= df_c[['quantity_tons_log','selling_price_log','item type','application','thickness_log','width','country',
         'customer','product_ref']]

# encoding categorical variables
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(X[['item type']])
X_ohe = ohe.fit_transform(X[['item type']]).toarray()
be = LabelBinarizer()
be.fit(Y)
y = be.fit_transform(Y)
# independent features after encoding
X = np.concatenate((X[['quantity_tons_log', 'selling_price_log','application', 'thickness_log', 'width','country',
                       'customer','product_ref']].values, X_ohe), axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# decision tree classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

# Evalution Metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# Predict the status for a new sample
# 'quantity_tons_log', 'selling_price_log','application', 'thickness_log', 'width','country','customer',
# 'product_ref']].values, X_ohe
new_sample = np.array([[np.log(700), np.log(956), 10, np.log(2),1500,28.0,30202938,1670798778,'W']])
new_sample_ohe = ohe.transform(new_sample[:, [8]]).toarray()
new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,7]], new_sample_ohe), axis=1)
new_sample = scaler.transform(new_sample)
new_pred = dtc.predict(new_sample)
if new_pred==1:
    print('The status is: Won')
else:
    print('The status is: Lost')

# Saving the model
import pickle
with open('cmodel.pkl', 'wb') as file:
    pickle.dump(dtc, file)
with open('cscaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('ct.pkl', 'wb') as f:
    pickle.dump(ohe, f)



#st.set_page_config(layout="centered")

st.write("""
<div style='text-align:center'>
    <h1 style='color:#009999;'>Industrial Copper Modeling Application</h1>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"])
with tab1:
    # Define the possible values for the dropdown menus
    status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered',
                      'Offerable']
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67.,
                           79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665',
               '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407',
               '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
               '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
               '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

    # Define the widgets for user input
    with st.form("my_form"):
        col1, col2, col3 = st.columns([5, 2, 5])
        with col1:
            st.write(' ')
            status = st.selectbox("Status", status_options, key=1)
            item_type = st.selectbox("Item Type", item_type_options, key=2)
            country = st.selectbox("Country", sorted(country_options), key=3)
            application = st.selectbox("Application", sorted(application_options), key=4)
            product_ref = st.selectbox("Product Reference", product, key=5)
        with col3:
            st.write(
                f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>',
                unsafe_allow_html=True)
            quantity_tons = st.text_input("Enter Quantity_Tons (Min:611728 & Max:1722207579)")
            thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            width = st.text_input("Enter width (Min:1, Max:2990)")
            customer = st.text_input("customer ID (Min:12458, Max:30408185)")
            submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
            st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #009999;
                        color: white;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)

        flag = 0
        pattern = "^(?:\d+|\d*\.\d+)$"
        for i in [quantity_tons, thickness, width, customer]:
            if re.match(pattern, i):
                pass
            else:
                flag = 1
                break

    if submit_button and flag == 1:
        if len(i) == 0:
            st.write("please enter a valid number space not allowed")
        else:
            st.write("You have entered an invalid value: ", i)

    if submit_button and flag == 0:
        import pickle

        with open(r"source/model.pkl", 'rb') as file:
            loaded_model = pickle.load(file)
        with open(r'source/scaler.pkl', 'rb') as f:
            scaler_loaded = pickle.load(f)

        with open(r"source/t.pkl", 'rb') as f:
            t_loaded = pickle.load(f)

        with open(r"source/s.pkl", 'rb') as f:
            s_loaded = pickle.load(f)

        new_sample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)), float(width),
                                country, float(customer), int(product_ref), item_type, status]])
        new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
        new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, ]], new_sample_ohe, new_sample_be), axis=1)
        new_sample1 = scaler_loaded.transform(new_sample)
        new_pred = loaded_model.predict(new_sample1)[0]
        st.write('## :green[Predicted selling price:] ', np.exp(new_pred))

with tab2:
    with st.form("my_form1"):
        col1, col2, col3 = st.columns([5, 1, 5])
        with col1:
            cquantity_tons = st.text_input("Enter Quantity_Tons (Min:611728 & Max:1722207579)")
            cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            cwidth = st.text_input("Enter width (Min:1, Max:2990)")
            ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
            cselling = st.text_input("Selling Price (Min:1, Max:100001015)")

        with col3:
            st.write(' ')
            citem_type = st.selectbox("Item Type", item_type_options, key=21)
            ccountry = st.selectbox("Country", sorted(country_options), key=31)
            capplication = st.selectbox("Application", sorted(application_options), key=41)
            cproduct_ref = st.selectbox("Product Reference", product, key=51)
            csubmit_button = st.form_submit_button(label="PREDICT STATUS")

        cflag = 0
        pattern = "^(?:\d+|\d*\.\d+)$"
        for k in [cquantity_tons, cthickness, cwidth, ccustomer, cselling]:
            if re.match(pattern, k):
                pass
            else:
                cflag = 1
                break

    if csubmit_button and cflag == 1:
        if len(k) == 0:
            st.write("please enter a valid number space not allowed")
        else:
            st.write("You have entered an invalid value: ", k)

    if csubmit_button and cflag == 0:
        import pickle

        with open(r"source/cmodel.pkl", 'rb') as file:
            cloaded_model = pickle.load(file)

        with open(r'source/cscaler.pkl', 'rb') as f:
            cscaler_loaded = pickle.load(f)

        with open(r"source/ct.pkl", 'rb') as f:
            ct_loaded = pickle.load(f)

        # Predict the status for a new sample
        # 'quantity_tons_log', 'selling_price_log','application', 'thickness_log', 'width','country','customer',
        # 'product_ref']].values, X_ohe
        new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication,
                                np.log(float(cthickness)), float(cwidth), ccountry, int(ccustomer), int(product_ref),
                                citem_type]])
        new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, 7]], new_sample_ohe), axis=1)
        new_sample = cscaler_loaded.transform(new_sample)
        new_pred = cloaded_model.predict(new_sample)
        if new_pred == 1:
            st.write('## :green[The Status is Won] ')
        else:
            st.write('## :red[The status is Lost] ')

st.write(f'<h6 style="color:rgb(0, 153, 153,0.35);">App Created by Rohini S</h6>', unsafe_allow_html=True)


# In[ ]:




