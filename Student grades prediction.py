#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from tensorflow import keras


# In[2]:


df = pd.read_csv("~/MacintoshHD/Users/noeljoseph/Downloads/Finalassignment/train.csv")
print(df)


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


#transpose the df description so it is easier to read.
df.describe().transpose()


# In[6]:


df.isnull().sum()


# In[7]:


import matplotlib.pyplot as plt
# Plot histogram
df.hist(bins=50, figsize=(14,10))
plt.show()


# In[8]:


import seaborn as sns
sns.heatmap(df.corr())


# In[9]:


#one hot encode the df
df = pd.get_dummies(df)
print(df.shape)


# In[10]:


corr= df.corr()['Grade'].sort_values(ascending=False)
corr = corr.drop(['Grade','id'])
corr


# In[11]:


#drop ID don't need it
df = df.drop('id', axis=1)


# In[12]:


#split DF into X&y variables.
from sklearn.model_selection import train_test_split
X = df.drop('Grade',axis=1)
y = df['Grade']


# In[13]:


#split df into training, testing and validation sets. Random State ensures consistency
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state= 42)

print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))
print("X_val shape: {}".format(y_train.shape))
print("y val shape: {}".format(y_test.shape))


# In[14]:


#one hot encode y varaibles 
y_train_o_h = tf.one_hot(y_train, 20)
y_test_o_h = tf.one_hot(y_test, 20)


# In[15]:


#scale the X variables.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[16]:


#import a range of packages and classifiers
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib as plt


# In[17]:


#Create a pipeline through which we can pass our data
models = []
#models.append(('LR', LogisticRegression()))
#Logistic regression is appropriate here as we are looking for a value between 0&1
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('SupVectClass', SVC()))
models.append (('RFC', RandomForestClassifier()))
scoring = 'accuracy'


# In[18]:


# Evaluate each model in turn
results = []
names = []
#pass the data through the pipline as above and print the results. 
#note the use of K-Fold - this increases robustness but takes longer to run
for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    modelresult = "%s: %f" % (name, cv_results.mean())
    print(modelresult)


# In[19]:


# Create Decision Tree classifer object
DTC = DecisionTreeClassifier()

# Train Decision Tree Classifer
DTC = DTC.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = DTC.predict(X_test)


# In[20]:


# Save model to folder to use later
import pickle
filename = ("DTC_Model")
pickle.dump(DTC, open(filename, 'wb'))


# In[21]:


# Load model from folder and use it to make new predictions
DTC_Model = pickle.load(open(filename, 'rb'))


# In[22]:


RF_Test = pd.read_csv('/content/test.csv')
RF_Test = RF_Test.drop('id',axis=1)
RF_Test = RF_Test.dropna()
RF_Test = pd.get_dummies(RF_Test)


# In[23]:


# this outputs an array containing two values for each case being 0 or 1 (e.g. 0.03, 0.87)
DTC_Pred = DTC_Model.predict(RF_Test) 
print(DTC_Pred)


# In[24]:


# outputs to a CSV
output = pd.DataFrame(DTC_Pred, columns=['DTC_Grades']).to_csv('DTC_Grades_submission.csv')


# In[25]:


#this section is a whole convoluted way of getting ID's showing in the CSV output.
file = pd.read_csv("DTC_Grades_submission.csv")
# adding headers 
headerList = ['id', 'Grade']
  
# converting new data frame to csv
file.to_csv("DTC_Grades_submission2.csv", header=headerList, index=False)
  
# export CSV to files and with a print function to check results before we download them.
file2 = pd.read_csv("DTC_Grades_submission2.csv")
print('\File with Headers:')
print(file2)


# In[26]:


# import packages
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras import regularizers 
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.models import Sequential

# create an early stopping dropout if the model stops learning from the data - stops overfitting 
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10, restore_best_weights=True)
model = Sequential()
model.add(Dense(units=20,activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.25))

model.add(Dense(units=20,activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.27))

model.add(Dense(units=20,activation='relu'))
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy']
              )


# In[27]:


history = model.fit(x=X_train,
          y=y_train_o_h,
          epochs=250,
          validation_data=(X_test, y_test_o_h), verbose=0,
          callbacks=[early_stop]
          )
model.summary()


# In[28]:


model.save("content/Saved_Model.h5")


# In[29]:


mse_test = model.evaluate(X_test, y_test_o_h)
print(mse_test)


# In[30]:


import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5)) 
plt.grid(True)


# In[31]:


# Apply the model to the test split of the dataset. 
y_pred = model.predict(X_test)
#Unencode the predictions to a simple array [e.g. 1,2,3,4,5]
decoded_y_pred = tf.argmax(y_pred, axis=1)
decoded_y_pred


# In[32]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, decoded_y_pred)
print("MSE: ", mse)
print("RMSE: ", mse*(1/2.0))


# In[33]:


from sklearn.metrics import f1_score
f1_score(y_test, decoded_y_pred, average='macro')


# In[34]:


from keras import optimizers


# In[35]:


wdmodel = keras.models.Sequential([
  keras.layers.Flatten(),
  keras.layers.Dense(5000, activation='relu'),
  keras.layers.Dense(20, activation='relu')
])
loss=tf.keras.losses.categorical_crossentropy
optimizers = tf.keras.optimizers.SGD(lr=0.01)
metrics = ["accuracy"]
wdmodel.compile(loss=loss, optimizer=optimizers, metrics=metrics)
batch_size = 32
epoch = 5


# In[36]:


wdmodel.save("content/Saved_Model.h5")


# In[37]:


# Apply the model to the test split of the dataset. 
y_pred = wdmodel.predict(X_test)
#Unencode the predictions to a simple array [e.g. 1,2,3,4,5]
decoded_y_pred = tf.argmax(y_pred, axis=1)
decoded_y_pred


# In[38]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, decoded_y_pred)
print("MSE: ", mse)
print("RMSE: ", mse*(1/2.0))


# In[39]:


from sklearn.metrics import f1_score
f1_score(y_test, decoded_y_pred, average='macro')


# In[40]:


RNNmodel = keras.models.Sequential([
keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]), 
keras.layers.SimpleRNN(20, return_sequences=True), 
keras.layers.SimpleRNN(20)
])

RNNmodel.compile(loss="mse", optimizer="adam", metrics=["mse"])


# In[41]:


RNNmodel.summary()
history = RNNmodel.fit(x=X_train,
          y=y_train_o_h,
          epochs=75,
          validation_data=(X_val, y_val), verbose=0,
          callbacks=[early_stop])


# In[42]:


# Apply the model to the test split of the dataset. 
RNN_y_pred = RNNmodel.predict(X_test)
#Unencode the predictions to a simple array [e.g. 1,2,3,4,5]
decoded_RNN_y_pred = tf.argmax(RNN_y_pred, axis=1)


# In[43]:


RNNmodel.save("content/SavedRNN_Model.h5")


# In[44]:


from sklearn.metrics import f1_score
f1_score(y_test, decoded_RNN_y_pred, average='macro')


# In[45]:


LSTMmodel = keras.models.Sequential([
  keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]), 
  keras.layers.LSTM(20, return_sequences=True), 
  keras.layers.LSTM(20)
])

model.compile(loss="mse", optimizer="adam", metrics=["mse"])

model.summary()
history = model.fit(x=X_train,y=y_train_o_h, epochs=10, verbose=0)
mse_test = model.evaluate(X_test, y_test_o_h)


# In[46]:


# Apply the model to the test split of the dataset. 
RNN_y_pred = LSTMmodel.predict(X_test)
#Unencode the predictions to a simple array [e.g. 1,2,3,4,5]
decoded_RNN_y_pred = tf.argmax(RNN_y_pred, axis=1)


# In[47]:


LSTMmodel.save("content/SavedLSTMRNN_Model.h5")


# In[48]:


from sklearn.metrics import f1_score
f1_score(y_test, decoded_RNN_y_pred, average='macro')


# In[49]:


from tensorflow.keras.models import load_model
from tensorflow import keras
from sklearn import preprocessing


# In[50]:


test_df = pd.read_csv('/content/test.csv')
test_df = test_df.drop('id', axis=1)
test_df


# In[51]:


test_df.address.describe()


# In[52]:


test_df = pd.get_dummies(test_df)
print(test_df.shape)


# In[53]:


min_max_scaler = preprocessing.MinMaxScaler()
scaled_test_df = min_max_scaler.fit_transform(test_df)
scaled_test_df


# In[54]:


model = keras.models.load_model("content/Saved_Model.h5")


# In[55]:


grade = model.predict(scaled_test_df)


# In[56]:


decoded = tf.argmax(grade, axis=1)


# In[57]:


grade=decoded
grade


# In[58]:


# create an output item so the results can be called and create a CSV with these results. 
output = pd.DataFrame(grade, columns=['Grade']).to_csv('content/submission.csv')


# In[59]:


# read contents of csv file
file = pd.read_csv("content/submission.csv")
print("\nOriginal file:")
print(file)
  
# adding headers 
headerList = ['id', 'Grade']
  
# converting new data frame to csv
file.to_csv("content/Grade_Class_submission.csv", header=headerList, index=False)
  
# export CSV to files and with a print function to check results before we download them.
file2 = pd.read_csv("content/Grade_Class_submission.csv")
print('\File with Headers:')
print(file2)


# In[60]:


Model_Performance = pd.DataFrame({
    'Model': ['K-Nearest Neighbours','Decision Tree Classifier', 'Support Vector Classifier', 'Random Forest Classifier'],
    'Accuracy Score': [0.54, 0.98, 0.68, 0.88]
})
#define header names
col_names = ["ML Model", "Accuracy Score", "Accuracy Score"]
#display table
print(tabulate(Model_Performance, headers=col_names, tablefmt="fancy_grid", showindex="always"))


# In[61]:


Model_Performance.plot(x="Model", y=["Accuracy Score"], kind="bar")


# In[62]:


from tabulate import tabulate
#create data
data = [["Decision Tree Machine Classifier", 0.210, 0.51, " max_depth=60,\n max_features=auto,\n min_samples_leaf=2,\n min_samples_split=2,\n n_estimators=600"],
        ["Deep Neural Network", 0.076, 0.750, "Layer 1: units=27,activation='relu',\n kernel_regularizer=regularizers.l2(0.01) \n Dropout(0.5) \n Layer 2: units=27,activation='relu', \n kernel_regularizer=regularizers.l2(0.01) \n Dropout(0.5), \n Layer 1: units=2,activation='relu'"],  
        ["Wide Neural Network",  0.05, 0.58, "Layer 1: units=5000,activation='relu' \n Layer 2: units=1,activation='relu"], 
        ["Recurrent Neural Network",  0.092, 0.312, "Layer 1: 20, return_sequences=True), \n Layer 2: 20, return_sequences=True"],
        ["LSTM Recurrent Neural Network",  0.095, 0.58, "Layer 1: 20, return_sequences=True), \n Layer 2: 20, return_sequences=True"]]

#define header names
col_names = ["Method", "f1 Scores (3dp)","Kaggle Score (3dp)", "Parameter Notes"]
  
#display table
print(tabulate(data, headers=col_names, tablefmt="fancy_grid", showindex="always"))


# In[63]:


# Create a DF of Model Performance Scores
Model_Performance = pd.DataFrame({
    'Model': ['Random Forest ','Deep Neural Network', 'Wide Neural Network', 'RNN', 'LSTM RNN'],
    'LogLoss Score': [0.21, 0.076, 0.05, 0.092, 0.095],
    'Kaggle Performance Score': [0.51, 0.75, 0.58,0.312, 0.58]
})
# plot the pretty graph
Model_Performance.plot(x="Model", y=["LogLoss Score", "Kaggle Performance Score"], kind="bar")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




