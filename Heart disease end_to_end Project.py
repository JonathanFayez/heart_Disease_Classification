#!/usr/bin/env python
# coding: utf-8

# ## Predicting if a patient has heart disease or not
# ### let's start with the EDA(Exploratory Data Analysis)

# What questions are we trying to solve?
# What kind of data do we have and how do we treat different types?
# What's missing from the data and how do we deal with it?
# Where are the outliers and why should we care about them?
# How can w add, change or remove features to get more out of our data?

# In[1]:


## importing the important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


heart_disease= pd.read_csv('heart_disease.csv')
heart_disease.shape


# In[3]:


heart_disease.head()


# In[4]:


heart_disease['target'].value_counts()


# we've got 165 patients with heart disease amd 138 without

# In[5]:


heart_disease['target'].value_counts().plot(kind='bar',color=['lightblue','blue']);


# In[6]:


heart_disease.info()


# In[7]:


heart_disease.isnull().sum()


# In[8]:


heart_disease.describe()


# In[9]:


### 1 is male and 0 is female
heart_disease['sex'].value_counts()


# In[10]:


pd.crosstab(heart_disease['target'],heart_disease['sex'])


# In[11]:


pd.crosstab(heart_disease['target'],heart_disease['sex']).plot(kind='bar')
plt.xlabel('0 is no disease and 1 is disease')
plt.legend(['Female','Male']);


# In[12]:


heart_disease.head()


# cp - chest pain type
# 0: Typical angina: chest pain related decrease blood supply to the heart
# 
# 1: Atypical angina: chest pain not related to heart
# 
# 2: Non-anginal pain: typically esophageal spasms (non heart related)
# 
# 3: Asymptomatic: chest pain not showing signs of disease

# In[13]:


pd.crosstab(heart_disease.cp,heart_disease.target)


# In[14]:


pd.crosstab(heart_disease.cp,heart_disease.target).plot.bar()
plt.xlabel('Chest pain type')
plt.legend(['No heart disease','heart disease'])


# isn't it weird that there is 69 out of 87 patients with type 2 which is not related to the heart have heart disease

# In[15]:


# Scatter with positive examples
plt.figure(figsize=(10,6))

plt.scatter(heart_disease.age[heart_disease.target==1],
            heart_disease.thalach[heart_disease.target==1],
            c="salmon")

# Scatter with negative examples
plt.scatter(heart_disease.age[heart_disease.target==0],
            heart_disease.thalach[heart_disease.target==0],
            c="lightblue")

# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);


# In[16]:


heart_disease.corr()


# In[17]:


plt.subplots(figsize=(15,10))
sns.heatmap(heart_disease.corr(),annot=True);


# 

# ### Prepare the data for machine learning

# In[18]:


from sklearn.model_selection import train_test_split
np.random.seed(40)
X = heart_disease.drop('target',axis=1)
y = heart_disease['target']

X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)


# we're gonna use 3 models for classification which are
# 
# 
# 1: KNeighborsClassifier
# 
# 2: LogisticRegression
# 
# 3:RandomForestClassifier

# In[19]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[20]:


def model_accuracy(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier()}
    np.random.seed(40)
    accuracies = {}
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions on the test data
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = model.score(X_test, y_test)
        accuracies[model_name] = accuracy
        
    return accuracies


# In[21]:


accuracies = model_accuracy(X_train, y_train, X_test, y_test)
print(accuracies)


# In[22]:


plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color=['orange', 'green', 'lightblue'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison: Accuracy on Test Data')  
plt.ylim(0,1) # Sets the y-axis limit from 0 to 1 (since accuracy ranges from 0 to 1)
plt.show()


# Even though our knn model is the lowest accuracy, let's try to tune it's hyperparameters and see how things would go
# 

# In[23]:


train_scores = []
test_scores = []

# Create a list of differnt values for n_neighbors
neighbors = range(1, 21)

# Setup KNN instance
knn = KNeighborsClassifier()

# Loop through different n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)
    
    # Fit the algorithm
    knn.fit(X_train, y_train)
    
    # Update the training scores list
    train_scores.append(knn.score(X_train, y_train))
    
    # Update the test scores list
    test_scores.append(knn.score(X_test, y_test))


# In[24]:


train_scores


# In[25]:


test_scores


# In[26]:



plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


# even after tunning the model still gives the same accuracy and it's still less than the other models without any tunning to them 
# so we will drop it.

# In[27]:


## let's tune our two other models
## after doing some research I found that the liblinear solver is better with smaller datasets and we have 303 rows here

lr_grid= {"C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],"solver":['liblinear']}

rf_grid={'n_estimators': np.arange(10,1000,50),
         'max_depth':[None,5,10,20],
         'min_samples_split': [2, 5, 10],
         'min_samples_leaf':[5,10,20],
         'max_features': ['auto', 'sqrt', 'log2']}


# In[28]:


from sklearn.model_selection import RandomizedSearchCV


# In[29]:


np.random.seed(42)
lr_rs=RandomizedSearchCV(LogisticRegression(),param_distributions=lr_grid,n_iter=7,verbose=True,cv=5)


# In[30]:


lr_rs.fit(X_train,y_train)


# In[31]:


lr_rs.best_params_


# In[32]:


lr_rs.score(X_test,y_test)


# In[33]:


np.random.seed(40)
rf_rs=RandomizedSearchCV(RandomForestClassifier(), param_distributions=rf_grid ,n_iter=19 ,verbose=True, cv=5)


# In[34]:


rf_rs.fit(X_train,y_train)


# In[35]:


rf_rs.best_params_


# In[36]:


rf_rs.score(X_test,y_test)


# In[37]:


accuracies


# our model got about 1.7% higher but still our logistic regression model out performs it with it's original accuracy, so we will also drop it

# let's try to improve our LogisticRegression model with GridSearchCV

# In[38]:


from sklearn.model_selection import GridSearchCV


# In[39]:


lr_gr = GridSearchCV(LogisticRegression(),param_grid=lr_grid,cv=5,verbose=True,scoring="accuracy")


# In[40]:


lr_gr.fit(X_train,y_train)


# In[41]:


lr_gr.score(X_test,y_test)


# In[43]:


y_preds= lr_gr.predict(X_test)


# the model performed very well with an accuracy of more than 90% with it's default hyperparameters without any tunning 

# In[45]:


from sklearn.metrics import confusion_matrix

sns.set(font_scale=1.5) # Increase font size
 
def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig = plt.subplots(figsize=(3, 3))
    sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True, # Annotate the boxes
                     cbar=False)
    plt.xlabel("Predicted label") # predictions go on the x-axis
    plt.ylabel("True label") # true labels go on the y-axis 
    
plot_conf_mat(y_test, y_preds)


# In[50]:


from sklearn.metrics import RocCurveDisplay


# In[54]:


roc_display = RocCurveDisplay.from_estimator(lr_gr, X_test, y_test)
plt.show()


# In[55]:


from sklearn.metrics import classification_report


# In[56]:


print(classification_report(y_test,y_preds))


# In[57]:


from sklearn.model_selection import cross_val_score


# In[60]:


lr_gr.best_params_


# In[61]:


clf=LogisticRegression(C= 0.1, solver ='liblinear')


# In[63]:


cv_accuracy = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="accuracy")
cv_accuracy


# In[68]:


cv_accuracy_mean = np.mean(cv_accuracy )
cv_accuracy_mean


# In[64]:


cv_precision =cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="precision")
cv_precision


# In[70]:


cv_precision_mean = np.mean(cv_precision)


# In[65]:


cv_recall = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="recall")
cv_recall = np.mean(cv_recall)
cv_recall


# In[66]:


cv_f1 = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="f1")
cv_f1 = np.mean(cv_f1)
cv_f1


# In[77]:


numbers = [cv_accuracy_mean, cv_precision_mean, cv_recall, cv_f1]
names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Plot the numbers with their names
plt.figure(figsize=(8, 5))
plt.bar(names, numbers, color='salmon')


plt.show()


# ## Feature Importance
# which features affected most the outcomes of the model and how did they affect it.
# 

# In[78]:


clf=LogisticRegression(C= 0.1, solver ='liblinear')
clf.fit(X_train,y_train)


# In[81]:


clf.coef_


# In[84]:


dict(zip(heart_disease.columns, clf.coef_[0]))


# In[90]:


feature_df = pd.DataFrame(dict(zip(heart_disease.columns, clf.coef_[0])), index=[0])
feature_df.T.plot.bar(title="Feature Importance", legend=False);


# In[ ]:




