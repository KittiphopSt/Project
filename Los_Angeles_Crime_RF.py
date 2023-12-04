#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import random


# In[2]:


df = pd.read_csv(r'C:\Users\Admin\Desktop\LA\Crime_Data_from_2010_to_2019.csv')
df


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


for i in range(len(df.columns)):
  print(f'{df.columns[i]} : {df.iloc[:,i].nunique()}')


# In[6]:


del df['DR_NO']
del df['Mocodes']
del df['Cross Street']


# In[7]:


df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
df['Date Rptd'] = pd.to_datetime(df['Date Rptd'])


# In[8]:


df['TIME OCC'].value_counts()


# In[9]:


df['TIME OCC'] = df['TIME OCC'].apply(lambda x: str(x).zfill(4))
df['TIME OCC'] = pd.to_datetime(df['TIME OCC'], format='%H%M').dt.strftime('%H:%M')


# In[10]:


df['Vict Sex'] = df['Vict Sex'].map({'M' : 'M','F' : 'F','nan' : np.nan,'H' : np.nan,'-' : np.nan, 'N' : np.nan,'X': np.nan})
print(df['Vict Sex'].value_counts())


# In[11]:


df['Vict Descent'].value_counts()


# In[12]:


df['Vict Descent'] = df['Vict Descent'].map({'H':'H','W':'W','B':'B','O':'O','X':'X','A':'A','K':'K','F':'F','C':'C','I':'I','J':'J','P':'P','V':'V','U':'U','Z':'Z','G':'G','S':'S','D':'D','L':'L','-':np.nan})
print(df['Vict Descent'].value_counts())
df['Vict Descent'].fillna('N',inplace=True)
df['Vict Descent'] = df['Vict Descent'].apply(lambda x: x if x != 'N' else 'N')
df['Vict Descent'].value_counts()


# In[13]:


dict(sorted(df['Vict Age'].value_counts().to_dict().items()))


# In[14]:


for i in range(len(df['Vict Age'])):
  if df['Vict Age'][i] < 1 or df['Vict Age'][i] > 100:
    df['Vict Age'][i] = None


# In[15]:


c = df[df[['Premis Desc']].isnull().any(axis=1)][['Premis Desc','Premis Cd']]
print(c)
print(c['Premis Cd'].value_counts())


# In[16]:


print(df[(df['Premis Cd'] == 418) | (df['Premis Cd'] == 256) | (df['Premis Cd'] == 838)]['Premis Desc'])
print(df[(df['Premis Cd'] == 418) | (df['Premis Cd'] == 256) | (df['Premis Cd'] == 838)]['Premis Desc'].value_counts())


# In[17]:


print(df[(pd.isnull(df['Weapon Desc'])) & (pd.notnull(df['Weapon Used Cd']))][['Weapon Desc','Weapon Used Cd']])
print(df[df['Weapon Used Cd'] == 222][['Weapon Desc','Weapon Used Cd']])
print(df['Weapon Used Cd'].value_counts())
df['Weapon Used Cd'] = df['Weapon Used Cd'].apply(lambda x: np.nan if x == 222 else x)
df['Weapon Used Cd'].fillna(0,inplace=True)
df['Weapon Used Cd'] = df['Weapon Used Cd'].apply(lambda x: 1 if x != 0 else 0)
print(df['Weapon Used Cd'].value_counts())
del df['Weapon Desc']


# In[18]:


df[df[['Status']].isnull().any(axis=1)][['Status','Status Desc']]


# In[19]:


print(df['Status Desc'].value_counts())
print(df['Status'].value_counts())
for i in range(len(df['Status'])):
  if df['Status'][i] in ['CC','TH','13','19']:
    df['Status'][i] = 'UNK'
df.drop(df[df['Status'] == 'UNK'].index, inplace=True)
print(df['Status'].value_counts())


# In[20]:


df[df[['Crm Cd 1']].isnull().any(axis=1)]


# In[21]:


df['Crm Cd'].value_counts().to_dict()


# In[22]:


df['Crm Cd 1'].fillna(0,inplace=True)
df['Crm Cd 1'] = df['Crm Cd 1'].apply(lambda x : 1 if x != 0 else 0)
df['Crm Cd 2'].fillna(0,inplace=True)
df['Crm Cd 2'] = df['Crm Cd 2'].apply(lambda x : 1 if x != 0 else 0)
df['Crm Cd 3'].fillna(0,inplace=True)
df['Crm Cd 3'] = df['Crm Cd 3'].apply(lambda x : 1 if x != 0 else 0)
df['Crm Cd 4'].fillna(0,inplace=True)
df['Crm Cd 4'] = df['Crm Cd 4'].apply(lambda x : 1 if x != 0 else 0)


# In[23]:


df.isnull().sum()


# In[24]:


df = df.dropna(axis=0)
df


# In[25]:


df.drop(df[(df['LAT'] == 0) | (df['LON'] == 0)].index,inplace=True)


# In[26]:


df.isnull().sum()


# In[27]:


df['Year OCC'] = df['DATE OCC'].dt.strftime('%Y')
df['Year OCC'] = df['Year OCC'].astype(int)
df['Year Rptd'] = df['Date Rptd'].dt.strftime('%Y')
df['Year Rptd'] = df['Year Rptd'].astype(int)


# In[28]:


for i in range(len(df.columns)):
  print(f'{df.columns[i]} : {df.iloc[:,i].nunique()}')


# In[29]:


premis_cd = []
premis_desc = []
for i in df['Premis Desc'].value_counts().to_dict():
  premis_desc.append(df['Premis Desc'].value_counts().to_dict()[i])
for i in df['Premis Cd'].value_counts().to_dict():
  premis_cd.append(df['Premis Cd'].value_counts().to_dict()[i])
[(cd, desc) for cd, desc in zip(premis_cd, premis_desc)]


# In[30]:


df[df['Premis Desc'] == 'RETIRED (DUPLICATE) DO NOT USE THIS CODE'][['Premis Cd','Premis Desc']]


# In[31]:


df['Premis Cd'] = df['Premis Cd'].replace([226, 803], 805)


# In[32]:


df.describe().round(2)


# In[33]:


fig = px.line(df.groupby('Year OCC')['Crm Cd'].count().reset_index(),x='Year OCC',y='Crm Cd',title='Crime Code Trend by Year',labels={'Year OCC': 'Year OCC', 'Crm Cd': 'Crime Count'},)
fig.update_xaxes(title_text='Year OCC')
fig.update_yaxes(title_text='Crime Count')
fig.show()


# In[34]:


fig = px.line(df.groupby('Year Rptd')['Crm Cd'].count().reset_index(),x='Year Rptd',y='Crm Cd',title='Crime Code Report Trend by Year',labels={'Year Rptd': 'Year Rptd', 'Crm Cd': 'Crime Count'},)
fig.update_xaxes(title_text='Year OCC')
fig.update_yaxes(title_text='Crime Count')
fig.show()


# In[35]:


fig = px.pie(names=df['Status Desc'].value_counts().index,values=df['Status Desc'].value_counts().values,title='Crime Status Distribution',)
fig.update_traces(textinfo='percent+label', showlegend=True,textposition='outside')
fig.show()


# In[36]:


fig = px.histogram(df, x='Vict Age', title='Count Plot of Victim Age',color='Vict Age',category_orders={'Vict Age': df['Vict Age'].sort_values().unique()})
fig.update_traces(texttemplate='%{y}', textposition='outside')
fig.show()


# In[37]:


fig = px.histogram(df, x='Vict Descent', title='Count Plot of Vict Descent',color='Vict Descent',)
fig = fig.update_xaxes(categoryorder='total descending')
fig.update_traces(texttemplate='%{y}', textposition='outside')
fig.show()


# In[38]:


fig = px.histogram(df, x='Vict Sex', title='Count Plot of Vict Sex',color='Vict Sex',)
fig = fig.update_xaxes(categoryorder='total descending')
fig.update_traces(texttemplate='%{y}', textposition='outside')
fig.show()


# In[39]:


fig = px.pie(names=df['AREA NAME'].value_counts().index,values=df['AREA NAME'].value_counts().values,title='AREA NAME Distribution',)
fig.update_traces(textinfo='percent+label', showlegend=True,textposition='outside')
fig.show()


# In[40]:


trace_date_occ = go.Scatter(x=df.groupby('DATE OCC')['Crm Cd'].count().reset_index()['DATE OCC'], y=df.groupby('DATE OCC')['Crm Cd'].count().reset_index()['Crm Cd'], name='DATE OCC', line=dict(color='blue'))
trace_date_rptd = go.Scatter(x=df.groupby('Date Rptd')['Crm Cd'].count().reset_index()['Date Rptd'], y=df.groupby('Date Rptd')['Crm Cd'].count().reset_index()['Crm Cd'], name='Date Rptd', line=dict(color='red'))
fig = go.Figure(data=[trace_date_occ, trace_date_rptd])
fig.update_layout(title='Crime Code Trend by Date OCC and Date Rptd',xaxis_title='DATE OCC / Date Rptd',yaxis_title='Crime Count')
fig.show()


# In[41]:


for i in [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]:
  a = df[df['Year OCC'] == i]
  trace_date_occ = go.Scatter(x=a.groupby('DATE OCC')['Crm Cd'].count().reset_index()['DATE OCC'], y=a.groupby('DATE OCC')['Crm Cd'].count().reset_index()['Crm Cd'], name='DATE OCC', line=dict(color='blue'))
  trace_date_rptd = go.Scatter(x=a.groupby('Date Rptd')['Crm Cd'].count().reset_index()['Date Rptd'], y=a.groupby('Date Rptd')['Crm Cd'].count().reset_index()['Crm Cd'], name='Date Rptd', line=dict(color='red'))
  fig = go.Figure(data=[trace_date_occ, trace_date_rptd])
  fig.update_layout(title=f'Crime Code Trend by Date OCC and Date Rptd in {i}',xaxis_title='DATE OCC / Date Rptd',yaxis_title='Crime Count')
  fig.show()


# In[42]:


fig = px.histogram(df, x='Year OCC', title='Count plot of Crm Cd by Year', color='Crm Cd',category_orders={'Year OCC': df['Year OCC'].sort_values().unique()})
fig.update_traces(texttemplate='%{y}', textposition='outside')
fig.show()


# In[43]:


df['Type of Crime'] = df['Crm Cd'].map({110:'HOMICIDE',113:'HOMICIDE',
                                        121:'RAPE',122:'RAPE',815:'RAPE',820:'RAPE',821:'RAPE',860:'RAPE',810:'RAPE',850:'RAPE',762:'RAPE',822:'RAPE',845:'RAPE',814:'RAPE',840:'RAPE',830:'RAPE',
                                        210:'ROBBERY',220:'ROBBERY',430:'ROBBERY',431:'ROBBERY',433:'ROBBERY',888:'ROBBERY',
                                        230:'AGG. ASSAULTS',231:'AGG. ASSAULTS',235:'AGG. ASSAULTS',946:'AGG. ASSAULTS',812:'AGG. ASSAULTS',813:'AGG. ASSAULTS',237:'AGG. ASSAULTS',910:'AGG. ASSAULTS',922:'AGG. ASSAULTS',920:'AGG. ASSAULTS',921:'AGG. ASSAULTS',522:'AGG. ASSAULTS',944:'AGG. ASSAULTS',931:'AGG. ASSAULTS',
                                        900:'VIOLATION ORDER',901:'VIOLATION ORDER',902:'VIOLATION ORDER',903:'VIOLATION ORDER',806:'VIOLATION ORDER',890:'VIOLATION ORDER',
                                        236:'DV',250:'DV',251:'DV',761:'DV',926:'DV',626:'DV',627:'DV',647:'DV',763:'DV',928:'DV',930:'DV',740:'DV',745:'DV',956:'DV',940:'DV',648:'DV',932:'DV',666:'DV',653:'DV',755:'DV',660:'DV',943:'DV',805:'DV',949:'DV',651:'DV',654:'DV',760:'DV',954:'DV',756:'DV',870:'DV',865:'DV',948:'DV',884:'DV',952:'DV',
                                        435:'SM ASSAULTS',436:'SM ASSAULTS',437:'SM ASSAULTS',622:'SM ASSAULTS',623:'SM ASSAULTS',624:'SM ASSAULTS',625:'SM ASSAULTS',886:'SM ASSAULTS',753:'SM ASSAULTS',438:'SM ASSAULTS',882:'SM ASSAULTS',
                                        310:'BURGLARY',320:'BURGLARY',
                                        510:'MVT',520:'MVT',433:'MVT',
                                        330:'BTFV',331:'BTFV',410:'BTFV',420:'BTFV',421:'BTFV',
                                        350:'PERSONAL THFT',351:'PERSONAL THFT',352:'PERSONAL THFT',353:'PERSONAL THFT',354:'PERSONAL THFT',450:'PERSONAL THFT',451:'PERSONAL THFT',452:'PERSONAL THFT',453:'PERSONAL THFT',354:'PERSONAL THFT',668:'PERSONAL THFT',951:'PERSONAL THFT',933:'PERSONAL THFT',670:'PERSONAL THFT',950:'PERSONAL THFT',
                                        341:'OTHER THEFT',343:'OTHER THEFT',345:'OTHER THEFT',440:'OTHER THEFT',441:'OTHER THEFT',442:'OTHER THEFT',443:'OTHER THEFT',444:'OTHER THEFT',445:'OTHER THEFT',470:'OTHER THEFT',471:'OTHER THEFT',472:'OTHER THEFT',473:'OTHER THEFT',474:'OTHER THEFT',475:'OTHER THEFT',480:'OTHER THEFT',485:'OTHER THEFT',487:'OTHER THEFT',491:'OTHER THEFT',649:'OTHER THEFT',662:'OTHER THEFT',664:'OTHER THEFT',661:'OTHER THEFT',347:'OTHER THEFT',924:'OTHER THEFT',942:'OTHER THEFT',880:'OTHER THEFT',446:'OTHER THEFT',349:'OTHER THEFT',})


# In[44]:


df.drop(df[df[['Type of Crime']].isnull().any(axis=1)].index,inplace=True)


# In[45]:


df


# In[46]:


df = df.groupby('Crm Cd', group_keys=False).apply(lambda x: x.sample(2500, replace=True))
df


# In[47]:


del df['Date Rptd']
del df['DATE OCC']
del df['TIME OCC']
del df['Crm Cd Desc']
del df['AREA NAME']
del df['Rpt Dist No']
del df['Premis Desc']
del df['Premis Cd']
del df['Status Desc']
del df['LOCATION']
del df['Year OCC']
del df['Year Rptd']


# In[48]:


df = pd.get_dummies(df)
df


# In[49]:


df.columns


# In[50]:


X = df.drop(['Crm Cd'],axis=1)
y = df['Crm Cd']


# In[51]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[52]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[53]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100 , criterion = 'entropy', random_state = 42)
classifier.fit(X_train,y_train)


# In[54]:


y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


# In[55]:


print(classification_report(y_test,y_pred))


# In[56]:


y_pred_proba = classifier.predict_proba(X_test)
from sklearn.preprocessing import label_binarize
y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
roc_auc_per_class = [roc_auc_score(y_test_binarized[:, i], y_pred_proba[:, i]) for i in range(135)]
roc_auc_dict = {class_label: roc_auc for class_label, roc_auc in zip(np.unique(y_test), roc_auc_per_class)}
for class_label, roc_auc in roc_auc_dict.items():
    print(f"Class {class_label}: ROC-AUC = {roc_auc:.4f}")

