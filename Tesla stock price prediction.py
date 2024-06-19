#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[21]:


file_path = r"C:\Users\Administrator\Desktop\MS Files\Tesla Stock price detection dataset.csv"
df = pd.read_csv(file_path)


# In[22]:


df = df.dropna()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# In[23]:


plt.figure(figsize=(9, 5))
plt.plot(df['Close'], label='Close Price', color='red')
plt.title('Tesla Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[24]:


df['Previous_Close'] = df['Close'].shift(1)
df = df.dropna()
X = df[['Previous_Close']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[26]:


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# In[27]:


plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual Price', color='purple')
plt.plot(y_test.index, y_pred, label='Predicted Price', color='yellow')
plt.title('Actual vs Predicted Tesla Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

