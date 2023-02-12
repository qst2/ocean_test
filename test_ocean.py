%%writefile file_name.py
import os, sys
import numpy as np
import seaborn as sb
import  pandas as pd
os.getcwd()
# !python -m pip install prophet

df = pd.read_csv('ocean.csv',index_col=None)
df.DATA = pd.to_datetime(df.DATA, dayfirst = True)
# df.DATA


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
imp_mean = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
# df[['01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h',
#        '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h',
#        '19h', '20h', '21h', '22h', '23h', '24h']] = pd.DataFrame(imp_mean.fit_transform(df[['01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h',
#        '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h',
#        '19h', '20h', '21h', '22h', '23h', '24h']]))
df['month'] = df['DATA'].dt.month_name()
df['year'] = pd.to_datetime(df['DATA']).dt.year


