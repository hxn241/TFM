
accidents = 'accidents_clean.csv'
cas_veh = 'df_merged_cat.csv'
path ='../../'

import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = cm.get_cmap
import seaborn as sns

import io
pd.set_option('display.max_columns', None)
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import cufflinks as cf
cf.set_config_file(theme='pearl',sharing='public',offline=True)

st.title("Accidents Visualization and Predictions")
st.markdown(
"""
This app is for visualizing the Accidents data for UK which is collected 
from the Academictorrents site https://academictorrents.com/details/c7d2d7a91ae3fd0256dd2ba2d7344960cb3c4dbb.

User can view EDA and predictions for every year and also for a global view.
"""
)

# Mostrar tablas de datos
st.subheader("Datos utilizados")
dfa = pd.read_csv(path + accidents,delimiter=',',encoding='UTF-8-SIG',index_col=0,nrows=100000)
dfm = pd.read_csv(path + cas_veh,delimiter=',',encoding='UTF-8-SIG',index_col=0,nrows=100000)

st.dataframe(dfa.head(5))
st.dataframe(dfm.head(5))

fig = dfa['Accident_Severity'].value_counts(normalize=True)\
                    .reset_index().iplot(kind='pie',
                    labels='index',values='Accident_Severity',
                    textinfo='percent+label',hole=0.4,
                    color = ['lightgreen', 'orange','red'],title='Accident Severity Chart',
                    asFigure=True,   )                
                                        
fig.update_layout(legend=dict()

                 
                 )
st.plotly_chart(fig)


