

accidents = 'accidents_labeled.csv'
cas_veh = 'veh_cas_labeled.csv'
path ='../../'
from sklearn.ensemble import RandomForestClassifier
import matplotlib.dates as mdates
import datetime as dt
from matplotlib.font_manager import FontProperties
from datetime import datetime
from matplotlib import figure

from matplotlib import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = cm.get_cmap
import seaborn as sns
import altair as alt
import io
pd.set_option('display.max_columns', None)
from plotly.subplots import make_subplots
import streamlit as st
import cufflinks as cf
cf.go_offline()
cf.set_config_file(theme='pearl',sharing='public',offline=True)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

import plotly
import pandas as pd
import cufflinks as cf
import numpy as np
import pickle
plotly.offline.init_notebook_mode()
#DATA
import plotly.express as px

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os
myfile = open("path.txt",mode="r")

datapath,repopath = myfile.readlines()
datapath = datapath.strip("\n")

#dfa = pd.read_csv("../../accidents_labeled.csv",delimiter=',',encoding='UTF-8-SIG',index_col=0,parse_dates=["Date"])
#dfa = dfa.sample(1000)
#dfm = pd.read_csv("../../veh_cas_labeled.csv",delimiter=',',encoding='UTF-8-SIG',index_col=0,nrows=1000000)
empty = st.empty()

#@st.cache()
@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def get_data(datapath):
    import pandas as pd
    import random
    p = 0.001 # 1% of the lines
    # keep the header, then take only 1% of lines
    # if random from [0,1] interval is greater than 0.01 the row will be skipped
    dfa = pd.read_csv(
         datapath + "accidents_labeled.csv",
         header=0,index_col=0, 
         skiprows=lambda i: i>0 and random.random() > p,delimiter=',',encoding='UTF-8-SIG',parse_dates=["Date"])
    dfm = pd.read_csv(datapath + "veh_cas_labeled.csv",skiprows=lambda i: i>0 and random.random() > p,delimiter=',',encoding='UTF-8-SIG',index_col=0)
    
    return dfa,dfm

dfa,dfm = get_data(datapath)


######################################################################################################################
#GRAFICOS
######################################################################################################################
#@st.cache


def plot1(dfa):   ##scatterplot   
     
    fig1 = dfa.iplot(kind="scatter",
                      x='Number_of_Casualties',
                      y='Number_of_Vehicles',
                      categories="Accident_Severity",dimensions=(700,400),
                      asFigure=True,
                      xTitle="Number_of_Casualties",showgrid=True,
                      yTitle="Number_of_Vehicles",title="Scatterplot")


    return fig1

   
#############
#@st.cache
def plot2(dfa):
    
    fig2 = dfa['Accident_Severity'].value_counts(normalize=True)\
                            .reset_index().iplot(kind='pie',dimensions=(700,400),
                            labels='index',values='Accident_Severity',
                            textinfo='percent+label',hole=0.4,
                            color = ['lightgreen', 'orange','red'],title='Accident Severity Chart',
                            asFigure=True)
    return fig2


######################
#@st.cache
def plot3(dfa):
    
    sns.set_style('white')
    fig3, ax = plt.subplots(figsize=(14,8))
    dfa.set_index('Date').resample('M').size().plot(label='Total Month', color='grey', ax=ax)
    dfa.set_index('Date').resample('M').size().rolling(window=12).mean()\
    .plot(color='lightgreen', linewidth=5, label='Montly average 12 months', ax=ax)
    ax.set_title('Accidents per Month', fontsize=18, fontweight='bold')
    ax.set(ylabel='Total Count\n', xlabel='Years')
    ax.legend(fontsize=10)
    ax.set_xlabel('Year',fontsize=15)
    ax.set_ylabel('Total counts\n',fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #ax.set_xticklabels(["1979", "1984", "1989", "1994","1999","2004"], fontsize=12)
    #ax.set_yticklabels(["16k", "18k", "20k", "22k","24k"], fontsize=12)
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=False);
    return fig3



#####################
#@st.cache
def plot4(dfa):
    
    df1 = dfa.groupby(['Year'])\
    .agg({'Accident_Index':'count', 'Number_of_Vehicles': 'sum','Number_of_Casualties': 'sum',})\
    .reset_index()
    sns.set_style("white")
    x = df1.Year
    labels = df1.Year
    width = 0.5
    Accidentcounts = df1['Accident_Index']
    Casualtycounts =  df1['Number_of_Casualties']
    fig4,ax =  plt.subplots(figsize=(14,8))
    ax.bar1 = ax.bar(x - width/2, Accidentcounts, width, label='Accident counts', color = 'paleturquoise');
    ax.bar2 = ax.bar(x + width/2, Casualtycounts, width, label='Casualty counts', color = 'slategrey');
    #ax.bar1[10].set_color('moccasin')
    #ax.bar2[10].set_color('lightcoral')
    ax.legend(fontsize=10)
    ax.set_title('\nAccidents / Casualties \n per Year\n', fontsize=18, fontweight='bold')
    ax.set_xlabel('\nYear',fontsize=15)
    ax.set_ylabel('Total counts\n',fontsize=15)
    ax.legend(fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_xticks(x)
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=False);
    return fig4

def plot5(dfa): # ACCIDENTS PER MONTH
    
    df2 = dfa.groupby(['Month'])['Accident_Index'].count().reset_index()
    months = ['January', 'February','March','April', 'May','June','July','August', 'September','October','November','December']
    df2['Month'] = pd.Categorical(df2['Month'], categories=months, ordered=True)
    #df2.sort_values(...)  # same as you have now; can use inplace=True
    df2 = df2.sort_values(by='Month')
    sns.set_style("white")
    x = df2['Month']
    y = df2['Accident_Index']
    fig5, ax =  plt.subplots(figsize=(14,8))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    bar1 = ax.bar(x,y,color='cornflowerblue',linewidth=4)
    #bar1[9].set_color('tomato')
    ax.set_title('Accidents per Month', fontsize=18, fontweight='bold')
    ax.set_xlabel('\n Month',fontsize=15)
    ax.set_ylabel('Total Count\n',fontsize=15)
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=False);

    return fig5

def plot6(dfa): # ACCIDENTS PER WEEK
    df3 = dfa.groupby(['Day_of_Week'])['Accident_Index']\
                .count()\
                .sort_values(ascending=False)\
                .reset_index()
    days = ['Sunday',
            'Monday',
            'Tuesday',
            'Wednesday',
            'Thursday',
            'Friday',
            'Saturday']

    df3['Day_of_Week'] = pd.Categorical(df3['Day_of_Week'],
                                        categories=days, 
                                        ordered=True)

    df3 = df3.sort_values(by='Day_of_Week',ascending=True)


    # sns.set_style('white')
    fig6, ax = plt.subplots(figsize=(14,8))

    barlist = plt.bar(df3['Day_of_Week'],df3['Accident_Index'],
                      color='orange')

    ax.set_title('\nAccidents per Weekday\n',
                 fontsize=14,
                 fontweight='bold')

    ax.set_xlabel('\n Weekday',fontsize=15)
    ax.set_ylabel('Total Count\n',fontsize=15)
    # remove all spines
    sns.despine(ax=ax, top=True, 
                right=True, 
                left=True, 
                bottom=False);

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    return fig6

def plot7(dfa):  #HEATMAP WEEKDAY DAYTIME
    df4 = dfa.groupby(['Day_of_Week','Daytime'])['Accident_Index'].count().reset_index()
    days = ['Sunday',
            'Monday',
            'Tuesday',
            'Wednesday',
            'Thursday',
            'Friday',
            'Saturday']
    df4['Day_of_Week'] = pd.Categorical(df4['Day_of_Week'], categories=days, ordered=True)
    df4 = df4.pivot(index='Day_of_Week', columns='Daytime', values='Accident_Index')
    fig7 = df4.iplot(kind="heatmap",
                    colorscale="Reds",
                    dimensions=(700,500),
                    title='Heatmap Daytime vs Weekday',
                    asFigure=True)
    fig7.update_layout(legend=dict(
        yanchor="top",
        y=1.15,
        xanchor="left",
        x=0.85,
            ),
    title_x=0.50)
    return fig7

def plot8(dfa): ##altair graphic
    
    import altair as alt
    df5 = dfa.groupby([
        'Day_of_Week',
        'Accident_Severity',
        'Light_Conditions_2'
    ])['Number_of_Casualties'].sum().reset_index()

    days = ['Sunday',
            'Monday',
            'Tuesday',
            'Wednesday',
            'Thursday',
            'Friday',
            'Saturday']

    df5['Day_of_Week'] = pd.Categorical(df5['Day_of_Week'],
                                        categories=days, 
                                        ordered=True
                                       )

    #source = df5

    c=alt.Chart(df5).mark_circle().encode(
        alt.X('Number_of_Casualties',scale=alt.Scale(zero=False)),
        alt.Y('Day_of_Week',sort=days,scale=alt.Scale(zero=False, padding=1)),
        color='Accident_Severity',
        size='Light_Conditions_2'
    ).properties(
        width=900,
        height=400,
        title='Acc_Severity by Day_of_Week and Daylight').interactive()
    return c
        
def plot9(dfa):
    fig9 = dfa.Hour.iplot(kind='histogram',
                           bins=40,
                           theme="white",
                           title="Accidents by Time",
                           dimensions=(700,400),
                           xTitle='Hour of the Day',
                           yTitle='Acc Count',
                           colors="darkseagreen",asFigure=True)
    fig9.update_layout(title_x=0.5,title_y=0.85)
    return fig9

def plot10(dfa):
    dfm['Male'] = dfm[dfm["Sex_of_Casualty"]=='Male']["Acc_Index"]
    dfm['Female'] = dfm[dfm["Sex_of_Casualty"]=='Female']["Acc_Index"]

    fig10=dfm.groupby(['Age_Band_of_Casualty']).agg({"Male":"count","Female":"count"})\
                                                  .reset_index()\
                                                  .sort_values(by="Age_Band_of_Casualty",ascending=True)\
                                                  .iplot(
                                                    kind="bar",
                                                    barmode="stack",
                                                    x="Age_Band_of_Casualty",
                                                    y=["Male","Female"],
                                                    colors=["lightgreen","lightpurple"],
                                                    title="Accidents by casualty Gender",
                                                    dimensions=(700,400),asFigure=True
    )
    return fig10

def plot11(dfm):

    driver = dfm.groupby(['Age_Band_of_Driver','Sex_of_Driver'])['Acc_Index']\
                     .count().reset_index()


    driver['%'] = (driver['Acc_Index']/driver['Acc_Index'].sum()*100)\
                                                          .sort_values(ascending=True).round(2)

    driver['%'] = driver['%'].astype(str).str[:4]+'%'

    import plotly.express as px

    fig11 = px.sunburst(driver,
                       path=['Sex_of_Driver','Age_Band_of_Driver','%'],
                       color='Sex_of_Driver')

    fig11.update_layout(height=500,
                       width=550,
                       title_text="Age_Band_of_Driver vs Sex of Driver",
                      title_x=0.5)
    return fig11


    
###########################################################################################################################   

menu=['Home','Visualizacion','Modelado']
choice=st.sidebar.selectbox('Menu',menu)
if choice=='Home':
    st.title("FRONT-END - UK SAFETY CAR")
    st.markdown(
    """
    Esta aplicación ha sido diseñada para visualizar los rasgos más comunes de un dataset basado en accidentes de tráfico en el Reino Unido, durante el período 1979 hasta 2004.
    Como también realizar predicciones sobre un dataset de test y comprobar el resultado.
    El dataset ha sido descargado del siguiente [enlace](https://academictorrents.com/details/c7d2d7a91ae3fd0256dd2ba2d7344960cb3c4dbb).
    """
    )
    # Mostrar tablas de datos
    st.subheader("Datos utilizados")
    
    st.write('Accidents dataset')
    st.dataframe(dfa.head(5))
    st.write('Vehicles_Casualties dataset')
    st.dataframe(dfm.head(5))


if choice=='Visualizacion':
    st.title("Análisis y gráficos")
    years = list(range(1979,2005,1))
    years.insert(0,"General")
    
    
    status = st.radio("", ('Total', 'Anual'))
    level = st.slider("",1979,2004)
  
    #choice1=st.selectbox("options:",years)
    
#    for i in years:
    if status == "Anual":
        data = dfa[dfa['Year']==level]
    else:
        data=dfa
    option=st.selectbox('options',(
        'Scatterplot / Pie Chart',
        'Accidentes por Año / Mes / Semana',
        'Mapa de calor: Dia de la semana - Franjas horarias',
        'Accidentes ocurridos por horas del día',
        'Total de acidentes por edad de conductor / víctima'))

    if option =='Scatterplot / Pie Chart':
        fig1=plot1(data)
        fig2=plot2(data)
        st.plotly_chart(fig2)
        st.plotly_chart(fig1)
        
    elif option =='Accidentes por Año / Mes / Semana':
        #fig3=plot3(data)
        #fig5=plot5(data)
        st.pyplot(plot4(data))
        st.pyplot(plot3(data))
        st.pyplot(plot5(data))
        st.pyplot(plot6(data))
        
    elif option == "Mapa de calor: Dia de la semana - Franjas horarias":
        fig7=plot7(data)
        st.plotly_chart(fig7)
    elif option == "Accidentes ocurridos por horas del día":
        st.plotly_chart(plot9(data))
    elif option == 'Total de acidentes por edad de conductor / víctima':
        st.plotly_chart(plot10(data))
        st.plotly_chart(plot11(dfm))
        
        
if choice=='Modelado':
    st.title("Predicciones y resultados")
    test_data = st.file_uploader('Selecciona el archivo de Test',type='csv')
    
    if test_data:
        df = pd.read_csv(test_data,index_col=0)
        y_test = pd.Series(df.iloc[:,0])
      
        model_selected = st.radio('', ['BaggingClassifier','DecisionTreeClassifier','RandomForestClassifier'],0)

        if model_selected == 'BaggingClassifier':
            feature=0
            filename = "model/BaggingClassifier_model.pkl"
            feature_imp = 'model/import_features_BaggingClassifier.txt'
            
        elif model_selected == 'DecisionTreeClassifier':
            feature=1            
            filename = "model/DecisionTreeClassifier_model.pkl"
            feature_imp = 'model/import_features_DecisionTreeClassifier.txt'
            
        elif model_selected == 'KNeighborsClassifier':
            feature=1
            filename = "model/KNeighborsClassifier_model.pkl"
            feature_imp = 'model/import_features_KNeighborsClassifier.txt'
            
        elif model_selected == 'RandomForestClassifier':
            feature=1
            filename = "model/RandomForestClassifier_model.pkl"
            feature_imp = 'model/import_features_RandomForestClassifier.txt'
        
        
        model = joblib.load(filename)
        y_hat = model.predict(df.iloc[:,1:11]) 
        results =  pd.DataFrame(y_hat)
        #dfa = df.iloc[:,1:13]
        dfa = df
        results = results.join(dfa)
        results.rename(columns={0:"Accident_Severity_predicted","Latitude":'lat','Longitude':'lon'},inplace=True)
        
        #PLOTS
        fig1,ax = plt.subplots(1,2,figsize=(15,5.5))
        #fig1.tight_layout()
        #plt.subplots_adjust(top=0.85)
        plt.subplots_adjust(wspace=1000, hspace=1000)

        # roc curve for classes
        n_class = 3
        fpr = {}
        tpr = {}
        thresh ={}
        probs = model.predict_proba(df.iloc[:,1:11])
        for i in range(n_class):    
            fpr[i], tpr[i], thresh[i] = roc_curve(y_test, probs[:,i], pos_label=i)
        #plotting Roc Curve     
        plt.subplot(121)
        #plt.subplots_adjust(wspace=10, hspace=10)
        plt.plot(fpr[0], tpr[0], linestyle='--',color='red', label='Fatal')
        plt.plot(fpr[1], tpr[1], linestyle='--',color='orange', label='Serious')
        plt.plot(fpr[2], tpr[2], linestyle='--',color='green', label='Slight')
        plt.title('Multiclass ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        #plt.savefig(folder+'/'+ modelts+'Multiclass ROC',dpi=300);
        #roc_auc_score(y_test, probs,multi_class='ovo', average='weighted')


        matrix = confusion_matrix(y_test, y_hat)
        dataframe = pd.DataFrame(matrix, index=['Fatal', 'Serious', 'Slight'], 
                                    columns=['Fatal', 'Serious', 'Slight'])
        # create matrixconfusion heatmap
        #fig,ax = plt.subplots()
        plt.subplot(122)
        sns.heatmap(dataframe, annot=True, cbar=None, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.tight_layout(), plt.xlabel('True Values'), plt.ylabel('Predicted Values')
        plt.show()
        
        with open(feature_imp , mode="r") as file:
            final_features_list = file.read().replace('"',"")\
                                             .replace("'","")\
                                                .replace("[","")\
                                                .replace("]","")\
                                            .split(",")

        st.pyplot(fig1)
        fig2,ax = plt.subplots()
        if feature == 1:
            feat_importances = pd.Series(model.steps[1][1].feature_importances_, index=final_features_list)
            feat_importances.nlargest(n=20).sort_values(ascending=True).plot(kind='barh',color='Orange',width=0.3, figsize=(15,7))
            plt.xlabel('Relative Feature Importance');

            st.pyplot(fig2)
            
            
        st.write('Predicción')
        st.map(results[results['Accident_Severity_predicted']==0][['lat','lon']])
        st.write('Valor real')
        st.map(results[results['Accident_Severity']==0][['lat','lon']])
        
        #st.dataframe(results['Accident_Severity_predicted'].head(20))
        #st.dataframe(results.head(10))
    else:
        pass
