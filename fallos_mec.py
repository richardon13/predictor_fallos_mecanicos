import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
sns.set()

# Carga de la base de datos cruda
df = pd.read_csv('./in/predictive_maintenance.csv')
print(df.head())
print('='*140)


# 1) Agrupe las máquinas por tipo y por tipo de fallo

df1 = df[['Type', 'Failure Type']]
print(df1)
print('='*140)

# 2) Listado de las máquinas, ordenándolas por la cantidad de rotaciones por minuto(rpm)

df2 = df[['Rotational speed [rpm]']].sort_values('Rotational speed [rpm]', ascending = False)
print(df2)
print('='*140)

# 3) Halle la cantidad de máquinas con fallas que tienen una temperatura en proceso mayor a la indicada por el usuario

temp = float(input('Indique la temperatura en proceso [K] de la maquina: ' ))

cant = len(df[(df['Process temperature [K]'] > temp) & (df['Failure Type'] != 'No Failure') | (df['Target'] == 1)].sort_values('Process temperature [K]', ascending = True))

print(f'Hay {cant} máquinas con fallas que tienen una temperatura en proceso mayor a {temp}')
print('='*140)

# 4) Halle la cantidad de máquinas por cada tipo e indique cuál tipo de máquina tiene un mayor conteo de máquinas 
# que presentan algún fallo

df4 = df['Type'].value_counts().head(1)
print(df4)

df4_fail = df.drop(df.index[(df['Failure Type'] == 'No Failure') & (df['Target'] == 0)])
print(df4_fail)

cant = df4_fail['Type'].value_counts().head(1)
print(cant)
print('='*140)

# 5) Halle rpm promedio de maquina por tipo de fallo (incluido el caso en el que no hay fallo), 
# haga una relación respecto a que tipo de fallo se ve relacionado con esta variable **************************

df5 = df.groupby('Failure Type')['Rotational speed [rpm]'].mean().to_frame()
print(df5)
print('='*140)

# 6) Halle el promedio de la temperatura del aire en las máquinas que tienen fallas por disipación de calor

#pd.set_option('display.float_format', '{:,.1f}'.format)
df[df['Failure Type'] == 'Heat Dissipation Failure']['Air temperature [K]'].mean()

# DE MANERA GRÁFICA PRESENTE LOS SIGUIENTES REPORTES

# 7) Grafique Temperatura en proceso promedio, mínima y máxima de cada máquina, por cada tipo ************
#df7 = df.groupby(['type'])['process_temperature_[k]'].mean()
df7 = df.groupby(['Type'])[['Process temperature [K]']].agg(['mean', 'min', 'max'])
df7=df7.reset_index()
X = list(df7['Process temperature [K]'].columns)
H= list(df7.iloc[0][1:])
L= list(df7.iloc[1][1:])
M= list(df7.iloc[2][1:])
fig, X_axis = plt.subplots()
X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, H, 0.2, label = 'H')
plt.bar(X_axis + 0.0, L, 0.2, label = 'L')
plt.bar(X_axis + 0.2, M, 0.2, label = 'M')
  
plt.xticks(X_axis, X)
plt.xlabel("group Aggregates")
plt.ylabel("Temp Values")
plt.title("Set of Machines")
plt.legend()
plt.show()

# 8) Porcentaje de máquinas por tipo

df8 = df['Type'].value_counts()
df8
#Explodes
explode = (0.0, 0.1, 0.2)
# Wedge properties
wp = { 'linewidth' : 1, 'edgecolor' : "blue" }

fig, ax = plt.subplots(figsize = (20, 9))
ax.pie(df8.values, labels = df8.index, autopct="%0.1f %%",explode = explode,shadow = True)
plt.title(label="Percentage of Machines by Type",fontsize= 20, pad='0.0',fontstyle='italic')
ax.legend(df8.index,
          title ="Machine Types",
          loc ="center left",
          bbox_to_anchor =(0.9, 0, 0.5, 1.5))
plt.show

# 9) Porcentaje de máquinas por fallos

# Forma 1. Con pie

df9 = df['Failure Type'].value_counts()
df9
#Explodes
explode = (0.1, 0.3,0.3,0.3,0.3,0.3)

fig, ax = plt.subplots(figsize = (20, 9))
legendaPlus =  [f'{i}, {f/100:0.1f}%' for i, f in zip(df9.index, df9.values)]
ax.pie(df9.values,labels = None, autopct="%0.1f %%", explode = explode,shadow = False,pctdistance =1.1) #"
ax.legend(df9.values,
          labels=legendaPlus,
          title ="Machine Failure Types",
          loc ="center left",
          bbox_to_anchor =(0.9, 0, 0.5, 1.5))
plt.title("Percentage of Machines by Type Failure")

# Forma 2. Con bar

df9 = df['Failure Type'].value_counts(normalize=True)
new_df = df9.mul(100).rename('Percent').reset_index()
new_df

g = sns.catplot(x='index', y='Percent', kind='bar', data=new_df)
g.ax.set_ylim(0,100)
for p in g.ax.patches:
    txt = str(p.get_height().round(1)) + '%'
    txt_x = p.get_x()
    txt_y = p.get_height()
    g.ax.text(txt_x,txt_y,txt)
plt.title("Percentage of Machines by Type Failure")

# Forma 3. Con bar excluyendo los valores de la variable 'Failure Type' = 'No Failure'

df9 = df[df['Failure Type']!='No Failure']['Failure Type'].value_counts(normalize=True)
new_df = df9.mul(100).rename('Percent').reset_index()
g = sns.catplot(y='index', x='Percent', kind='bar', data=new_df)
g.ax.set_xlim(0,50)
g.set(xlabel = "Percent", ylabel = "Failure Name")
for p in g.ax.patches:
    width = p.get_width()    # get bar length
    g.ax.text(width + 1,       # set the text at 1 unit right of the bar
            p.get_y() + p.get_height() / 2, # get Y coordinate + X coordinate / 2
            '{:1.1f}'.format(width) + '%', # set variable to display, 2 decimals
            ha = 'left',   # horizontal alignment
            va = 'center')  # vertical alignment
plt.title("Percentage of Machines by Type Failure")

# 10) Correlación entre las características de las máquinas

corr_values = df.corr().unstack() # Muestra la correlacion entre cada una de las varibles
corr_values.sort_values(kind = 'quicksort', ascending = False)# Muestra la correlacion entre cada una de las 
                                                              # varibles en orden descendente
# 10) Correlación entre las características de las máquinas

sns.heatmap(df.corr(), annot = True, annot_kws={"size": 8},fmt='.2f',cmap = 'coolwarm')
sns.set(rc={'figure.figsize':(8,8)})
plt.title('Seaborn heatmap - Correlation between Variables',fontsize= 15, fontweight='bold', pad='50.0',fontstyle='italic')

# Como la gráfica de mapa de calor muestra básicamente correlación entre [Torque] y [Rotational speed [rpm]], a continuacion se muestra un sns.pairplot 
# con estas dos columnas.

def centenas(valor):
  valor = (valor//100)*100+50
  return valor
df['Rotational speed [rpm]']=df['Rotational speed [rpm]'].apply(centenas)
df.tail()

sns.pairplot(df, vars=['Rotational speed [rpm]','Torque [Nm]'], palette='paired') #, hue='categorical'
plt.title('Seaborn pairplot - Correlation ',fontsize= 15, fontweight='bold', pad='0.0',fontstyle='italic')

## MACHINE LEARNING

# Se crea una columna llamada 'Fail' de tipo categorica y es la que contiene las etiquetas de las maquinas buenas y malas

for i in range(0, len(df)):
  if (df.loc[i, 'Failure Type'] != 'No Failure') | (df.loc[i, 'Target'] == 1):
    df.loc[i, 'Fail'] = 1
  else:
    df.loc[i, 'Fail'] = 0

df['Fail'] = df['Fail'].astype('int64')
print(df)
print('='*140)

df_new = df[['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Fail']]
print(df_new)
print('='*140)

# Convierto los valores de la variable categorica Type (string) en int

from sklearn.preprocessing import LabelEncoder

label_type = LabelEncoder()
df_new['Type'] = label_type.fit_transform(df_new['Type'])
df_new['Type'].unique()

# Solo trabajo con las columnas de interes

df_new_features = df_new[['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
df_new['Fail'] = df_new['Fail'].astype('int64')
x = df_new_features.values
y = df_new['Fail'].values
df_new

# Convierto la variable Fail en categorica

df['Fail'] = df['Fail'].astype('category')


# Determino cuantas maquinas estan fallando

print(y.shape)

# Aplico un modelo de Regresion Logística

from sklearn.linear_model import LogisticRegression

# Entreno el modelo

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .40, random_state = 42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Exactitud de:', metrics.accuracy_score(y_test, y_pred))
