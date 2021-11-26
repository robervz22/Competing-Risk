
#Leemos los datos a partir de el CSV
import numpy as np
import pandas as pd
import string
import random
import statistics as st
import math as mt
import matplotlib.pyplot as plt
from numpy import hstack
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.integrate import quad
from lifelines import AalenJohansenFitter

#----------------------------------------------------------------------------------------------------
#Leemos los datos
np.random.seed(30)
random.seed(30)
df=pd.read_excel("logArt.ods",engine="odf")
#df=pd.read_csv('tlognorCR.csv')
T0=df['Tiempos'].values.tolist()
T=np.log(T0)
completCause=df['Causa'].values.tolist()
aux=df['Mascara'].values.tolist()
masking=[]
for i in range(len(T)):
    masking.append([int(k) for k in aux[i].strip("[]").split(",")])

#--------------------------------------------------------------------------------------------------
#Definimos las funciones que vamos a utilizar

#Funcion de densidad de la normal truncada
def W_s(mu_s,sigma_s):
    w_s=[]
    for i in range(len(T)):
        aux=[]
        for j in range(3):
            value=(norm.pdf((T[i]-mu_s[j])/sigma_s[j]))/((norm.sf((T[i]-mu_s[j])/sigma_s[j])))
            aux.append(value)
        w_s.append(aux)
    return w_s

#Esperanzas de Z y Z**2
def m1s(mu_s,sigma_s,w_s):
    m1=[]
    for i in range(len(T)):
        aux=[]
        for j in range(3):
            value=mu_s[j]+sigma_s[j]*w_s[i][j]
            aux.append(value)
        m1.append(aux)
    return m1

def m2s(mu_s,sigma_s,w_s):
    m2=[]
    for i in range(len(T)):
        aux=[]
        for j in range(3):
            value=mu_s[j]**2+sigma_s[j]**2+sigma_s[j]*(mu_s[j]+T[i])*w_s[i][j]
            aux.append(value)
        m2.append(aux)
    return m2
#Checar si el componente esta en la máscara
def ifMask(i,j):
    for k in masking[i]:
        if(j==k):
            return True
    return False
#Esperanza de la variable Bernoulli
def Gamma_s(w_s,sigma_s):
    gamma_s=[]
    for i in range(len(T)):
        aux=[]
        for j in range(1,4):
            if ifMask(i,j):
                denominador=0
                for l in masking[i]:
                    denominador=denominador+(w_s[i][l-1]/sigma_s[l-1])
                value=(w_s[i][j-1]/sigma_s[j-1])/denominador
            else:
                value=0
            aux.append(value)
        gamma_s.append(aux)
    return gamma_s


def Gamma_s1():
    gamma_s=[]
    for i in range(len(T)):
        aux=[]
        for j in range(1,4):
            if j==completCause[i]:
                value=1
            else:
                value=0
            aux.append(value)
        gamma_s.append(aux)
    return gamma_s

#--------------------------------------------------------------------------------------------------
#Aquí implementaremos el algoritmo EM para resolver el caso de 3 componentes

N=100 #Numero de iteraciones
def EM_AlgMask(N):
    #Inicializamos las variables
    mu_s=[random.choice(T),random.choice(T),random.choice(T)]
    std_T=st.stdev(T)
    sigma_s=[std_T,std_T,std_T]
    for k in range(N):
        w_s=W_s(mu_s,sigma_s)
        m1=m1s(mu_s,sigma_s,w_s)
        m2=m2s(mu_s,sigma_s,w_s)
        gamma_s=Gamma_s(w_s,sigma_s)
        w_s=list(zip(*w_s))
        m1=list(zip(*m1))
        m2=list(zip(*m2))
        gamma_s=list(zip(*gamma_s))
        for j in range(3):
            mu_s[j]=(1/len(T))*sum([gamma_s[j][i]*T[i]+(1-gamma_s[j][i])*m1[j][i] for i in range(len(T))])
            sigma_s[j]=mt.sqrt((1/len(T))*(sum([gamma_s[j][i]*(T[i]**2)+(1-gamma_s[j][i])*m2[j][i] for i in range(len(T))]))-mu_s[j]**2)

    estimador=hstack((mu_s,sigma_s))
    return estimador


def EM_AlgComp(N):
    #Inicializamos las variables
    mu_s=[random.choice(T),random.choice(T),random.choice(T)]
    std_T=st.stdev(T)
    sigma_s=[std_T,std_T,std_T]
    for k in range(N):
        w_s=W_s(mu_s,sigma_s)
        m1=m1s(mu_s,sigma_s,w_s)
        m2=m2s(mu_s,sigma_s,w_s)
        gamma_s=Gamma_s1()
        w_s=list(zip(*w_s))
        m1=list(zip(*m1))
        m2=list(zip(*m2))
        gamma_s=list(zip(*gamma_s))
        for j in range(3):
            mu_s[j]=(1/len(T))*sum([gamma_s[j][i]*T[i]+(1-gamma_s[j][i])*m1[j][i] for i in range(len(T))])
            sigma_s[j]=mt.sqrt((1/len(T))*sum([gamma_s[j][i]*(T[i]**2)+(1-gamma_s[j][i])*m2[j][i] for i in range(len(T))])-mu_s[j]**2)
    estimador=hstack((mu_s,sigma_s))
    return estimador
#-------------------------------------------------------------------------------------------------
#Aplicamos el algoritmo

theta1=EM_AlgMask(N)
theta2=EM_AlgComp(N)
print("Estimacion con los datos ocultos")
print(theta1)
print("Estimacion con los datos completos")
print(theta2)

#-----------------------------------------------------------------------------------------------------
# Validaremos el modelo con algunas graficas de CIF

#Funcion de riesgo (hazard)
def hz(t,j,theta):
    value=lognorm.pdf(t,s=theta[j+3],scale=np.exp(theta[j]))/lognorm.sf(t,s=theta[j+3],scale=np.exp(theta[j]))
    return value
#Supervivencia General (survivor)    
def sfGen(t,theta):
    value=1
    for j in range(3):
        value=value*lognorm.sf(t,s=theta[j+3],scale=np.exp(theta[j]))
    return value
#Densidad de la CIF (Cumulative Incidence Function)    
def cifpdf(t,j,theta):
    value=sfGen(t,theta)*hz(t,j,theta)
    return value
#CIF
def cif(x,j,theta):
    value=quad(cifpdf,0,x,args=(j,theta))
    return value


#Graficar
T0.sort()

aj1=AalenJohansenFitter()
aj2=AalenJohansenFitter()
aj3=AalenJohansenFitter()


#####################################################################################################
#Calculamos la CIF para cada estimacion que hicimos
Y1=[cif(t,0,theta1) for t in T0]  #Datos ocultos
Y2=[cif(t,0,theta2) for t in T0]  #Datos completos
Z1=[cif(t,1,theta1) for t in T0]
Z2=[cif(t,1,theta2) for t in T0]
W1=[cif(t,2,theta1) for t in T0]
W2=[cif(t,2,theta2) for t in T0]

plt.style.use('ggplot')

#Causa 1
aj1.fit(durations=df['Tiempos'].to_numpy(),event_observed=df['Causa'].to_numpy(),event_of_interest=1)
aj1.plot(label='Ajuste Aalen',color='blue')
l1=plt.plot(T0,Y1,'b-')
l2=plt.plot(T0,Y2,'b--')

#Diseño
plt.xlabel('Tiempos de fallo')
plt.ylabel('Probabilidad')
plt.title('Función de Incidencia Causa 1')

#Mostrar
plt.show()


#####################################################################################################

plt.style.use('ggplot')

#Causa 2
aj2=AalenJohansenFitter()
aj2.fit(durations=df['Tiempos'].to_numpy(),event_observed=df['Causa'].to_numpy(),event_of_interest=2)
aj2.plot(label='Ajuste Aalen',color='green')
plt.plot(T0,Z1,'g-',label='ocultos')
plt.plot(T0,Z2,'g--',label='completos')

#Diseño
plt.xlabel('Tiempos de fallo')
plt.ylabel('Probabilidad')
plt.title('Función de Incidencia Causa 2')

#Mostrar
plt.show()



#####################################################################################################
plt.style.use('ggplot')
#Causa 3
aj3=AalenJohansenFitter()
aj3.fit(durations=df['Tiempos'].to_numpy(),event_observed=df['Causa'].to_numpy(),event_of_interest=3)
aj3.plot(label='Ajuste Aalen',color='magenta')
plt.plot(T0,W1,'m-',label='ocultos')
plt.plot(T0,W2,'m--',label='completos')

#Diseño
plt.xlabel('Tiempos de fallo')
plt.ylabel('Probabilidad')
plt.title('Función de Incidencia Causa3')

#Mostrar
plt.show()


######################################################################################################
#Comparativo de las 3 causas
plt.style.use('ggplot')

aj1.plot(label='Causa 1',color='blue',ci_show=False)
plt.plot(T0,Y1,'b-')
plt.plot(T0,Y2,'b--')
aj2.plot(label='Causa 2',color='green',ci_show=False)
plt.plot(T0,Z1,'g-')
plt.plot(T0,Z2,'g--')
aj3.plot(label='Causa 3',color='magenta',ci_show=False)
plt.plot(T0,W1,'m-')
plt.plot(T0,W2,'m--')

#Diseño
plt.xlabel('Tiempos de fallo')
plt.ylabel('Probabilidad')
plt.title('CIF')

#Mostrar
plt.show()
