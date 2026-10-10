
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Configuración principal de la página
st.set_page_config(page_title="Predicción Suscripción Bancaria", page_icon="🏦", layout="wide")

# Cargar el modelo en caché para que no se recargue cada vez que el usuario toca un botón
@st.cache_resource
def load_model():
    with open('modelo_final.pkl', 'rb') as file:
        return pickle.load(file)

modelo = load_model()

# Encabezados
st.title('🏦 Herramienta de Predicción - Marketing Bancario')
st.markdown('Introduzca los datos del cliente para evaluar la probabilidad de suscripción a un depósito a plazo.')
st.markdown("---")

st.header('Datos del Cliente')

# Utilizamos tres columnas para organizar mejor el formulario
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Datos Demográficos")
    age = st.number_input('Edad', min_value=18, max_value=120, value=35)
    job = st.selectbox('Tipo de trabajo (job)', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
    marital = st.selectbox('Estado civil (marital)', ['divorced', 'married', 'single', 'unknown'])
    education = st.selectbox('Educación (education)', ['unknown', 'primary', 'secondary', 'tertiary'])

with col2:
    st.subheader("Datos Financieros")
    balance = st.number_input('Saldo medio anual (balance)', value=0)
    default = st.selectbox('¿Tiene algún crédito en mora? (default)', ['no', 'yes', 'unknown'])
    housing = st.selectbox('¿Tiene hipoteca? (housing)', ['no', 'yes', 'unknown'])
    loan = st.selectbox('¿Tiene un préstamo personal? (loan)', ['no', 'yes', 'unknown'])

with col3:
    st.subheader("Datos de Campaña")
    contact = st.selectbox('Tipo de contacto (contact)', ['cellular', 'telephone', 'unknown'])
    month = st.selectbox('Mes del último contacto (month)', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day = st.number_input('Día del mes (day)', min_value=1, max_value=31, value=15)
    duration = st.number_input('Duración de la llamada en seg. (duration)', min_value=0, value=150)
    campaign = st.number_input('Contactos en esta campaña (campaign)', min_value=1, value=1)
    pdays = st.number_input('Días desde último contacto ant. (pdays)', value=-1, help="-1 significa que no fue contactado antes")
    previous = st.number_input('Contactos antes de esta campaña (previous)', min_value=0, value=0)
    poutcome = st.selectbox('Resultado campaña anterior (poutcome)', ['failure', 'nonexistent', 'success', 'unknown'])

st.markdown("---")

# Botón para ejecutar la predicción
if st.button('Generar Predicción', use_container_width=True):
    
    # Recolectamos todas las entradas en un DataFrame con las columnas exactas del dataset original
    input_data = pd.DataFrame([[
        age, balance, day, duration, campaign, pdays, previous, 
        education, job, marital, default, housing, loan, contact, month, poutcome
    ]], columns=[
        'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous',
        'education', 'job', 'marital', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'
    ])

    input_data['pdays'] = pd.cut(input_data['pdays'],
                                 bins=[-2, -1, 30, 105, 180, 210, np.inf],
                                 labels=[0, 5, 4, 3, 2, 1]).astype(int)
    
    # Realizamos la predicción usando el pipeline completo
    prediccion = modelo.predict(input_data)[0]
    
    # Mostramos el resultado de forma visual
    st.header('Resultado:')
    
    # Mostramos el resultado
    if prediccion == 'yes' or prediccion == 1:
        st.success('El modelo predice que este cliente **SÍ** suscribirá el producto bancario.')
    else:
        st.error('El modelo predice que este cliente **NO** suscribirá el producto bancario.')