#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#######################
# Page configuration
st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>
[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}
[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}
[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>
""", unsafe_allow_html=True)

#######################
# Load data
df = pd.read_csv('Housing_Data.csv')

#######################
# Sidebar
with st.sidebar:
    st.title('🏠 Prediksi Harga Rumah Berdasarkan Luas')
    
    # Input untuk luas
    area_input = st.number_input("Masukkan luas rumah (m²)", min_value=0.0)

#######################
# Memisahkan fitur dan target
X = df[['area']]  # Menggunakan kolom 'area'
y = df['price']  # Menggunakan kolom 'price'

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan fit model regresi linier
model = LinearRegression()
model.fit(X_train, y_train)

# Melakukan prediksi
y_pred = model.predict(X_test)

# Menghitung metrik evaluasi
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#######################
# Dashboard Main Panel
col1, col2 = st.columns(2)

with col1:
    st.markdown('### Hasil Evaluasi Model')
    st.write(f"Mean Squared Error: {mse:.3f}")
    st.write(f"R² Score: {r2:.3f}")

with col2:
    # Visualisasi prediksi vs nilai aktual
    st.markdown('### Prediksi vs Nilai Aktual')
    lr_diff = pd.DataFrame({'Actual Value': y_test, 'Predicted Value': y_pred})
    st.line_chart(lr_diff)

# Tombol prediksi
if st.button("Prediksi Harga"):
    if area_input > 0:  # Pastikan input area valid
        # Melakukan prediksi harga berdasarkan input luas
        input_data = pd.DataFrame([[area_input]], columns=['area'])
        predicted_price = model.predict(input_data)[0]
        st.markdown(f"### Harga yang diprediksi untuk luas {area_input} m²: **${predicted_price:.2f}**")
    else:
        st.warning("Silakan masukkan luas yang valid.")
