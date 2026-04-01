import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# =====================
# تحميل قاعدة البيانات
# =====================
df = pd.read_excel("D:\A Doc\Mini Projects\soil_dataset.xlsx")

# تحويل نوع التربة إلى أرقام
le = LabelEncoder()
df["Soil_Type"] = le.fit_transform(df["Soil_Type"])

# تدريب النموذج
X = df.drop("Crop", axis=1)
y = df["Crop"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# =====================
# واجهة التطبيق
# =====================

st.title("🌱 Soil Crop Predictor")
st.write("أدخل نتائج تحليل التربة لمعرفة المحصول المناسب")

# إدخال البيانات
ph = st.number_input("pH", 0.0, 14.0)
ec = st.number_input("EC")
n = st.number_input("Nitrogen (N)")
p = st.number_input("Phosphorus (P)")
k = st.number_input("Potassium (K)")

soil = st.selectbox(
    "نوع التربة",
    ["clay", "sandy", "loam"]
)

moisture = st.number_input("Moisture (%)")
temp = st.number_input("Temperature (°C)")

# زر التحليل
if st.button("Predict Crop"):

    soil_encoded = le.transform([soil])[0]

    sample = [[ph, ec, n, p, k, soil_encoded, moisture, temp]]

    prediction = model.predict(sample)
    probabilities = model.predict_proba(sample)

    st.success(f"المحصول الأنسب: {prediction[0]}")

    st.write("### الاحتمالات")

    for crop, prob in zip(model.classes_, probabilities[0]):
        st.write(f"{crop} : {round(prob*100,2)} %")
    
    # Run from Anaconda Prompt: streamlit run app.py
