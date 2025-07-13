import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kanker Paru-paru",
    page_icon="🫁",
    layout="wide"
)

# CSS untuk kompatibilitas dark mode
st.markdown("""
<style>
.header-container {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    border: 2px solid #667eea;
}
.header-title {
    color: #ffffff !important;
    text-align: center;
    margin: 0;
    font-size: 2.5rem;
    font-weight: bold;
}
.header-subtitle {
    color: #ffffff !important;
    text-align: center;
    margin: 5px 0 0 0;
    font-size: 1.2rem;
}
.researcher-container {
    background-color: rgba(70, 130, 180, 0.1);
    border: 2px solid #4682b4;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.researcher-title {
    color: #4682b4 !important;
    margin-top: 0;
    font-size: 1.5rem;
}
.researcher-info {
    font-size: 16px;
    margin: 5px 0;
    color: inherit;
}
</style>
""", unsafe_allow_html=True)

# Header dengan identitas peneliti
st.markdown("""
<div class='header-container'>
    <h1 class='header-title'>🫁 Sistem Prediksi Kanker Paru-paru</h1>
    <p class='header-subtitle'>Menggunakan Algoritma Random Forest</p>
</div>
""", unsafe_allow_html=True)

# Identitas peneliti
st.markdown("""
<div class='researcher-container'>
    <h3 class='researcher-title'>👩‍🎓 Peneliti</h3>
    <p class='researcher-info'><strong>Nama:</strong> Olivia Anjelika Sitepu</p>
    <p class='researcher-info'><strong>Status:</strong> Penelitian Ilmiah</p>
</div>
""", unsafe_allow_html=True)

# Latar belakang penelitian
with st.expander("📖 Latar Belakang Penelitian", expanded=False):
    st.markdown("""
    **Kanker paru-paru** merupakan salah satu penyakit kanker yang paling umum dan mematikan di seluruh dunia. 
    The American Cancer Society memperkirakan akan ada **234.580 kasus baru** kanker paru-paru pada tahun 2024, 
    dengan tingkat mortalitas yang tinggi.
    
    **Tantangan Utama:**
    - Keterlambatan dalam diagnosis dapat mengakibatkan kemajuan penyakit yang tidak terkendali
    - Mengurangi peluang kesembuhan dan memperburuk prognosis pasien
    - Masalah utama pada identifikasi awal yang tepat dan manajemen risiko yang efektif
    
    **Solusi dengan Machine Learning:**
    Perkembangan teknologi kecerdasan buatan telah membuka peluang baru dalam meningkatkan deteksi dini 
    dan manajemen risiko kanker paru-paru. **Algoritma Random Forest** terbukti mampu menganalisis berbagai 
    faktor risiko dan gejala klinis secara mendalam, memungkinkan pembuatan prediksi yang lebih tepat dan 
    personalisasi tingkat risiko berdasarkan karakteristik individu.
    
    Pengembangan sistem prediksi yang akurat dan mudah diakses diharapkan dapat memberikan kontribusi nyata 
    dalam upaya pencegahan dan penanganan kanker paru-paru di Indonesia dan secara global.
    """)

# Sidebar untuk informasi
st.sidebar.header("ℹ️ Informasi Aplikasi")
st.sidebar.info(
    "Aplikasi ini menggunakan model Random Forest untuk memprediksi "
    "kemungkinan kanker paru-paru berdasarkan gejala dan faktor risiko."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**👩‍🎓 Peneliti:**")
st.sidebar.markdown("Olivia Anjelika Sitepu")
st.sidebar.markdown("Penelitian Ilmiah")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/random_forest_lung_cancer_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model tidak ditemukan! Pastikan file 'random_forest_lung_cancer_model.pkl' ada di direktori yang sama.")
        return None

# Fungsi prediksi
def predict_lung_cancer(model, input_data):
    prediction = model.predict([input_data])
    probability = model.predict_proba([input_data])
    return prediction[0], probability[0]

# Load model
model = load_model()

if model is not None:
    # Membuat form input
    st.header("📝 Masukkan Data Pasien")
    
    # Membagi kolom untuk input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Kebiasaan & Gaya Hidup")
        smoking = st.selectbox("Merokok", ["Tidak", "Ya"])
        alcohol = st.selectbox("Konsumsi Alkohol", ["Tidak", "Ya"])
        peer_pressure = st.selectbox("Tekanan Teman Sebaya", ["Tidak", "Ya"])
        
        st.subheader("Gejala Fisik")
        yellow_fingers = st.selectbox("Jari Kuning", ["Tidak", "Ya"])
        anxiety = st.selectbox("Kecemasan", ["Tidak", "Ya"])
        chronic_disease = st.selectbox("Penyakit Kronis", ["Tidak", "Ya"])
        fatigue = st.selectbox("Kelelahan", ["Tidak", "Ya"])
    
    with col2:
        st.subheader("Gejala Pernapasan")
        allergy = st.selectbox("Alergi", ["Tidak", "Ya"])
        wheezing = st.selectbox("Mengi", ["Tidak", "Ya"])
        coughing = st.selectbox("Batuk", ["Tidak", "Ya"])
        swallowing_difficulty = st.selectbox("Kesulitan Menelan", ["Tidak", "Ya"])
        chest_pain = st.selectbox("Nyeri Dada", ["Tidak", "Ya"])
        
        st.subheader("Faktor Tambahan")
        anxyelfin = st.selectbox("ANXYELFIN", ["Tidak", "Ya"])
    
    # Konversi input ke format numerik
    def convert_to_numeric(value):
        return 1 if value == "Ya" else 0
    
    # Membuat array input untuk prediksi
    input_data = [
        convert_to_numeric(smoking),
        convert_to_numeric(yellow_fingers),
        convert_to_numeric(anxiety),
        convert_to_numeric(peer_pressure),
        convert_to_numeric(chronic_disease),
        convert_to_numeric(fatigue),
        convert_to_numeric(allergy),
        convert_to_numeric(wheezing),
        convert_to_numeric(alcohol),
        convert_to_numeric(coughing),
        convert_to_numeric(swallowing_difficulty),
        convert_to_numeric(chest_pain),
        convert_to_numeric(anxyelfin)
    ]
    
    # Tombol prediksi
    if st.button("🔍 Prediksi Kanker Paru-paru", type="primary"):
        if model is not None:
            try:
                # Melakukan prediksi
                prediction, probability = predict_lung_cancer(model, input_data)
                
                # Menampilkan hasil
                st.markdown("---")
                st.header("📊 Hasil Prediksi")
                
                # Membuat kolom untuk hasil
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.error("⚠️ RISIKO TINGGI")
                        st.markdown("**Prediksi: Berisiko Kanker Paru-paru**")
                    else:
                        st.success("✅ RISIKO RENDAH")
                        st.markdown("**Prediksi: Tidak Berisiko Kanker Paru-paru**")
                
                with col2:
                    st.info("📈 Probabilitas")
                    st.markdown(f"**Tidak Berisiko:** {probability[0]:.2%}")
                    st.markdown(f"**Berisiko:** {probability[1]:.2%}")
                
                with col3:
                    st.warning("⚠️ Disclaimer")
                    st.markdown(
                        "Hasil ini hanya untuk referensi. "
                        "Konsultasikan dengan dokter untuk diagnosis yang akurat."
                    )
                
                # Progress bar untuk probabilitas
                st.subheader("Visualisasi Probabilitas")
                prob_risk = probability[1]
                st.progress(prob_risk)
                st.caption(f"Probabilitas Risiko: {prob_risk:.2%}")
                
                # Menampilkan data input
                st.subheader("📋 Ringkasan Input Data")
                input_df = pd.DataFrame({
                    'Faktor': [
                        'Merokok', 'Jari Kuning', 'Kecemasan', 'Tekanan Teman Sebaya',
                        'Penyakit Kronis', 'Kelelahan', 'Alergi', 'Mengi',
                        'Konsumsi Alkohol', 'Batuk', 'Kesulitan Menelan', 'Nyeri Dada',
                        'ANXYELFIN'
                    ],
                    'Status': [
                        'Ya' if x == 1 else 'Tidak' for x in input_data
                    ]
                })
                st.dataframe(input_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
        else:
            st.error("Model tidak dapat dimuat!")
    
    # Informasi tambahan
    st.markdown("---")
    st.subheader("📚 Informasi Tambahan")
    
    with st.expander("Tentang Model Random Forest"):
        st.write("""
        Random Forest adalah algoritma machine learning yang menggunakan ensemble dari banyak decision trees 
        untuk membuat prediksi. Model ini cocok untuk klasifikasi dan regresi, dan umumnya memberikan 
        akurasi yang baik dengan overfitting yang minimal.
        """)
    
    with st.expander("Faktor Risiko Kanker Paru-paru"):
        st.write("""
        - **Merokok**: Faktor risiko utama kanker paru-paru
        - **Paparan Zat Kimia**: Asbes, radon, dan polusi udara
        - **Riwayat Keluarga**: Genetik dapat berperan
        - **Usia**: Risiko meningkat seiring bertambahnya usia
        - **Penyakit Paru Kronis**: COPD dan fibrosis paru
        """)
    
    with st.expander("Gejala yang Perlu Diperhatikan"):
        st.write("""
        - Batuk yang tidak kunjung sembuh
        - Sesak napas atau mengi
        - Nyeri dada yang persisten
        - Batuk berdarah
        - Kelelahan yang tidak biasa
        - Penurunan berat badan tanpa sebab yang jelas
        """)

else:
    st.error("Aplikasi tidak dapat berjalan karena model tidak ditemukan!")
    st.info("Pastikan file 'random_forest_lung_cancer_model.pkl' ada di direktori yang sama dengan aplikasi ini.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; border: 1px solid #4682b4; border-radius: 10px; background-color: rgba(70, 130, 180, 0.05);'>
    <p style='margin: 5px 0; font-weight: bold; color: #4682b4;'>Aplikasi Prediksi Kanker Paru-paru</p>
    <p style='margin: 5px 0;'>Peneliti: Olivia Anjelika Sitepu | Penelitian Ilmiah</p>
    <p style='margin: 5px 0;'>Menggunakan Algoritma Random Forest untuk Deteksi Dini Kanker Paru-paru</p>
    <p style='margin: 5px 0; font-style: italic;'>Untuk Tujuan Edukasi dan Penelitian</p>
</div>
""", unsafe_allow_html=True)
