import streamlit as st
import pandas as pd
import joblib

# \--- Fungsi Konversi & Model ---
#@st.cache\_data
def convert_df_to_csv(df):
            """Mengonversi DataFrame ke format CSV (UTF-8) tanpa indeks."""
return df.to\_csv(index=False).encode('utf-8')

# Load model dari file tunggal
modul_dict = joblib.load('modul.pkl')

encoder = modul_dict['encoder']
scaler = modul_dict['scaler']
svd = modul_dict['svd']
dbscan = modul_dict['dbscan']
encoded_columns = modul_dict['encoded_columns']  # typo: tadi kamu tulis encoded_column (singular)
numerik_cols = joblib.load('num_cols.pkl')

# Kolom input
num_cols = ['TransactionAmount', 'CustomerAge', 'TransactionDuration', 'LoginAttempts',
            'AccountBalance','time_deff', 'Mean_Transaction', 'Mean_Account_Balance']

cat_cols = ['AccountID', 'TransactionType', 'Location', 'DeviceID', 'IP Address',
            'MerchantID', 'Channel', 'CustomerOccupation']

expected_columns = num_cols + cat_cols

st.title("🔍 Clustering Dataset (DBSCAN)")
st.write()
st.write("Pastikan Dataset yang diupload memuat kolom-kolom yang dibutuhkan")
st.write("Kolom Object : AccountID, TransactionType, Location, DeviceID, IP Address, MerchantID, Channel, CustomerOccupation")
st.write("Kolom Numerik : TransactionAmount, CustomerAge, TransactionDuration, LoginAttempts, AccountBalance, time_deff, Mean_Transaction, Mean_Account_Balance")

# \--- Bagian Unduh Template Kosong ---
st.subheader("1. Unduh Template")
st.write("Jika Anda belum memiliki dataset, unduh template di bawah ini dan isi sesuai format.")

# Membuat DataFrame kosong hanya dengan header
template\_df = pd.DataFrame(columns=expected\_columns)

# Menyiapkan template untuk diunduh
csv\_template = convert\_df\_to\_csv(template\_df)

st.download\_button(
label="📥 Unduh Template Kosong (CSV)",
data=csv\_template,
file\_name='template\_transaksi.csv',
mime='text/csv',
)
st.write("---")

uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Data yang diunggah:")
    st.dataframe(df.head(10))

    if not set(expected_columns).issubset(set(df.columns)):
        st.error("❌ Kolom dataset tidak lengkap atau tidak sesuai.")
        st.stop()

    #df = df[expected_columns]
    #st.dataframe(df.head())        

    try:
        # Step 1: Encoding
        encoded_array = encoder.transform(df[cat_cols])
        #st.write("📌 Bentuk hasil encoded_array:", encoded_array.shape)        
        temp_column = encoder.get_feature_names_out(cat_cols)
        #st.write("Tipe data encoded_array:", type(encoded_array))        
        #st.write("📌 Nama kolom hasil encoding saat ini:", temp_column)
        #st.write("📏 Jumlah kolom hasil encoding:", len(temp_column))        
        #st.write("📌 Nama kolom hasil encoding training:", encoded_columns.shape)        

        encoded_df = pd.DataFrame(encoded_array, columns=temp_column)
        encoded_df = encoded_df.reindex(columns=encoded_columns, fill_value=0)
        #st.dataframe(encoded_df.head(10))


        #missing = set(encoded_columns) - set(encoded_df.columns)
        #extra = set(encoded_df.columns) - set(encoded_columns)
        #st.write("❗ Kolom hilang dari data saat ini:", missing)
        #st.write("❗ Kolom tambahan yang tidak dikenali:", extra)   
        #st.dataframe(df[cat_cols].head())
        #st.dataframe(df[num_cols].head())
        #st.write("Tipe data numerik kolom:", type(numerik_cols))

        # Step 2: Scaling
        scaled_array = scaler.transform(df[num_cols])
        scaled_df = pd.DataFrame(scaled_array, columns=numerik_cols)
        #st.dataframe(scaled_df.head())

        # Step 3: Gabungkan
        combined = pd.concat([scaled_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        #st.dataframe(combined.head())

        # Step 4: SVD + DBSCAN
        reduced = svd.transform(combined)
        #st.write(reduced[0:4])        
        labels = dbscan.fit_predict(reduced)
        df['Cluster'] = labels

        st.success("✅ Klastering berhasil dilakukan!")
        st.dataframe(df['Cluster'].value_counts())
        st.write()
        st.write("Kluster 0 adalah Fraud")
        st.write("Kluster 1 dan -1 adalah Non-Fraud")
        st.write()
        df['Cluster'] = df['Cluster'].replace({0 : 'Fraud', 1 : 'Non-Fraud', -1 : 'Non-Fraud'})        
        st.dataframe(df)

        if 'button_fraud' not in st.session_state:
            st.session_state.button_fraud = False
            st.session_state.button_fraud = False

        if 'button_non_fraud' not in st.session_state:
            st.session_state.button_non_fraud = False

        # Fungsi tombol fraud
        def click_button_fraud():
            st.session_state.button_fraud = not st.session_state.button_fraud
            st.session_state.button_non_fraud = False  # Optional: supaya hanya satu aktif

# Fungsi tombol non-fraud
        def click_button_non_fraud():
            st.session_state.button_non_fraud = not st.session_state.button_non_fraud
            st.session_state.button_fraud = False  # Optional

# Tombol-tombol
        st.button('Transaksi Fraud', on_click=click_button_fraud)
        st.button('Transaksi Non-Fraud', on_click=click_button_non_fraud)

# Tampilkan dataframe sesuai tombol
        if st.session_state.button_fraud:
            df_fraud = df[df['Cluster'] == 'Fraud']
            st.dataframe(df_fraud)

        if st.session_state.button_non_fraud:
            df_non_fraud = df[df['Cluster'] == 'Non-Fraud']
            st.dataframe(df_non_fraud)
    except Exception as e:
        st.error(f"❌ Terjadi error saat menjalankan pipeline: {e}")
