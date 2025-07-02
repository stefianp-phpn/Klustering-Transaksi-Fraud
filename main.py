import streamlit as st
import pandas as pd
import joblib

# Load model dari file tunggal
modul_dict = joblib.load('modul.pkl')

encoder = modul_dict['encoder']
scaler = modul_dict['scaler']
svd = modul_dict['svd']
dbscan = modul_dict['dbscan']
encoded_columns = modul_dict['encoded_columns']  # typo: tadi kamu tulis encoded_column (singular)

# Kolom input
num_cols = ['AccountBalance', 'LoginAttempts', 'TransactionDuration', 'CustomerAge',
            'TransactionAmount','time_deff', 'Mean_Transaction', 'Mean_Account_Balance']

cat_cols = ['AccountID', 'TransactionType', 'Location', 'DeviceID', 'IP Address',
            'MerchantID', 'Channel', 'CustomerOccupation']

expected_columns = num_cols + cat_cols

st.title("🔍 Clustering Dataset (DBSCAN)")

uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Data yang diunggah:")
    st.dataframe(df.head(10))

    if not set(expected_columns).issubset(set(df.columns)):
        st.error("❌ Kolom dataset tidak lengkap atau tidak sesuai.")
        st.stop()

    df = df[expected_columns]

    try:
        # Step 1: Encoding
        encoded_array = encoder.transform(df[cat_cols])
        st.write("📌 Bentuk hasil encoded_array:", encoded_array.shape)        
        temp_column = encoder.get_feature_names_out(cat_cols)
        #st.write("Tipe data encoded_array:", type(encoded_array))        
        st.write("📌 Nama kolom hasil encoding saat ini:", temp_column)
        st.write("📏 Jumlah kolom hasil encoding:", len(temp_column))        
        st.write("📌 Nama kolom hasil encoding training:", encoded_columns.shape)        

        encoded_df = pd.DataFrame(encoded_array, columns=temp_column)
        #encoded_df = encoded_df.reindex(columns=encoded_columns, fill_value=0)
                
        #missing = set(encoded_columns) - set(encoded_df.columns)
        #extra = set(encoded_df.columns) - set(encoded_columns)
        #st.write("❗ Kolom hilang dari data saat ini:", missing)
        #st.write("❗ Kolom tambahan yang tidak dikenali:", extra)   
                
        # Step 2: Scaling
        scaled_array = scaler.transform(df[num_cols])
        scaled_df = pd.DataFrame(scaled_array, columns=num_cols)

        # Step 3: Gabungkan
        combined = pd.concat([scaled_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # Step 4: SVD + DBSCAN
        reduced = svd.transform(combined.values)
        labels = dbscan.fit_predict(reduced)
        df['Cluster'] = labels

        st.success("✅ Klastering berhasil dilakukan!")
        st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Terjadi error saat menjalankan pipeline: {e}")
