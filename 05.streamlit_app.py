import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Store Sales Forecaster",
    page_icon="🛒",
    layout="wide"
)

# --- Helper Functions ---
def resolve_model_path(filename):
    possible_paths = [
        os.path.join("models", filename),
        os.path.join("src", filename),
        filename,
        os.path.join("..", "models", filename)
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def preprocess_input(df):
    """
    Applies Feature Engineering to match training data.
    """
    df = df.copy()
    
    # 1. Date Features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
    
    # 2. Family Encoding
    families = [
        'AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 'BREAD/BAKERY', 
        'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS', 
        'GROCERY I', 'GROCERY II', 'HARDWARE', 'HOME AND KITCHEN I', 
        'HOME AND KITCHEN II', 'HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR', 
        'LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 
        'PERSONAL CARE', 'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY', 
        'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD'
    ]
    fam_map = {f: i for i, f in enumerate(sorted(families))}
    
    if 'family' in df.columns:
        df['family'] = df['family'].map(fam_map).fillna(0)
        
    return df

@st.cache_resource
def load_model():
    model_path = resolve_model_path("best_model.pkl")
    if model_path:
        # CRITICAL: Using mmap_mode='r' to prevent MemoryError
        return joblib.load(model_path, mmap_mode='r')
    return None

# --- Main App ---
def main():
    st.title("🛒 Store Sales Forecasting")
    
    model = load_model()
    if model is None:
        st.error("🚨 Model not found.")
        return

    tab1, tab2 = st.tabs(["📝 Single Prediction (Manual)", "📁 Batch Prediction (CSV)"])

    with tab1:
        st.subheader("Forecast for a Specific Item")
        col1, col2 = st.columns(2)
        with col1:
            date_input = st.date_input("Date", value=pd.to_datetime("2017-08-16"))
            store_nbr = st.number_input("Store Number", min_value=1, max_value=54, value=1)
        with col2:
            families_list = sorted([
                'AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 'BREAD/BAKERY', 
                'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS', 
                'GROCERY I', 'GROCERY II', 'HARDWARE', 'HOME AND KITCHEN I', 
                'HOME AND KITCHEN II', 'HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR', 
                'LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 
                'PERSONAL CARE', 'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY', 
                'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD'
            ])
            family_input = st.selectbox("Product Family", families_list)
            onpromotion = st.selectbox("On Promotion?", [0, 1], index=0)

        if st.button("Predict Sales"):
            input_data = pd.DataFrame({
                'date': [pd.to_datetime(date_input)],
                'store_nbr': [store_nbr],
                'family': [family_input],
                'onpromotion': [onpromotion]
            })
            processed_data = preprocess_input(input_data)
            features = ['store_nbr', 'family', 'onpromotion', 'year', 'month', 'day', 'dayofweek']
            X = processed_data[features]
            
            try:
                pred_log = model.predict(X)
                pred_real = np.expm1(pred_log)[0]
                st.divider()
                st.metric(label="Predicted Sales (Units)", value=f"{pred_real:.2f}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")

    with tab2:
        st.subheader("Upload Test Data (CSV)")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if st.button("Forecast Batch"):
                    processed_df = preprocess_input(df)
                    features = ['store_nbr', 'family', 'onpromotion', 'year', 'month', 'day', 'dayofweek']
                    if all(c in processed_df.columns for c in features):
                        X_batch = processed_df[features]
                        preds_log = model.predict(X_batch)
                        preds_real = np.expm1(preds_log)
                        df['sales'] = np.maximum(preds_real, 0)
                        st.success("Done!")
                        st.write(df[['date', 'store_nbr', 'family', 'sales']].head())
                        csv = df[['id', 'sales']].to_csv(index=False).encode('utf-8') if 'id' in df.columns else df[['sales']].to_csv(index=False).encode('utf-8')
                        #st.download_button("Download Submission CSV", csv, "submission.csv", "text/csv")
                    else:
                        st.error("Missing required columns after preprocessing.")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
