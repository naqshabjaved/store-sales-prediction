import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

try:
    model = joblib.load('model.pkl')
    le_fat = joblib.load('le_fat.pkl')
    le_size = joblib.load('le_size.pkl')
    le_loc = joblib.load('le_loc.pkl')
    le_type = joblib.load('le_type.pkl')
    ohe = joblib.load('ohe.pkl')
    imputation_values = joblib.load('imputation_values.pkl')
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please run the notebook to generate all .pkl files.")
    st.stop()

def preprocess_input(data):
    
    df = pd.DataFrame([data])
    
    df['Item_Weight'] = df['Item_Weight'].fillna(imputation_values['Item_Weight'])
    df['Outlet_Size'] = df['Outlet_Size'].fillna(imputation_values['Outlet_Size'])
    
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat', 'low fat':'Low Fat', 'reg':'Regular'})
    
    df['Item_Visibility'] = df['Item_Visibility'].replace(0, imputation_values['Item_Visibility'])
    
    def get_item_category(item_type):
        if item_type in ['Dairy', 'Meat', 'Breads', 'Baking Goods', 'Breakfast', 'Fruits and Vegetables', 'Seafood', 'Starchy Foods', 'Soft Drinks']:
            return 'Food'
        elif item_type == 'Health and Hygiene' or item_type == 'Household' or item_type == 'Others':
            return 'Non-Consumable'
        else:
            return 'Drink'
    df['Item_Category'] = df['Item_Type'].apply(get_item_category)

    df['Outlet_Years'] = datetime.now().year - df['Outlet_Establishment_Year']
    
    df['Item_Fat_Content'] = le_fat.transform(df['Item_Fat_Content'])
    df['Outlet_Size'] = le_size.transform(df['Outlet_Size'])
    df['Outlet_Location_Type'] = le_loc.transform(df['Outlet_Location_Type'])
    df['Outlet_Type'] = le_type.transform(df['Outlet_Type'])
    
    ohe_cols = ohe.get_feature_names_out(['Item_Category'])
    ohe_data = ohe.transform(df[['Item_Category']]).toarray()
    ohe_df = pd.DataFrame(ohe_data, columns=ohe_cols)
    
    df = pd.concat([df, ohe_df], axis=1)
    df = df.drop(columns=['Item_Type', 'Outlet_Establishment_Year', 'Item_Category'], axis=1)
    
    df_aligned = df.reindex(columns=model_columns, fill_value=0)
    
    return df_aligned

st.set_page_config(page_title="Store Sales Prediction", layout="wide")
st.title('Store Sales Prediction')
st.write("Predict the sales for an item in a store based on its properties.")

col1, col2 = st.columns(2)

with col1:
    st.header("Item Details")
    Item_Weight = st.number_input('Item Weight', min_value=1.0, max_value=25.0, value=12.8, help="Leave as 12.8 to use the average weight")
    Item_Fat_Content = st.selectbox('Item Fat Content', ('Low Fat', 'Regular'))
    Item_Visibility = st.number_input('Item Visibility (0 to 1)', min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    Item_Type = st.selectbox('Item Type', 
                             ('Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 
                              'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast', 
                              'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads', 
                              'Starchy Foods', 'Others', 'Seafood'))
    Item_MRP = st.number_input('Item MRP (Price)', min_value=1.0, max_value=300.0, value=150.0)

with col2:
    st.header("Outlet Details")
    Outlet_Establishment_Year = st.slider('Outlet Establishment Year', 1980, 2010, 2000)
    Outlet_Size = st.selectbox('Outlet Size', ('Small', 'Medium', 'High'), index=1)
    Outlet_Location_Type = st.selectbox('Outlet Location Type', ('Tier 1', 'Tier 2', 'Tier 3'))
    Outlet_Type = st.selectbox('Outlet Type', 
                               ('Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'))

if st.button('Predict Sales', type="primary", use_container_width=True):
    input_data = {
        'Item_Weight': Item_Weight,
        'Item_Fat_Content': Item_Fat_Content,
        'Item_Visibility': Item_Visibility,
        'Item_Type': Item_Type,
        'Item_MRP': Item_MRP,
        'Outlet_Establishment_Year': Outlet_Establishment_Year,
        'Outlet_Size': Outlet_Size,
        'Outlet_Location_Type': Outlet_Location_Type,
        'Outlet_Type': Outlet_Type
    }
    
    processed_data = preprocess_input(input_data)
    
    prediction = model.predict(processed_data)
    predicted_sales = prediction[0]
    
    st.success(f"**Predicted Item Outlet Sales: `${max(0, predicted_sales):.2f}`**")