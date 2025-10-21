import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import duckdb 
import re
import os
import zipfile # <-- NEW: Import for handling ZIP files

# --- 1. CONFIGURATION & MAPPING ---
st.set_page_config(
    page_title="Oman Aluminium & Materials Market Overview",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Map between the internal DataFrame variable name and a recognizable substring
# of the filename expected inside the ZIP archive.
REQUIRED_FILES_MAP = {
    'df_company_hs': 'Company_Data_with_HS_Codes',
    'df_alum_co': 'ALUMINIUM COMPANIES',
    'df_imports': 'Materials Imports',
    'df_exports': 'Materials Export',
    'df_reexports': 'Materials Re Exports',
    'df_prod_1980': 'Production Shift 1980',
    'df_prod_2016': 'Production Shift 2016',
    'df_prod_2022': 'Production Shift 2022',
    'df_alum_ie': 'Aluminium Imports Exports',
    'df_partner_exp_2023': 'Trading Partners Export 2023',
    'df_partner_imp_2023': 'Trading Partners Import 2023',
    'df_hs_exp_7601': 'Export and Re-export 7601',
    'df_hs_exp_7604': 'Export and Re-export 7604',
    'df_hs_exp_7605': 'Export and Re-export 7605',
    'df_hs_imp_7601': 'Imports 7601',
    'df_hs_imp_7604': 'Imports 7604',
    'df_hs_imp_7605': 'Imports 7605',
}

# --- 2. DATA LOADING & CORE PROCESSING UTILITIES (UPDATED) ---

def load_file_from_buffer(file_buffer, file_name):
    """
    Loads data from an in-memory buffer (BytesIO) based on its file name/type.
    This replaces load_uploaded_file and is used for files extracted from the ZIP.
    """
    if file_buffer is None:
        return None
    
    try:
        file_name = file_name.lower()
        
        if file_name.endswith(('.parquet', '.feather')):
            # Parquet/Feather: Read directly from buffer
            df = pd.read_parquet(file_buffer)
        elif file_name.endswith(('.xlsx', '.xls', '.csv')):
            # CSV/Excel: Use file type specific reader
            if file_name.endswith(('.xlsx', '.xls')):
                # For Excel, use BytesIO to simulate file path
                df = pd.read_excel(file_buffer, sheet_name=0) 
            else:
                # For CSV, decode content before reading
                file_buffer.seek(0)
                df = pd.read_csv(file_buffer, engine='c') 
        else:
            st.warning(f"Unsupported file type inside ZIP: {file_name}. Skipping.")
            return None
        
        return df
    except Exception as e:
        # st.error(f"Error loading {file_name}: {e}")
        return None

def unzip_and_load_data(zip_file_uploader):
    """
    Handles the ZIP file upload, extracts all files in memory,
    and loads the 17 required files into DataFrames based on REQUIRED_FILES_MAP.
    """
    if zip_file_uploader is None:
        return [None] * 17

    if not zip_file_uploader.name.lower().endswith('.zip'):
        st.error("Please upload a single ZIP file containing all data files.")
        return [None] * 17
    
    data_frames = {}
    missing_files = []
    
    try:
        # Read ZIP file content into an in-memory BytesIO buffer
        with zipfile.ZipFile(BytesIO(zip_file_uploader.read())) as z:
            zip_file_list = z.namelist()
            
            # Create a dictionary to hold the loaded DataFrames
            df_results = {var_name: None for var_name in REQUIRED_FILES_MAP.keys()}
            
            # Iterate over the required files
            for var_name, keyword in REQUIRED_FILES_MAP.items():
                
                # Find the matching file in the zip file list
                found_file = next(
                    (f for f in zip_file_list if keyword in f and not f.endswith('/')), 
                    None
                )
                
                if found_file:
                    with z.open(found_file) as f:
                        # Read file content into a buffer and load
                        file_buffer = BytesIO(f.read())
                        df = load_file_from_buffer(file_buffer, found_file)
                        df_results[var_name] = df
                else:
                    missing_files.append(keyword)

        if missing_files:
            st.warning(f"Could not find {len(missing_files)} file(s) in the ZIP matching keywords: {', '.join(missing_files)}.")
            st.info("Ensure your files inside the ZIP contain these keywords in their name.")
            
        # Return the list of DataFrames in the correct order for process_data
        return [df_results[var] for var in REQUIRED_FILES_MAP.keys()]

    except zipfile.BadZipFile:
        st.error("The uploaded file is not a valid ZIP file.")
        return [None] * 17
    except Exception as e:
        st.error(f"An unexpected error occurred during ZIP extraction: {e}")
        return [None] * 17

# Caching the main data processing function for optimal performance.
@st.cache_data
def process_data(df_company_hs, df_alum_co, df_imports, df_exports, df_reexports,
                 df_prod_1980, df_prod_2016, df_prod_2022,
                 df_alum_ie, df_partner_exp_2023, df_partner_imp_2023,
                 df_hs_exp_7601, df_hs_exp_7604, df_hs_exp_7605,
                 df_hs_imp_7601, df_hs_imp_7604, df_hs_imp_7605):
    """Performs all necessary data cleaning and transformation with type optimization on 17 files."""
    
    # --- Data Processing Logic (UNCHANGED from previous working version) ---
    
    # 1. Company Data (df_company_hs)
    if df_company_hs is not None:
        df_company_hs = df_company_hs.dropna(subset=['HS Code']).copy()
        df_company_hs['HS Code'] = df_company_hs['HS Code'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        df_company_hs['HS_4DG'] = df_company_hs['HS Code'].str[:4].astype('category')
        df_company_hs['Governorate'] = df_company_hs['Governorate'].astype('category')
        df_company_hs['Village'] = df_company_hs['Village'].astype('category') 
    
    # 2. Aluminium Companies Data (df_alum_co)
    if df_alum_co is not None:
        df_alum_co['lat'] = pd.to_numeric(df_alum_co['lat'], errors='coerce').astype(np.float32)
        df_alum_co['lon'] = pd.to_numeric(df_alum_co['lon'], errors='coerce').astype(np.float32)
        df_alum_co['capacity_tpy'] = pd.to_numeric(df_alum_co['capacity_tpy'], errors='coerce').fillna(0).astype(np.float32)
    
    # 3. Trade Data (Using DuckDB for High-Speed SQL Unpivot/Union)
    trade_data_frames = {"Import": df_imports, "Export": df_exports, "Re-export": df_reexports}
    df_trade = pd.DataFrame() # Initialize empty trade dataframe
    
    # Check if any trade data is available
    if any(df is not None and not df.empty for df in trade_data_frames.values()):
        con = duckdb.connect(':memory:')
        sql_union_parts = []
        
        # Get column names from the first non-empty dataframe
        first_df = next((df for df in trade_data_frames.values() if df is not None and not df.empty), None)
        
        if first_df is not None:
            # Dynamically determine the year/value/weight columns
            year_value_cols = [col for col in first_df.columns if '(Value in OMR)' in col]
            year_weight_cols = [col for col in first_df.columns if '(Weight in KG)' in col]
                
            for trade_type, df in trade_data_frames.items():
                if df is None or df.empty: continue
                    
                safe_trade_type = trade_type.lower().replace('-', '_')
                con.register(f'df_{safe_trade_type}', df)
                
                for col_index, year_col in enumerate(year_value_cols):
                    year = year_col.split(' ')[0] 
                    # Ensure corresponding weight column exists, handle potential index out of range if columns are mismatched
                    if col_index < len(year_weight_cols):
                        weight_col = year_weight_cols[col_index]

                        # Union the Value data
                        sql_union_parts.append(f"""
                            SELECT 
                                CAST(HS_6DG AS VARCHAR) AS HS_6DG, 
                                ENG_DESC, 
                                CAST(COUNTRY AS VARCHAR) AS COUNTRY, 
                                '{year}' AS Year, 
                                'Value (OMR)' AS Metric, 
                                CAST(COALESCE("{year_col}", '0') AS FLOAT) AS Amount,
                                '{trade_type}' AS Trade_Type
                            FROM df_{safe_trade_type}
                        """)
                        
                        # Union the Weight data
                        sql_union_parts.append(f"""
                            SELECT 
                                CAST(HS_6DG AS VARCHAR) AS HS_6DG, 
                                ENG_DESC, 
                                CAST(COUNTRY AS VARCHAR) AS COUNTRY, 
                                '{year}' AS Year, 
                                'Weight (KG)' AS Metric, 
                                CAST(COALESCE("{weight_col}", '0') AS FLOAT) AS Amount,
                                '{trade_type}' AS Trade_Type
                            FROM df_{safe_trade_type}
                        """)
        
            if sql_union_parts:
                full_sql_query = " UNION ALL ".join(sql_union_parts)
                df_trade = con.execute(full_sql_query).fetchdf()
        
        con.close()
    
    # Final cleaning of df_trade
    if not df_trade.empty:
        df_trade['Amount'] = df_trade['Amount'].fillna(0).astype(np.float32)
        df_trade['HS_6DG'] = df_trade['HS_6DG'].astype('category')
        df_trade['COUNTRY'] = df_trade['COUNTRY'].astype('category')
        df_trade['Trade_Type'] = df_trade['Trade_Type'].astype('category')
        df_trade['Year'] = df_trade['Year'].astype('category')
        df_trade['HS_4DG'] = df_trade['HS_6DG'].str[:4].astype('category')

    # 4. Process new files for Key Players Page
    
    # 4.1 Production Shift Data (3a)
    df_prod_shift_proc = pd.DataFrame()
    # Check for all three files
    if df_prod_1980 is not None and df_prod_2016 is not None and df_prod_2022 is not None:
        df_prod_shift_proc = pd.concat([df_prod_1980, df_prod_2016, df_prod_2022], ignore_index=True)
        if not df_prod_shift_proc.empty:
            df_prod_shift_proc['Year'] = df_prod_shift_proc['Year'].astype(str).astype('category')
            # The percentage is already a decimal (e.g., 0.22) so we keep it as a float
            df_prod_shift_proc['Percentage'] = pd.to_numeric(df_prod_shift_proc['Percentage'], errors='coerce')

    # 4.2 Alum Imports/Exports Share (3b)
    df_alum_ie_proc = df_alum_ie.copy() if df_alum_ie is not None else pd.DataFrame()
    if not df_alum_ie_proc.empty and df_alum_ie_proc.shape[1] >= 3:
        # FIX B: Clean and rename columns for robust melting and ensure numeric type
        # The file content shows 'Year', 'Aluminum share of total exports and re-exports', 'Aluminum share of total imports'
        df_alum_ie_proc.columns = ['Year', 'Export_Share_Raw', 'Import_Share_Raw']
        df_alum_ie_proc['Year'] = df_alum_ie_proc['Year'].astype(str)
        # The raw data is already in percentage format (e.g., 4.97) but will be divided by 100 for consistent plotting logic later.
        # Since the provided plotting logic multiplies by 100, we need to convert the raw percentage to a fraction first for correct math.
        df_alum_ie_proc['Export_Share'] = pd.to_numeric(df_alum_ie_proc['Export_Share_Raw'], errors='coerce') / 100.0
        df_alum_ie_proc['Import_Share'] = pd.to_numeric(df_alum_ie_proc['Import_Share_Raw'], errors='coerce') / 100.0
        # Drop the raw columns after conversion
        df_alum_ie_proc = df_alum_ie_proc.drop(columns=['Export_Share_Raw', 'Import_Share_Raw'])
    
    # 4.3 Alum Trading Partners 2023 (3c)
    df_partner_2023_proc = pd.DataFrame()
    if df_partner_exp_2023 is not None and df_partner_imp_2023 is not None:
        df_partner_2023_combined = pd.concat([
            df_partner_exp_2023.assign(Trade_Type='Export/Re-export'),
            df_partner_imp_2023.assign(Trade_Type='Import')
        ], ignore_index=True)
        # FIX C: Ensure Percentage is numeric (it's a fraction in the file, e.g., 0.205)
        if not df_partner_2023_combined.empty and 'Percentage' in df_partner_2023_combined.columns:
            df_partner_2023_combined['Percentage'] = pd.to_numeric(df_partner_2023_combined['Percentage'], errors='coerce')
        df_partner_2023_proc = df_partner_2023_combined
    
    # 4.4 HS Partner Deep Dive (3d)
    df_hs_partners_proc = pd.DataFrame()
    hs_partner_files = [
        (df_hs_exp_7601, 'Export/Re-export'), (df_hs_exp_7604, 'Export/Re-export'), (df_hs_exp_7605, 'Export/Re-export'),
        (df_hs_imp_7601, 'Import'), (df_hs_imp_7604, 'Import'), (df_hs_imp_7605, 'Import'),
    ]
    # Filter out None or empty DataFrames, then assign the 'Type' column
    valid_hs_partner_data = [df.assign(Type=t) for df, t in hs_partner_files if df is not None and not df.empty]
    
    if valid_hs_partner_data:
        df_hs_partners_proc = pd.concat(valid_hs_partner_data, ignore_index=True)
        # FIX D: Ensure Percentage is numeric (it's a fraction in the file, e.g., 0.164)
        if not df_hs_partners_proc.empty and 'Percentage' in df_hs_partners_proc.columns:
            df_hs_partners_proc['Percentage'] = pd.to_numeric(df_hs_partners_proc['Percentage'], errors='coerce')
            df_hs_partners_proc['HS Code'] = df_hs_partners_proc['HS Code'].astype(str).astype('category')
            df_hs_partners_proc['Type'] = df_hs_partners_proc['Type'].astype('category')


    return (
        df_company_hs, df_alum_co, df_trade,
        df_prod_shift_proc, df_alum_ie_proc, df_partner_2023_proc, df_hs_partners_proc
    )

# --- Remaining Functions (UNCHANGED) ---

# Approximate center coordinates for Oman's Governorates (Used as proxy for Villages)
GOVERNORATE_COORDS = {
    'محافظة مسقط': (23.5859, 58.3180), 'محافظة مسندم': (26.0000, 56.2500),
    'محافظة ظفار': (17.0167, 54.1167), 'محافظة شمال الباطنة': (24.4167, 56.6667),
    'محافظة جنوب الباطنة': (23.5000, 57.5000), 'محافظة الداخلية': (22.7667, 57.5333),
    'محافظة شمال الشرقية': (22.5000, 58.7500), 'محافظة جنوب الشرقية': (21.5000, 59.2500),
    'محافظة الظاهرة': (23.0000, 56.5000), 'محافظة البريمي': (24.2500, 55.7500),
    'محافظة الوسطى': (19.8333, 56.0000),
}

# HIGH-SPEED CACHE: Village-Level Deep Dive Map Data
@st.cache_data
def get_deep_dive_map_data(df_filtered_4dg):
    """Caches the heavy groupby for the Omani Companies map (Village Level)."""
    
    if df_filtered_4dg is None or df_filtered_4dg.empty:
        return pd.DataFrame()
        
    df_map_data_prep = df_filtered_4dg[['Governorate', 'Village', 'Record Name']].drop_duplicates().copy()

    df_map_data = df_map_data_prep.groupby(['Governorate', 'Village']).agg(
        company_count=('Record Name', 'nunique'),
        company_names=('Record Name', lambda x: '<br>'.join(sorted(x.unique().astype(str)))) 
    ).reset_index()

    df_map_data['lat'] = df_map_data['Governorate'].map(lambda x: GOVERNORATE_COORDS.get(x, (21.0, 57.0))[0])
    df_map_data['lon'] = df_map_data['Governorate'].map(lambda x: GOVERNORATE_COORDS.get(x, (21.0, 57.0))[1])

    df_map_data['Location'] = df_map_data['Village'].astype(str) + ' (' + df_map_data['Governorate'].astype(str) + ')'

    return df_map_data

# HIGH-SPEED CACHE: Trade Flow Analysis (for Bar Chart)
@st.cache_data
def get_trade_flow_data(df_trade, selected_hs_6dg, selected_year, selected_metric):
    """Filters, aggregates, and caches the result for the Trade Flow Bar Chart."""
    
    if df_trade.empty:
        return None, None
        
    df_bar_chart = df_trade[
        (df_trade['HS_6DG'] == selected_hs_6dg) &
        (df_trade['Year'] == selected_year) &
        (df_trade['Metric'] == selected_metric)
    ].copy()

    if df_bar_chart.empty:
        return None, None
    
    desc = df_bar_chart['ENG_DESC'].iloc[0] if not df_bar_chart.empty else "No description available."
    
    df_chart_agg = df_bar_chart.groupby(['COUNTRY', 'Trade_Type'])['Amount'].sum().reset_index()
    df_chart_agg = df_chart_agg[df_chart_agg['Amount'] > 0]
    
    return df_chart_agg, desc

# HIGH-SPEED CACHE: Trend Analysis (MODIFIED for combined line chart)
@st.cache_data
def get_trend_analysis_data_combined(df_trade, selected_hs_6dg_trend, selected_country):
    """Filters, aggregates, and caches the result for the Combined Time Series Line Chart (all 3 trade types)."""
    
    if df_trade.empty:
        return None
        
    # Filter for selected HS Code, Country, and only Value (OMR) metric
    df_trend = df_trade[
        (df_trade['HS_6DG'] == selected_hs_6dg_trend) &
        (df_trade['COUNTRY'] == selected_country) &
        (df_trade['Metric'] == 'Value (OMR)')
    ].copy()

    if df_trend.empty:
        return None
    
    # Group by Year and Trade_Type to get separate lines
    df_trend_agg = df_trend.groupby(['Year', 'Trade_Type'])['Amount'].sum().reset_index()
    
    return df_trend_agg


# HIGH-SPEED CACHE: HS Partner Deep Dive Data
@st.cache_data
def get_hs_partner_deep_dive_data(df_hs_partners, selected_hs_code, selected_type):
    """Filters and aggregates data for the specific HS partner bar chart."""
    
    if df_hs_partners.empty:
        return pd.DataFrame()
    
    df_filtered = df_hs_partners[
        (df_hs_partners['HS Code'] == selected_hs_code) &
        (df_hs_partners['Type'] == selected_type)
    ].copy()
    
    if df_filtered.empty:
        return pd.DataFrame()

    # Sort and take the top partners
    df_filtered = df_filtered.sort_values(by='Percentage', ascending=False).head(10)
    
    return df_filtered


# --- 3. PAGE 1: HS Code Deep Dive & Trade Analysis ---
def page_hs_deep_dive(df_company_hs, df_trade):
    st.title("Product-Centric Analysis: HS Code Deep Dive")
    st.markdown("---")
    
    if df_company_hs is None or df_company_hs.empty:
        st.info("No Company Data available. Please upload the data and click 'Process & Build Dashboard' in the sidebar.")
        return

    hs_4dg_options = sorted(df_company_hs['HS_4DG'].dropna().astype(str).unique().tolist())
    selected_hs_4dg = st.selectbox(
        "Select 4-Digit HS Code",
        options=hs_4dg_options,
        help="Choose a primary product category to see affiliated companies and trade details.",
        key="main_hs4_select"
    )

    df_filtered_4dg = df_company_hs[df_company_hs['HS_4DG'] == selected_hs_4dg].copy()
    
    if df_filtered_4dg.empty:
        st.warning("No company data found for the selected HS 4-Digit Code.")
        return

    # --- Company Info & Map (Village Level) ---
    st.subheader(f"Associated Company & Product Details (HS {selected_hs_4dg})")
    category_name = df_filtered_4dg['Category'].iloc[0] if not df_filtered_4dg.empty else "N/A"
    st.info(f"**Category:** {category_name}")
    company_details = df_filtered_4dg[['Record Name', 'Registration Number', 'Legal Form', 'Activity Name', 'Governorate', 'Village']].drop_duplicates().reset_index(drop=True)
    st.dataframe(company_details, use_container_width=True, hide_index=True)

    st.subheader("Omani Companies Geographical Distribution (Village Level)")
    
    df_map_data = get_deep_dive_map_data(df_filtered_4dg)
    
    if df_map_data.empty:
        st.info("No location data found for companies with the selected HS Code.")
    else:
        fig_map = px.scatter_mapbox(
            df_map_data, lat="lat", lon="lon", hover_name="Location",
            hover_data={"company_names": True, "company_count": True, "lat": False, "lon": False}, 
            color="company_count", size="company_count",
            color_continuous_scale=px.colors.sequential.Plotly3, zoom=5.5,
            center={"lat": 21.0, "lon": 57.0}, height=500,
            title="Company Location by Village (Point size shows company count)"
        )
        
        fig_map.update_layout(
            mapbox_style="open-street-map", 
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) 
        )
        fig_map.update_traces(
            hovertemplate="<b>%{hovertext}</b><br><br>Total Companies: %{customdata[1]}<br>--- Companies ---<br>%{customdata[0]}<extra></extra>",
            customdata=df_map_data[['company_names', 'company_count']]
        )
        st.plotly_chart(fig_map, use_container_width=True)


    # --- Trade Analysis Bar Chart (with Submit Button) ---
    st.subheader("Global Trade Flow Analysis")
    
    if df_trade.empty:
        st.info("Trade Data (Imports, Exports, Re-exports) is required to view global flow analysis.")
        return

    df_trade_filtered_4dg = df_trade[df_trade['HS_4DG'] == selected_hs_4dg].copy()

    with st.form(key='trade_flow_form'):
        col_trade_1, col_trade_2, col_trade_3 = st.columns(3)
        
        # Robust options
        hs_6dg_options_flow = sorted(df_trade_filtered_4dg['HS_6DG'].dropna().astype(str).unique().tolist()) if 'HS_6DG' in df_trade_filtered_4dg.columns and not df_trade_filtered_4dg.empty else ["(No HS Code data)"]
        year_options = sorted(df_trade_filtered_4dg['Year'].dropna().astype(str).unique().tolist()) if 'Year' in df_trade_filtered_4dg.columns and not df_trade_filtered_4dg.empty else ["(No Year data)"]
        metric_options = df_trade_filtered_4dg['Metric'].dropna().astype(str).unique().tolist() if 'Metric' in df_trade_filtered_4dg.columns and not df_trade_filtered_4dg.empty else ["(No Metric data)"]

        with col_trade_1:
            selected_hs_6dg = st.selectbox("Select 6-Digit HS Code", options=hs_6dg_options_flow)

        with col_trade_2:
            selected_year = st.selectbox("Select Year", options=year_options)
            
        with col_trade_3:
            selected_metric = st.selectbox("Select Metric", options=metric_options)
        
        submitted = st.form_submit_button("Submit Analysis")

    if submitted:
        if selected_hs_6dg.startswith("(No") or selected_year.startswith("(No") or selected_metric.startswith("(No"):
            st.warning("Cannot generate chart: Missing data for one or more selections.")
            return

        with st.spinner("Generating trade analysis chart..."):
            df_chart_agg, desc = get_trade_flow_data(df_trade, selected_hs_6dg, selected_year, selected_metric)
            
            st.markdown(f"**Product Description:** *{desc if desc else 'No description available.'}*")

            if df_chart_agg is not None and not df_chart_agg.empty:
                country_order = df_chart_agg.groupby('COUNTRY')['Amount'].sum().sort_values(ascending=False).index.tolist()

                fig_bar = px.bar(
                    df_chart_agg, x='COUNTRY', y='Amount', color='Trade_Type', barmode='group',
                    title=f"Trade Flow for HS {selected_hs_6dg} in {selected_year} by Country ({selected_metric})",
                    labels={'Amount': selected_metric, 'COUNTRY': 'Trade Partner Country'},
                    color_discrete_map={'Import': '#0070c0', 'Export': '#00b050', 'Re-export': '#ffc000'},
                    height=500
                )
                fig_bar.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': country_order})
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("No trade data found for the selected parameters.")
    else:
        st.info("Select your trade filters above and click 'Submit Analysis' to view the chart.")


# --- 4. PAGE 2: Trend Analysis (UPDATED) ---
def page_trend_analysis(df_trade):
    st.title("Time-Series Trade Trend Analysis (Value in OMR)")
    st.markdown("---")
    
    if df_trade.empty:
        st.info("Trade Data (Imports, Exports, Re-exports) is required to view trend analysis. Please upload the data and click 'Process & Build Dashboard' in the sidebar.")
        return

    is_trade_data_available = not df_trade.empty and all(col in df_trade.columns for col in ['HS_6DG', 'Trade_Type', 'COUNTRY'])

    if is_trade_data_available:
        all_country_options = sorted(df_trade['COUNTRY'].dropna().astype(str).unique().tolist())
        hs_6dg_options = sorted(df_trade['HS_6DG'].dropna().astype(str).unique().tolist())
    else:
        hs_6dg_options = []
        all_country_options = []

    hs_6dg_display = hs_6dg_options if hs_6dg_options else ["(No HS Code data)"]
    country_options_display = all_country_options if all_country_options else ["(No country data)"]

    # --- Trend Filter Form ---
    with st.form(key='trend_analysis_form'):
        col_trend_1, col_trend_2 = st.columns(2)
    
        with col_trend_1:
            selected_hs_6dg_trend = st.selectbox("Select 6-Digit HS Code", options=hs_6dg_display)
            
        with col_trend_2:
            selected_country = st.selectbox("Select Country", options=country_options_display)

        submitted = st.form_submit_button("Generate Combined Trend Chart")

    if submitted:
        if selected_hs_6dg_trend.startswith("(No") or selected_country.startswith("(No"):
            st.warning("Cannot generate chart: Missing data for one or more selections.")
            return

        with st.spinner("Generating combined trend analysis chart..."):
            
            # Uses MODIFIED CACHED function to get all 3 trade types
            df_trend_agg = get_trend_analysis_data_combined(df_trade, selected_hs_6dg_trend, selected_country)

            st.markdown(f"### Yearly Trend (Value in OMR) for HS {selected_hs_6dg_trend} with **{selected_country}**")

            if df_trend_agg is not None and not df_trend_agg.empty:
                
                # Define custom color map
                trade_color_map = {'Import': '#FF4B4B', 'Export': '#00b050', 'Re-export': '#0070c0'} # Red, Green, Blue
                
                fig_line = px.line(
                    df_trend_agg, x='Year', y='Amount', color='Trade_Type',
                    title=f"Trade Value Trend Over Time (Value in OMR) for {selected_country}",
                    labels={'Amount': 'Trade Value (OMR)', 'Trade_Type': 'Flow'},
                    markers=True, line_shape='spline', color_discrete_map=trade_color_map, height=550
                )
                fig_line.update_yaxes(tickprefix='OMR ', separatethousands=True) 
                fig_line.update_xaxes(dtick=1, tickformat='d')
                
                # Enhance legend title
                fig_line.update_layout(legend_title_text='Trade Flow Type')
                
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.warning(f"No trade data (Value in OMR) found for the selected parameters.")
    else:
        st.info("Select your HS Code and Country above and click 'Generate Combined Trend Chart' to view all three trade flows on one graph.")


# --- 5. VISUALIZATION UTILITIES for PAGE 3 ---

# 5.1. Plot (a): Global Production Shift (Uses df_prod_shift_proc)
def plot_production_shift(df_prod_shift):
    st.subheader("a. Global Aluminium Production Share Shift")
    
    if df_prod_shift.empty:
        st.info("Production Shift Data (1980, 2016, 2022) is missing or incomplete.")
        return
        
    years = sorted(df_prod_shift['Year'].unique().tolist())
    
    # Use selectbox for better space management
    selected_year = st.selectbox(
        "Select Year to view Global/Regional Production Share:",
        options=years,
        key="prod_shift_year"
    )
    
    df_filtered = df_prod_shift[df_prod_shift['Year'] == selected_year]
    
    # Ensure percentages are numeric (already a fraction in the data: 0.22)
    df_filtered['Percentage_Display'] = df_filtered['Percentage'] * 100 

    fig_pie = px.pie(
        df_filtered, values='Percentage_Display', names='Region',
        title=f"Global Aluminium Production Share in {selected_year}",
        hover_data={'Percentage_Display': ':.1f'}, labels={'Percentage_Display': 'Share'},
        height=550,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    
    fig_pie.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
    st.plotly_chart(fig_pie, use_container_width=True)

# 5.2. Plot (b): Imports/Exports Share (Uses df_alum_ie_proc)
def plot_imports_exports_share(df_alum_ie):
    st.subheader("b. Aluminium Trade Share of Oman's Total Trade (2012-2023)")
    
    if df_alum_ie.empty:
        st.info("Aluminium Imports/Exports Share Data is missing.")
        return
        
    df_ie_melt = df_alum_ie.melt(
        id_vars='Year',
        value_vars=['Export_Share', 'Import_Share'], # Use cleaned column names (fractions)
        var_name='Trade Type',
        value_name='Percentage'
    )
    
    # Replace melted names with descriptive labels for plotting
    df_ie_melt['Trade Type'] = df_ie_melt['Trade Type'].replace({
        'Export_Share': 'Aluminum share of total exports and re-exports',
        'Import_Share': 'Aluminum share of total imports'
    })

    # Convert the fractional percentage back to a display percentage (e.g., 0.0497 -> 4.97)
    df_ie_melt['Percentage'] = df_ie_melt['Percentage'] * 100 
    
    fig_line = px.line(
        df_ie_melt, x='Year', y='Percentage', color='Trade Type',
        title="Aluminium Share of Total National Imports and Exports/Re-exports (2012-2023)",
        labels={'Percentage': 'Percentage Share (%)', 'Year': 'Year'},
        markers=True,
        color_discrete_map={
            'Aluminum share of total exports and re-exports': '#00b050', # Green
            'Aluminum share of total imports': '#FF4B4B' # Red
        },
        height=450
    )
    # Correct property used: 'ticksuffix'
    fig_line.update_yaxes(ticksuffix='%', rangemode='tozero')
    fig_line.update_xaxes(dtick=1, tickformat='d')
    st.plotly_chart(fig_line, use_container_width=True)

# 5.3. Plot (c): Top 10 Trading Partners 2023 (Uses df_partner_2023_proc)
def plot_2023_partners(df_partner_2023_combined):
    st.subheader("c. Top 10 Trading Partners for All Aluminium (2023)")
    
    if df_partner_2023_combined.empty:
        st.info("2023 Trading Partners Data is missing or incomplete.")
        return
        
    col_exp, col_imp = st.columns(2)
    
    # Sort and take the top 10 (Percentage is a fraction: 0.205)
    df_exp = df_partner_2023_combined[df_partner_2023_combined['Trade_Type'] == 'Export/Re-export'].sort_values(by='Percentage', ascending=False).head(10).copy()
    df_imp = df_partner_2023_combined[df_partner_2023_combined['Trade_Type'] == 'Import'].sort_values(by='Percentage', ascending=False).head(10).copy()
    
    # Export/Re-export Chart
    with col_exp:
        fig_exp = px.bar(
            df_exp, x='Partner', y='Percentage',
            title='Top Export/Re-export Destinations (2023)',
            labels={'Percentage': 'Share of Total Alum. Exports', 'Partner': 'Country'},
            color_discrete_sequence=['#00b050']
        )
        # Use tickformat to display fraction as percentage
        fig_exp.update_yaxes(tickformat=".1%", rangemode='tozero')
        fig_exp.update_layout(xaxis={'categoryorder': 'total descending'}, height=500)
        st.plotly_chart(fig_exp, use_container_width=True)

    # Import Chart
    with col_imp:
        fig_imp = px.bar(
            df_imp, x='Partner', y='Percentage',
            title='Top Import Sources (2023)',
            labels={'Percentage': 'Share of Total Alum. Imports', 'Partner': 'Country'},
            color_discrete_sequence=['#FF4B4B']
        )
        # Use tickformat to display fraction as percentage
        fig_imp.update_yaxes(tickformat=".1%", rangemode='tozero')
        fig_imp.update_layout(xaxis={'categoryorder': 'total descending'}, height=500)
        st.plotly_chart(fig_imp, use_container_width=True)

# 5.4. Plot (d): Interactive HS Partner Deep Dive (Uses df_hs_partners_proc)
def plot_hs_partner_deep_dive(df_hs_partners):
    st.subheader("d. Interactive Deep Dive: Top 10 Trading Partners by HS Code (2023)")
    
    if df_hs_partners.empty:
        st.info("HS Code Specific Partner Data (7601, 7604, 7605) is missing or incomplete.")
        return
        
    # Column names 'Country', 'Percentage', 'HS Code', 'Type' are consistent
    hs_options = sorted(df_hs_partners['HS Code'].unique().tolist())
    trade_type_options = sorted(df_hs_partners['Type'].unique().tolist())

    col_hs, col_type = st.columns(2)
    with col_hs:
        selected_hs = st.selectbox("Select HS Code for Deep Dive (7601/7604/7605)", options=hs_options, key="deep_dive_hs")
    with col_type:
        selected_type = st.selectbox("Select Trade Flow Type", options=trade_type_options, key="deep_dive_type")

    # This uses cached data function to get the top 10 for the selected combination
    df_chart = get_hs_partner_deep_dive_data(df_hs_partners, selected_hs, selected_type)
    
    if df_chart.empty:
        st.warning(f"No top 10 partner data found for HS {selected_hs} - {selected_type}.")
        return

    # Determine color based on trade type
    chart_color = '#00b050' if selected_type == 'Export/Re-export' else '#FF4B4B'
    title_prefix = 'Top Export/Re-export Destinations' if selected_type == 'Export/Re-export' else 'Top Import Sources'

    fig_bar = px.bar(
        df_chart, x='Country', y='Percentage',
        title=f"{title_prefix} for HS {selected_hs} in 2023",
        labels={'Percentage': 'Share of HS Trade (%)', 'Country': 'Trade Partner Country'},
        color_discrete_sequence=[chart_color]
    )
    
    # Use tickformat to display fraction as percentage
    fig_bar.update_yaxes(tickformat=".1%", rangemode='tozero')
    fig_bar.update_layout(xaxis={'categoryorder': 'total descending'}, height=550)
    st.plotly_chart(fig_bar, use_container_width=True)


# --- 6. NEW PAGE DEFINITION: Aluminium Key Players (Fixes NameError) ---
def page_key_players(df_alum_co, df_prod_shift, df_alum_ie, df_partner_2023, df_hs_partners):
    st.title("Aluminium Market: Key Players, Global Context, and Trade Partners")
    st.markdown("---")

    # --- Section: Map of Aluminium Companies ---
    st.header("Omani Aluminium Industry Footprint")
    if df_alum_co is not None and not df_alum_co.empty:
        st.info(f"Showing **{df_alum_co['name'].nunique()}** Key Aluminium Companies and their capacity.")
        
        # Plotting the map
        fig_map = px.scatter_mapbox(
            df_alum_co.dropna(subset=['lat', 'lon']), 
            lat="lat", lon="lon", hover_name="name",
            hover_data={"name": True, "capacity_tpy": ':.0f', "lat": False, "lon": False}, 
            color="name", size="capacity_tpy",
            color_discrete_sequence=px.colors.qualitative.Dark24, 
            zoom=5.5,
            center={"lat": 21.0, "lon": 57.0}, 
            height=500,
            title="Location and Capacity (tpy) of Key Aluminium Companies"
        )
        
        fig_map.update_layout(
            mapbox_style="open-street-map", 
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Aluminium Companies Data (Key Players) is missing.")

    st.markdown("---")

    # --- Section: New Visualizations (a, b, c, d) ---
    st.header("Global and National Aluminium Trade Dynamics")
    
    # Plot (a): Production Shift
    plot_production_shift(df_prod_shift)
    st.markdown("---")
    
    # Plot (b): Aluminium Share of Total Trade (Fixed for 'ticksuffix')
    plot_imports_exports_share(df_alum_ie)
    st.markdown("---")

    # Plot (c): Top 10 Trading Partners 2023
    plot_2023_partners(df_partner_2023)
    st.markdown("---")

    # Plot (d): Interactive HS Partner Deep Dive
    plot_hs_partner_deep_dive(df_hs_partners)
    st.markdown("---")


# --- 7. NAVIGATION & MAIN APP LOGIC (UPDATED FOR ZIP) ---

st.sidebar.title("Sultanate of Oman")
st.sidebar.markdown("### 1. Upload Data")

# Initialize session state for data storage
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# --- File Upload Form (MODIFIED for single ZIP upload) ---
with st.sidebar.form(key='data_upload_form'):
    st.markdown("##### Upload a single ZIP file containing all 17 data files.")
    uploaded_zip = st.file_uploader(
        "Upload Oman data.zip", 
        type=['zip'], 
        help="The ZIP file must contain all 17 CSV/Excel/Parquet files."
    )

    submitted = st.form_submit_button("2. Process & Build Dashboard 🚀")

# --- Submission Logic ---
if submitted:
    if uploaded_zip is None:
        st.error("Please upload the ZIP file before processing.")
        st.session_state.processed_data = None
    else:
        # Step 1: Unzip and Load all 17 DataFrames
        with st.spinner("Step 1/2: Extracting files from ZIP and loading into memory..."):
            # Call the new function that returns the list of 17 DataFrames in order
            loaded_dfs = unzip_and_load_data(uploaded_zip)

        # Check if core files failed to load (a simple check that something was loaded)
        if all(df is None for df in loaded_dfs) and uploaded_zip is not None:
             st.error("Failed to load any of the required dataframes from the ZIP. Please check your file names inside the archive.")
             st.session_state.processed_data = None
        else:
            # Step 2: Processing (Heavy Lifting)
            with st.spinner("Step 2/2: Optimizing and processing data with **DuckDB**..."):
                try:
                    # Unpack the list of loaded DataFrames for the process_data function
                    processed_data_tuple = process_data(*loaded_dfs)
                    st.session_state.processed_data = processed_data_tuple
                    st.success("Data Processing Complete! You can now navigate the dashboard pages.")
                    st.rerun() 
                except Exception as e:
                    st.error(f"An error occurred during data processing. Check file formats/integrity: {e}")
                    st.session_state.processed_data = None
                    st.cache_data.clear()

# --- Dashboard Display Logic ---
if st.session_state.processed_data is not None:
    # Unpack processed data from session state
    (
        df_company_hs_proc, df_alum_co_proc, df_trade_proc,
        df_prod_shift_proc, df_alum_ie_proc, df_partner_2023_proc, df_hs_partners_proc
    ) = st.session_state.processed_data
    
    # Check if critical data is missing and provide a warning
    if df_trade_proc.empty or df_company_hs_proc is None or df_alum_co_proc is None:
        st.sidebar.warning("⚠️ Some critical data (Trade/Company) is missing. Parts of the dashboard may be empty.")
    else:
        st.sidebar.success("Data is Ready! (7 Processed DataFrames in memory)")
        
    st.sidebar.markdown("---")
    
    page_selection = st.sidebar.radio(
        "Go to Dashboard Page",
        ["Product Deep Dive", "Trade Trend Analysis", "Aluminium Key Players"]
    )
    st.sidebar.markdown("---")

    if page_selection == "Product Deep Dive":
        page_hs_deep_dive(df_company_hs_proc, df_trade_proc)
    elif page_selection == "Trade Trend Analysis":
        page_trend_analysis(df_trade_proc)
    elif page_selection == "Aluminium Key Players":
        # Call the newly defined function, passing all required dataframes
        page_key_players(
            df_alum_co_proc, 
            df_prod_shift_proc, 
            df_alum_ie_proc, 
            df_partner_2023_proc,
            df_hs_partners_proc
        )
else:
    # Initial instruction state
    st.title("Oman Building Material Market Overview")
    st.info("To begin, please upload the single **ZIP file** containing all 17 required data files in the sidebar on the left and then click the **'Process & Build Dashboard'** button. The dashboard will appear once processing is complete.")