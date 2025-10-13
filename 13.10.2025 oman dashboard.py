import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from io import BytesIO
import duckdb 

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Oman Building Material Market Overview",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. DATA LOADING & CORE PROCESSING UTILITIES ---

def load_uploaded_file(uploaded_file):
    """
    Loads uploaded files, prioritizing fast columnar formats (Parquet/Feather) 
    over standard formats (CSV, Excel).
    """
    if uploaded_file is None:
        return None
    
    try:
        file_name = uploaded_file.name.lower()
        
        if file_name.endswith(('.parquet', '.feather')):
            # HYPEREXTRACT EQUIVALENT: Fast columnar read
            df = pd.read_parquet(uploaded_file)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, sheet_name=0)
        elif file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, engine='c')
        else:
            st.error(f"Unsupported file type for {uploaded_file.name}. Please upload a Parquet, Feather, CSV, or Excel file.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {e}")
        return None

# Caching the main data processing function for optimal performance.
@st.cache_data(show_spinner="Processing trade data with DuckDB (Fastest OLAP engine)...")
def process_data(df_company_hs, df_alum_co, df_imports, df_exports, df_reexports):
    """Performs all necessary data cleaning and transformation with type optimization."""
    
    # 1. Company Data
    if df_company_hs is not None:
        df_company_hs = df_company_hs.dropna(subset=['HS Code']).copy()
        df_company_hs['HS Code'] = df_company_hs['HS Code'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        df_company_hs['HS_4DG'] = df_company_hs['HS Code'].str[:4].astype('category')
        df_company_hs['Governorate'] = df_company_hs['Governorate'].astype('category')
        df_company_hs['Village'] = df_company_hs['Village'].astype('category') 
    
    # 2. Aluminium Companies Data
    if df_alum_co is not None:
        df_alum_co['lat'] = pd.to_numeric(df_alum_co['lat'], errors='coerce').astype(np.float32)
        df_alum_co['lon'] = pd.to_numeric(df_alum_co['lon'], errors='coerce').astype(np.float32)
        df_alum_co['capacity_tpy'] = pd.to_numeric(df_alum_co['capacity_tpy'], errors='coerce').fillna(0).astype(np.float32)
    
    # 3. Trade Data (Using DuckDB for High-Speed SQL Unpivot/Union)
    trade_data_frames = {"Import": df_imports, "Export": df_exports, "Re-export": df_reexports}
    
    con = duckdb.connect(':memory:')
    sql_union_parts = []
    
    # Get column names from the first non-empty dataframe
    first_df = next((df for df in trade_data_frames.values() if df is not None and not df.empty), None)
    if first_df is not None:
        year_value_cols = [col for col in first_df.columns if '(Value in OMR)' in col]
        year_weight_cols = [col for col in first_df.columns if '(Weight in KG)' in col]
    else:
        return df_company_hs, df_alum_co, pd.DataFrame() # Return empty trade DF if no data
        
    for trade_type, df in trade_data_frames.items():
        if df is None or df.empty: continue
            
        # FIX: Replace hyphen with underscore to create a valid SQL identifier
        safe_trade_type = trade_type.lower().replace('-', '_')
        con.register(f'df_{safe_trade_type}', df)
        
        # SQL for MELT/Unpivot equivalent - highly optimized in DuckDB
        for col_index, year_col in enumerate(year_value_cols):
            year = year_col.split(' ')[0] # Extract '2010' from '2010 (Value in OMR)'
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
    
    if not sql_union_parts:
        df_trade = pd.DataFrame()
    else:
        full_sql_query = " UNION ALL ".join(sql_union_parts)
        df_trade = con.execute(full_sql_query).fetchdf() # Retrieve final DataFrame
    
    con.close()
    
    # Final cleaning of df_trade (mostly handled by SQL, but ensure dtypes)
    if not df_trade.empty:
        df_trade['Amount'] = df_trade['Amount'].fillna(0).astype(np.float32)
        df_trade['HS_6DG'] = df_trade['HS_6DG'].astype('category')
        df_trade['COUNTRY'] = df_trade['COUNTRY'].astype('category')
        df_trade['Trade_Type'] = df_trade['Trade_Type'].astype('category')
        df_trade['Year'] = df_trade['Year'].astype('category')

        df_trade['HS_4DG'] = df_trade['HS_6DG'].str[:4].astype('category')
        
    return df_company_hs, df_alum_co, df_trade

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

# HIGH-SPEED CACHE: Trade Flow Analysis
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

# HIGH-SPEED CACHE: Trend Analysis
@st.cache_data
def get_trend_analysis_data(df_trade, selected_hs_6dg_trend, selected_trade_type, selected_country):
    """Filters, aggregates, and caches the result for the Time Series Line Chart."""
    
    if df_trade.empty:
        return None
        
    df_trend = df_trade[
        (df_trade['HS_6DG'] == selected_hs_6dg_trend) &
        (df_trade['Trade_Type'] == selected_trade_type) &
        (df_trade['COUNTRY'] == selected_country) &
        (df_trade['Metric'] == 'Value (OMR)')
    ].copy()

    if df_trend.empty:
        return None
    
    df_trend_agg = df_trend.groupby(['Year'])['Amount'].sum().reset_index()
    
    return df_trend_agg


# --- 3. PAGE 1: HS Code Deep Dive & Trade Analysis ---
def page_hs_deep_dive(df_company_hs, df_trade):
    st.title("Product-Centric Analysis: HS Code Deep Dive")
    st.markdown("---")
    
    if df_company_hs is None or df_company_hs.empty:
        st.info("Please upload the Company Data file to begin this analysis.")
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
        st.info("Please upload all Trade Data files (Imports, Exports, Re-exports) to view trade analysis.")
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

        with st.spinner("Generating trade analysis chart... (subsequent views will be instant)"):
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


# --- 4. PAGE 2: Trend Analysis ---
def page_trend_analysis(df_trade):
    st.title("Time-Series Trade Trend Analysis (Value in OMR)")
    st.markdown("---")
    
    if df_trade.empty:
        st.info("Please upload all Trade Data files (Imports, Exports, Re-exports) to view trend analysis.")
        return

    # Check for core columns availability
    is_trade_data_available = not df_trade.empty and all(col in df_trade.columns for col in ['HS_6DG', 'Trade_Type', 'COUNTRY'])

    if is_trade_data_available:
        # FIX: Drop NaN values and ensure all entries are strings before sorting
        all_country_options = sorted(df_trade['COUNTRY'].dropna().astype(str).unique().tolist())
        hs_6dg_options = sorted(df_trade['HS_6DG'].dropna().astype(str).unique().tolist())
        trade_type_options = sorted(df_trade['Trade_Type'].dropna().astype(str).unique().tolist())
    else:
        # Default empty lists or placeholders if data is not loaded or incomplete
        hs_6dg_options = []
        trade_type_options = []
        all_country_options = []

    # Prepare the display options for selectbox
    hs_6dg_display = hs_6dg_options if hs_6dg_options else ["(No HS Code data)"]
    trade_type_display = trade_type_options if trade_type_options else ["(No Trade Flow data)"]
    country_options_display = all_country_options if all_country_options else ["(No country data)"]

    with st.form(key='trend_analysis_form'):
        col_trend_1, col_trend_2, col_trend_3 = st.columns(3)
    
        with col_trend_1:
            selected_hs_6dg_trend = st.selectbox("Select 6-Digit HS Code", options=hs_6dg_display)
            
        with col_trend_2:
            selected_trade_type = st.selectbox("Select Trade Flow", options=trade_type_display)
            
        with col_trend_3:
            selected_country = st.selectbox("Select Country", options=country_options_display)

        submitted = st.form_submit_button("Generate Trend Chart")

    if submitted:
        # Check if placeholders were selected (prevents KeyError in cached function)
        if selected_hs_6dg_trend.startswith("(No") or selected_country.startswith("(No") or selected_trade_type.startswith("(No"):
            st.warning("Cannot generate chart: Missing data for one or more selections. Please ensure all trade files are uploaded and contain data.")
            return

        with st.spinner("Generating trend analysis chart... (subsequent views will be instant)"):
            
            # Uses HIGH-SPEED CACHED function
            df_trend_agg = get_trend_analysis_data(df_trade, selected_hs_6dg_trend, selected_trade_type, selected_country)

            st.markdown(f"### Yearly Trend (Value in OMR) for HS {selected_hs_6dg_trend} - {selected_trade_type} with **{selected_country}**")

            if df_trend_agg is not None and not df_trend_agg.empty:
                
                fig_line = px.line(
                    df_trend_agg, x='Year', y='Amount',
                    title=f"Trade Value Trend Over Time (Value in OMR) for {selected_country}",
                    labels={'Amount': f'{selected_trade_type} Value (OMR)'},
                    markers=True, line_shape='spline', color_discrete_sequence=['#FF4B4B'], height=550
                )
                fig_line.update_yaxes(tickprefix='OMR ', separatethousands=True) 
                fig_line.update_xaxes(dtick=1, tickformat='d')
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.warning(f"No trade data (Value in OMR) found for the selected parameters: HS {selected_hs_6dg_trend}, {selected_trade_type} with {selected_country}.")
    else:
        st.info("Select your trend filters above and click 'Generate Trend Chart' to view.")


# --- 5. PAGE 3: Aluminium Industry Key Players ---
def page_key_players(df_alum_co):
    st.title("Aluminium Industry Key Players & Capacity")
    st.markdown("---")
    
    if df_alum_co is None or df_alum_co.empty:
        st.info("Please upload the Aluminium Companies Data file to view this analysis.")
        return

    # 1. Key Player Companies Table
    st.subheader("Key Player Companies & Affiliated HS Codes")
    df_display = df_alum_co[['name', 'segment', 'capacity_tpy', 'hs_codes']].copy()
    df_display.columns = ['Company Name', 'Segment', 'Capacity (TPY)', 'Affiliated HS Codes']
    st.dataframe(df_display.sort_values('Capacity (TPY)', ascending=False), use_container_width=True, hide_index=True)

    # 2. Geospatial Visualization (Key Players Map)
    st.subheader("Geographical Distribution of Key Aluminium Facilities")
    df_map_alum = df_alum_co.dropna(subset=['lat', 'lon']).copy()

    fig_alum_map = px.scatter_mapbox(
        df_map_alum, lat="lat", lon="lon",
        hover_name="name", hover_data=["segment", "capacity_tpy"], 
        color="segment", size="capacity_tpy",
        color_discrete_sequence=px.colors.qualitative.Bold, zoom=6,
        center={"lat": 22.0, "lon": 57.0}, height=600,
        title="Key Aluminium Players in Oman (Size based on Capacity)"
    )
    
    # Black font for legibility
    fig_alum_map.update_layout(
        mapbox_style="open-street-map", 
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        legend=dict(
            orientation="h", 
            yanchor="top", y=0.01, 
            xanchor="center", x=0.5, 
            bgcolor="rgba(255, 255, 255, 0.9)", 
            bordercolor="Black", borderwidth=1,
            font=dict(color='Black') 
        )
    )
    st.plotly_chart(fig_alum_map, use_container_width=True)


# --- 6. NAVIGATION & MAIN APP LOGIC ---
st.sidebar.title("Sultanate of Oman")
st.sidebar.markdown("Upload Data Files")

file_types = ['parquet', 'feather', 'csv', 'xlsx']

uploaded_company = st.sidebar.file_uploader("1. Company Data with HS Codes", type=file_types)
uploaded_alum = st.sidebar.file_uploader("2. Aluminium Companies Data", type=file_types)
st.sidebar.markdown("---")
st.sidebar.markdown("##### Materials Trade Data")
uploaded_imports = st.sidebar.file_uploader("3. Imports Data", type=file_types)
uploaded_exports = st.sidebar.file_uploader("4. Exports Data", type=file_types)
uploaded_reexports = st.sidebar.file_uploader("5. Re-exports Data", type=file_types)

# --- Main Logic ---
if all([uploaded_company, uploaded_alum, uploaded_imports, uploaded_exports, uploaded_reexports]):
    df_company_hs = load_uploaded_file(uploaded_company)
    df_alum_co = load_uploaded_file(uploaded_alum)
    df_imports = load_uploaded_file(uploaded_imports)
    df_exports = load_uploaded_file(uploaded_exports)
    df_reexports = load_uploaded_file(uploaded_reexports)

    if all(df is not None for df in [df_company_hs, df_alum_co, df_imports, df_exports, df_reexports]):
        # This heavy step is run once and aggressively cached.
        df_company_hs_proc, df_alum_co_proc, df_trade_proc = process_data(
            df_company_hs, df_alum_co, df_imports, df_exports, df_reexports
        )
        st.sidebar.success("Data Loaded & Processed! (Hyper-Optimized with DuckDB)")
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
            page_key_players(df_alum_co_proc)
    else:
        st.error("One or more files failed to load. Please check the uploaded file formats.")
else:
    st.title("Oman Building Material Market Overview")
    st.info("To begin, please upload all 5 required data files in the sidebar on the left. Consider converting large trade files to **Parquet (.parquet)** for Hyper-like speed.")