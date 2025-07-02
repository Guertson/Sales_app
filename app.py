import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import numpy as np

# allows for data be saved on reruns of the app
st.set_page_config(
    page_title='Financial Dashboard',
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)
st.title("Sales Data Analysis Dashboard")


months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
colors_obj = {
    "primary": "#01f8df",
    "secondary": "#01ee97",
    "budget": "#01f8df",
    "forecast": "#01ee97"
}

# create a function for caching my csvs
@st.cache_data
def load_data_file(path: str):
    
    if path.endswith('.xlsx'):
        return pd.read_excel(path)
    
    else: 
        return pd.read_csv(path)

# Load all datasets
datasets = {
    'main': load_data_file("./Financial Data Clean 2024.xlsx"),
    'sales_locations': load_data_file("sales_locations_updated.csv"),
    '10_year': load_data_file("./10_year_analysis.csv"),
    'market_share': load_data_file("./market_share.csv"),
    'market_share_projected': load_data_file("./market_share_projected.csv"),
    'seasonal': load_data_file("./seasonal_2024.csv"),
    'city_sales': load_data_file("./city_sales.csv")
}

df = datasets['main']
months_select = df.columns[5:19]

# fromatting the dollar amouns
def format_currency(value, unit='M'):
    
    if unit == 'M':
        return f"${value/1000000:.0f}M"
    elif unit == 'B':
        return f"${value/1000000000:.1f}B"
    else: 
        return f"${value:,.0f}"

def create_horizontal_bar_chart(data, x_col, y_col, title, color=colors_obj['primary']):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data[x_col],
        y=data[y_col],
        orientation='h',
        marker_color=color,
        text=[format_currency(v) for v in data[x_col]],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Sales ($)',
        yaxis_title=y_col.replace('_', ' ').title(),
        height=300 if len(data) <= 4 else 400,
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig

def create_line_chart(data, x_col, y_col, title, color=colors_obj['primary']):
    
    fig = px.line(
        data, x=x_col, y=y_col, markers=True, title=title,
        color_discrete_sequence=[color]
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_layout(height=400)
    return fig

def create_tabbed_charts(data, chart_func, categories, title_template):
    
    tabs = st.tabs([cat.title() for cat in categories.keys()])
    
    for tab, (category, column) in zip(tabs, categories.items()):
        with tab:
            if category == 'total' and 'Total_Sales' not in data.columns:
                
                data['Total_Sales'] = sum(data[col] for col in data.columns 
                                        if any(x in col for x in ['Software', 'Hardware', 'Advertising']))
            
            title = title_template.format(category.title())
            fig = chart_func(data, column, 'City' if 'City' in data.columns else 'Season', title)
            st.plotly_chart(fig, use_container_width=True)


def pydeck_sales_map():
    st.subheader("2025 Q1 Sales Locations")
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=datasets['sales_locations'],
        get_position=["longitude", "latitude"],
        get_color=[19, 164, 171, 200],
        get_radius=8000,
        radius_scale=1,
        radius_min_pixels=4,
        radius_max_pixels=20,
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        line_width_min_pixels=1,
        get_line_color=[255, 255, 255, 100]
    )
    
    view_state = pdk.ViewState(
        latitude=39.8283, longitude=-98.5795, zoom=3.5, pitch=0, bearing=0
    )
    
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="dark",
        tooltip={
            "html": "<b>Department:</b> {Department}<br/>"
                   "<b>Nearest City:</b> {Nearest Metropolitan Area}<br/>"
                   "<b>Zip Code:</b> {Zip}<br/>"
                   "<b>Sale Amount:</b> ${sale_amount}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white",
                "fontSize": "12px",
                "padding": "10px",
                "borderRadius": "5px"
            }
        }
    )
    
    st.pydeck_chart(deck, use_container_width=True)

def top_cities_sales_plot():
    st.subheader("2024 Top 10 Cities with Highest Sales")
    
    categories = {
        'total': 'Total_Sales',
        'software': 'Software_Sales', 
        'hardware': 'Hardware_Sales',
        'advertising': 'Advertising_Sales'
    }
    
    def create_city_chart(data, x_col, y_col, title):
        top_data = data.nlargest(10, x_col)
        return create_horizontal_bar_chart(top_data, x_col, y_col, f"{title}")
    
    create_tabbed_charts(datasets['city_sales'], create_city_chart, categories, "Top 10 Cities - {}")

def budget_forecast_plot():
        db_query = duckdb.connect()
        db_query.register('df_table', df)
        
        sales_df = db_query.execute(f"""
            WITH sales_df AS (
                SELECT Scenario, {','.join([f'"{month}"' for month in months])}
                FROM df_table
                WHERE Year = '2025' AND Account = 'Sales' AND business_unit = 'Software'
            )
            UNPIVOT sales_df ON {','.join([f'"{month}"' for month in months])}
            INTO NAME month VALUE sales
        """).fetchdf()
        
        db_query.close()
        
        fig = px.line(
            sales_df, x="month", y="sales", color="Scenario", markers=True,
            text="sales", title="2025 Monthly Budget vs. Forecast",
            color_discrete_map={"Budget": colors_obj['budget'], "Forecast": colors_obj['forecast']}
        )
        fig.update_traces(textposition='top center', textfont=dict(size=10))
        fig.update_layout(height=500)  
        st.plotly_chart(fig, use_container_width=True)
    
    

def yearly_per_account_plot():
    st.subheader("10-Year Sales Trends")
    
    categories = {
        'total': 'Total_Sales',
        'software': 'Software Sales',
        'hardware': 'Hardware Sales', 
        'advertising': 'Advertising Sales'
    }
    
    def create_yearly_chart(data, x_col, y_col, title):
        return create_line_chart(data, 'Year', x_col, f"{title}")
    
    create_tabbed_charts(datasets['10_year'], create_yearly_chart, categories, "{} Sales")

def seasonal_sales_plot():
    st.subheader("2024 Seasonal Sales Analysis")
    
    categories = {
        'total': 'Total_Sales',
        'software': 'Software_Sales',
        'hardware': 'Hardware_Sales',
        'advertising': 'Advertising_Sales'
    }
    
    def create_seasonal_chart(data, x_col, y_col, title):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data[x_col],
            y=data[y_col],
            orientation='h',
            marker_color=colors_obj['primary'],
            text=[format_currency(v) for v in data[x_col]],
            textposition='auto'
        ))
        fig.update_layout(
            title=title,
            xaxis_title='Sales ($)',
            yaxis_title=y_col.replace('_', ' ').title(),
            height=400, 
            yaxis={'categoryorder': 'total ascending'}
        )
        return fig
    
    create_tabbed_charts(datasets['seasonal'], create_seasonal_chart, categories, "2024 {} Sales by Season")

def calculate_cagr(start_value, end_value, num_years):
    
    if start_value <= 0 or end_value <= 0:
        return 0
    return (((end_value / start_value) ** (1 / num_years)) - 1) * 100

def format_market_share_data(df):
    
    df_display = df.copy()
    df_display['Our Revenue'] = df_display['Our_Revenue'].apply(lambda x: format_currency(x))
    df_display['Total Market Size'] = df_display['Total_Market_Size'].apply(lambda x: format_currency(x, 'B'))
    df_display['Market Share'] = df_display['Market_Share_Percentage'].apply(lambda x: f"{x:.1f}%")
    df_display['YoY Change'] = df_display['YoY_Revenue_Change_Percent'].apply(lambda x: f"{x:+.1f}%")
    
    return df_display[['Department', 'Our Revenue', 'Total Market Size', 'Market Share', 'YoY Change']], df

def apply_styling(df_formatted, df_original):
    
    def color_yoy_change(val):
        if '+' in str(val):
            return 'color: green'
        elif '-' in str(val):
            return 'color: red'
        return 'color: black'
    
    def highlight_max_values(s):
        
        if s.name == 'Our Revenue':
            max_revenue = df_original['Our_Revenue'].max()
            max_revenue_formatted = format_currency(max_revenue)
            return ['color: #66CDAA' if val == max_revenue_formatted else 'color: white' for val in s]
        elif s.name == 'Total Market Size':
            max_market_size = df_original['Total_Market_Size'].max()
            max_market_size_formatted = format_currency(max_market_size, 'B')
            return ['color: #66CDAA' if val == max_market_size_formatted else 'color: white' for val in s]
        elif s.name == 'Market Share':
            max_market_share_pct = df_original['Market_Share_Percentage'].max()
            max_market_share_formatted = f"{max_market_share_pct:.1f}%"
            return ['color: #66CDAA' if val == max_market_share_formatted else 'color: white' for val in s]
        return ['color: white'] * len(s)
    
    return (df_formatted.style
            
            .applymap(color_yoy_change, subset=['YoY Change'])
            .apply(highlight_max_values, subset=['Our Revenue', 'Total Market Size', 'Market Share']))

def market_share_analysis_section():
    st.subheader("Market Share Analysis")
    
    tab1, tab2 = st.tabs(["2024 Actual", "2025 Projections"])
    
    for tab, dataset_key in zip([tab1, tab2], ['market_share', 'market_share_projected']):
        with tab:
            df_formatted, df_original = format_market_share_data(datasets[dataset_key])
            styled_df = apply_styling(df_formatted, df_original)
            st.dataframe(styled_df, hide_index=True)
    

    st.markdown("---")
    st.subheader("Key Metrics")
    
    df_10_year = datasets['10_year']
    sales_columns = ['Software Sales', 'Hardware Sales', 'Advertising Sales']
    
    total_start = sum(df_10_year.iloc[0][col] for col in sales_columns)
    total_end = sum(df_10_year.iloc[-1][col] for col in sales_columns)
    total_cagr = calculate_cagr(total_start, total_end, 9)
    total_10_year_sales = sum(df_10_year[col].sum() for col in sales_columns)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Company CAGR (10yr)", f"{total_cagr:.1f}%", 
                 f"{format_currency(total_end - total_start)} growth")
    with col2:
        st.metric("Total Sales (10yr Period)", format_currency(total_10_year_sales, 'B'), 
                 "2015-2024 cumulative")

#dash layout
st.markdown("Analysis on Sales Data from 2014-2024 and Projections for 2025")

with st.expander("Full Dataset"):
    st.dataframe(
        df,
        column_config={
            col: st.column_config.NumberColumn(format="dollar", width=120)
            for col in months_select
        }
    )

col_budget, col_market = st.columns(2)
with col_budget:
    with st.container(border=True):
        budget_forecast_plot() #

with col_market:
    with st.container(border=True):
        market_share_analysis_section() #

col_map, col_cities = st.columns([0.65, 0.35])
with col_map:
    with st.container(border=True):
        pydeck_sales_map() ##

with col_cities:
    with st.container(border=True):
        top_cities_sales_plot() ##

col_trends, col_seasonal = st.columns([0.65, 0.35])
with col_trends:
    with st.container(border=True):
        yearly_per_account_plot() ###

with col_seasonal:
    with st.container(border=True):
        seasonal_sales_plot() ###