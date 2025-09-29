# -*- coding: utf-8 -*-
"""
Netflix Catalog Analysis Dashboard with Streamlit

This script creates an interactive web application to visualize insights from the Netflix dataset.

Usage Instructions:
1. Ensure you have Python installed.
2. Install the required libraries:
   pip install streamlit pandas plotly wordcloud pygwalker
3. Download the 'netflix_titles.csv' dataset and place it in the same folder as this script.
4. Run the application from your terminal:
   streamlit run netflix_analysis_dashboard.py
5. A browser tab will open with the interactive dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pygwalker as pyw

# --- Section 0: Page Configuration and Data Loading ---

# Configure the page layout to be wider
st.set_page_config(layout="wide")

st.title('Netflix Catalog Analysis Dashboard')

# Netflix brand color palette for visual consistency
netflix_colors = {
    "red": "#E50914",
    "black": "#221f1f",
    "grey": "#808080"
}

# Use cache to load data only once, improving performance
@st.cache_data
def load_and_clean_data():
    """Loads, cleans, and prepares the Netflix dataset."""
    try:
        df = pd.read_csv('netflix_titles.csv')
    except FileNotFoundError:
        st.error("Error: The file 'netflix_titles.csv' was not found. Please place it in the same folder.")
        return None

    # Basic cleaning
    df.drop_duplicates(subset=['title', 'type'], inplace=True)
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    # MODIFIED: Changed imputation from mode to 'Unknown' for better accuracy
    df['country'] = df['country'].fillna('Unknown')
    df.dropna(subset=['date_added', 'rating', 'duration'], inplace=True)
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df.dropna(subset=['date_added'], inplace=True)
    df['year_added'] = df['date_added'].dt.year

    # Advanced cleaning
    df = df[df['year_added'] >= df['release_year']].copy()
    
    def group_rating(rating):
        if rating in ['TV-MA', 'R', 'NC-17', 'UR']: return 'Adult'
        elif rating in ['TV-14', 'PG-13']: return 'Teen'
        else: return 'Family/Kids'
    df['rating_group'] = df['rating'].apply(group_rating)

    # Feature Engineering
    df['month_added'] = df['date_added'].dt.month
    df['content_lag_years'] = df['year_added'] - df['release_year']
    
    movie_mask = df['type'] == 'Movie'
    tv_show_mask = df['type'] == 'TV Show'
    
    df.loc[movie_mask, 'duration_minutes'] = df.loc[movie_mask, 'duration'].astype(str).str.replace(' min', '').astype(float)
    df.loc[tv_show_mask, 'duration_seasons'] = df.loc[tv_show_mask, 'duration'].astype(str).str.replace(r' Seasons?| Season', '', regex=True).astype(float)
    
    return df

# Load the data
df_original = load_and_clean_data()

if df_original is not None:
    # --- Section 1: Sidebar with Filters ---
    st.sidebar.header('Dashboard Filters')

    # Filter for content type
    selected_type = st.sidebar.multiselect(
        'Select Content Type:',
        options=df_original['type'].unique(),
        default=df_original['type'].unique()
    )
    
    # Filter for rating group
    selected_rating = st.sidebar.multiselect(
        'Select Rating:',
        options=df_original['rating_group'].unique(),
        default=df_original['rating_group'].unique()
    )

    # Filter for country
    unique_countries = sorted(list(set(', '.join(df_original['country'].dropna()).split(', '))))
    selected_country = st.sidebar.multiselect(
        'Select Production Country:',
        options=unique_countries,
        default=['United States']
    )
    
    # Filter for genre (for the word cloud)
    unique_genres = sorted(list(set(', '.join(df_original['listed_in'].dropna()).split(', '))))
    selected_genre = st.sidebar.selectbox(
        'Select a Genre for the Word Cloud:',
        options=['All'] + unique_genres,
        index=0
    )

    # Filter for release year
    min_year, max_year = int(df_original['release_year'].min()), int(df_original['release_year'].max())
    selected_year = st.sidebar.slider(
        'Select Release Year Range:',
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    # Apply filters to the dataframe
    df = df_original[
        (df_original['type'].isin(selected_type)) &
        (df_original['rating_group'].isin(selected_rating)) &
        (df_original['release_year'] >= selected_year[0]) &
        (df_original['release_year'] <= selected_year[1]) &
        (df_original['country'].str.contains('|'.join(selected_country)))
    ]

    # --- Section 2: Dashboard Visualizations ---
    
    st.header('Catalog Overview')

    # Key Metrics
    total_titles = len(df)
    movie_count = len(df[df['type'] == 'Movie'])
    tv_show_count = len(df[df['type'] == 'TV Show'])

    metric1, metric2, metric3 = st.columns(3)
    with metric1:
        st.metric(label="Total Titles", value=f"{total_titles:,}")
    with metric2:
        st.metric(label="Movies", value=f"{movie_count:,}")
    with metric3:
        st.metric(label="TV Shows", value=f"{tv_show_count:,}")


    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Movies vs. TV Shows')
        type_counts = df['type'].value_counts()
        fig1 = px.pie(
            values=type_counts.values, 
            names=type_counts.index, 
            title='Content Distribution',
            color_discrete_map={'Movie': netflix_colors['red'], 'TV Show': netflix_colors['black']}
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader('Primary Target Audience')
        rating_counts = df['rating_group'].value_counts()
        fig2 = px.bar(
            x=rating_counts.index, 
            y=rating_counts.values,
            labels={'x': 'Audience Segment', 'y': 'Number of Titles'},
            title='Grouped Ratings',
            color_discrete_sequence=[netflix_colors['red']]
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.header('Evolution and Trends')

    st.subheader('Catalog Growth Over the Years')
    content_by_year = df.groupby('year_added')['type'].value_counts().unstack().fillna(0).reset_index()
    
    # FIX: Dynamically get the columns to plot based on what's available in the filtered data
    columns_to_plot = [col for col in content_by_year.columns if col != 'year_added']
    
    fig3 = px.line(
        content_by_year, 
        x='year_added', 
        y=columns_to_plot, # Use the dynamic list of columns here
        labels={'year_added': 'Year Added', 'value': 'Number of Titles', 'variable': 'Content Type'},
        title='Content Added to Netflix by Year',
        markers=True,
        color_discrete_map={'Movie': netflix_colors['red'], 'TV Show': netflix_colors['black']}
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.header('Geographic and Genre Analysis')
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader('Top 10 Producing Countries')
        countries_exploded = df[df['country'] != 'Unknown']['country'].str.split(', ').explode()
        top_10_countries = countries_exploded.value_counts().head(10)
        fig5 = px.bar(
            y=top_10_countries.index, 
            x=top_10_countries.values,
            orientation='h',
            labels={'y': 'Country', 'x': 'Number of Titles'},
            title='Top 10 Countries by Content Production',
            color_discrete_sequence=px.colors.sequential.Reds_r
        )
        fig5.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig5, use_container_width=True)

    with col4:
        st.subheader('Top 10 Most Common Genres')
        genres_exploded = df['listed_in'].str.split(', ').explode()
        top_10_genres = genres_exploded.value_counts().head(10)
        fig6 = px.bar(
            y=top_10_genres.index, 
            x=top_10_genres.values,
            orientation='h',
            labels={'y': 'Genre', 'x': 'Number of Titles'},
            title='Top 10 Most Common Genres',
            color_discrete_sequence=px.colors.sequential.Reds_r
        )
        fig6.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig6, use_container_width=True)

    # --- Section 3: Content Duration Analysis ---
    st.header('Content Duration Analysis')
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader('Movie Duration Distribution (Minutes)')
        df_movies = df[df['type'] == 'Movie'].dropna(subset=['duration_minutes'])
        fig7 = px.histogram(
            df_movies,
            x='duration_minutes',
            nbins=40,
            title='Movie Duration Histogram',
            labels={'duration_minutes': 'Duration (Minutes)', 'count': 'Number of Movies'},
            color_discrete_sequence=[netflix_colors['red']]
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col6:
        st.subheader('Series Count by Number of Seasons')
        df_shows = df[df['type'] == 'TV Show'].dropna(subset=['duration_seasons'])
        
        # FIX: Added a check for empty data to prevent errors
        if not df_shows.empty:
            seasons_count_df = df_shows['duration_seasons'].value_counts().sort_index().reset_index()
            seasons_count_df.columns = ['duration_seasons', 'count']

            fig8 = px.bar(
                seasons_count_df,
                x='duration_seasons',
                y='count',
                title='Series Count by Seasons',
                labels={'duration_seasons': 'Number of Seasons', 'count': 'Number of Series'},
                color_discrete_sequence=[netflix_colors['black']]
            )
            fig8.update_xaxes(type='category') # To treat seasons as categories
            st.plotly_chart(fig8, use_container_width=True)
        else:
            st.info("No TV Show data available for the selected filters to display season counts.")
        
    # --- Section 4: Thematic Analysis of Descriptions (Word Cloud) ---
    st.header('Thematic Analysis of Descriptions')

    df_wordcloud = df.copy()
    if selected_genre != 'All':
        df_wordcloud = df[df['listed_in'].str.contains(selected_genre)]
    
    text = " ".join(review for review in df_wordcloud.description)
    if text:
        st.subheader(f'Word Cloud for Genre: {selected_genre}')
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(stopwords=stopwords, background_color="white", colormap='Reds', width=800, height=400).generate(text)
        
        fig9, ax9 = plt.subplots(figsize=(10, 5))
        ax9.imshow(wordcloud, interpolation='bilinear')
        ax9.axis("off")
        st.pyplot(fig9)
    else:
        st.warning(f"No descriptions available for the genre '{selected_genre}' with the current filters.")


    # --- Section 5: Clean Data Visualization ---
    st.sidebar.header('Data Sample')
    
    # Slider to select the number of rows to display
    if not df.empty:
        num_rows = st.sidebar.slider(
            'Number of rows to display:',
            min_value=5,
            max_value=len(df),
            value=min(10, len(df)), # Ensures the value is not greater than the data
            step=5
        )
        
        st.header('Explore Filtered Data')
        st.write(f"Showing a random sample of {num_rows} records based on the applied filters.")
        st.dataframe(df.sample(num_rows))
    else:
        st.warning("No data matches the selected filters. Please adjust your filters.")

