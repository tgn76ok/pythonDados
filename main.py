# -*- coding: utf-8 -*-
"""
Dashboard de Análise do Catálogo da Netflix com Streamlit

Este script cria uma aplicação web interativa para visualizar os insights do dataset da Netflix.

Instruções de Uso:
1. Garanta que tem o Python instalado.
2. Instale as bibliotecas necessárias:
   pip install streamlit pandas plotly wordcloud pygwalker
3. Faça o download do dataset 'netflix_titles.csv' e coloque-o na mesma pasta deste script.
4. Execute a aplicação a partir do seu terminal:
   streamlit run netflix_analysis_dashboard.py
5. Uma aba do navegador será aberta com o dashboard interativo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pygwalker as pyw

# --- Secção 0: Configuração da Página e Carregamento de Dados ---

# Configura o layout da página para ser mais largo
st.set_page_config(layout="wide")

st.title('Dashboard de Análise do Catálogo da Netflix')

# Paleta de cores da marca Netflix para consistência visual
netflix_colors = {
    "red": "#E50914",
    "black": "#221f1f",
    "grey": "#808080"
}

# Usar cache para carregar os dados apenas uma vez, melhorando o desempenho
@st.cache_data
def load_and_clean_data():
    """Carrega, limpa e prepara o dataset da Netflix."""
    try:
        df = pd.read_csv('netflix_titles.csv')
    except FileNotFoundError:
        st.error("Erro: O ficheiro 'netflix_titles.csv' não foi encontrado. Por favor, coloque-o na mesma pasta.")
        return None

    # Limpeza básica
    df.drop_duplicates(subset=['title', 'type'], inplace=True)
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    # MODIFICADO: Alterada a imputação da moda para 'Desconhecido' para maior precisão
    df['country'] = df['country'].fillna('Unknown')
    df.dropna(subset=['date_added', 'rating', 'duration'], inplace=True)
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df.dropna(subset=['date_added'], inplace=True)
    df['year_added'] = df['date_added'].dt.year

    # Limpeza avançada
    df = df[df['year_added'] >= df['release_year']].copy()
    
    def group_rating(rating):
        if rating in ['TV-MA', 'R', 'NC-17', 'UR']: return 'Adulto'
        elif rating in ['TV-14', 'PG-13']: return 'Adolescente'
        else: return 'Família/Crianças'
    df['rating_group'] = df['rating'].apply(group_rating)

    # Engenharia de Funcionalidades
    df['month_added'] = df['date_added'].dt.month
    df['content_lag_years'] = df['year_added'] - df['release_year']
    
    movie_mask = df['type'] == 'Movie'
    tv_show_mask = df['type'] == 'TV Show'
    
    df.loc[movie_mask, 'duration_minutes'] = df.loc[movie_mask, 'duration'].astype(str).str.replace(' min', '').astype(float)
    df.loc[tv_show_mask, 'duration_seasons'] = df.loc[tv_show_mask, 'duration'].astype(str).str.replace(r' Seasons?| Season', '', regex=True).astype(float)
    
    return df

# Carrega os dados
df_original = load_and_clean_data()

if df_original is not None:
    # --- Secção 1: Barra Lateral com Filtros ---
    st.sidebar.header('Filtros do Dashboard')

    # Filtro para tipo de conteúdo
    selected_type = st.sidebar.multiselect(
        'Selecione o Tipo de Conteúdo:',
        options=df_original['type'].unique(),
        default=df_original['type'].unique()
    )
    
    # Filtro para classificação agrupada
    selected_rating = st.sidebar.multiselect(
        'Selecione a Classificação:',
        options=df_original['rating_group'].unique(),
        default=df_original['rating_group'].unique()
    )

    # Filtro para país
    unique_countries = sorted(list(set(', '.join(df_original['country'].dropna()).split(', '))))
    selected_country = st.sidebar.multiselect(
        'Selecione o País de Produção:',
        options=unique_countries,
        default=['United States']
    )
    
    # Filtro para género (para a nuvem de palavras)
    unique_genres = sorted(list(set(', '.join(df_original['listed_in'].dropna()).split(', '))))
    selected_genre = st.sidebar.selectbox(
        'Selecione um Género para a Nuvem de Palavras:',
        options=['Todos'] + unique_genres,
        index=0
    )

    # Filtro para ano de lançamento
    min_year, max_year = int(df_original['release_year'].min()), int(df_original['release_year'].max())
    selected_year = st.sidebar.slider(
        'Selecione o Intervalo do Ano de Lançamento:',
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    # Aplicar filtros ao dataframe
    df = df_original[
        (df_original['type'].isin(selected_type)) &
        (df_original['rating_group'].isin(selected_rating)) &
        (df_original['release_year'] >= selected_year[0]) &
        (df_original['release_year'] <= selected_year[1]) &
        (df_original['country'].str.contains('|'.join(selected_country)))
    ]

    # --- Secção 2: Visualizações do Dashboard ---
    
    st.header('Visão Geral do Catálogo')

    # Métricas Principais
    total_titles = len(df)
    movie_count = len(df[df['type'] == 'Movie'])
    tv_show_count = len(df[df['type'] == 'TV Show'])

    metric1, metric2, metric3 = st.columns(3)
    with metric1:
        st.metric(label="Total de Títulos", value=f"{total_titles:,}")
    with metric2:
        st.metric(label="Filmes", value=f"{movie_count:,}")
    with metric3:
        st.metric(label="Séries de TV", value=f"{tv_show_count:,}")


    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Filmes vs. Séries de TV')
        type_counts = df['type'].value_counts()
        fig1 = px.pie(
            values=type_counts.values, 
            names=type_counts.index, 
            title='Distribuição de Conteúdo',
            color_discrete_map={'Movie': netflix_colors['red'], 'TV Show': netflix_colors['black']}
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader('Público-Alvo Principal')
        rating_counts = df['rating_group'].value_counts()
        fig2 = px.bar(
            x=rating_counts.index, 
            y=rating_counts.values,
            labels={'x': 'Segmento de Público', 'y': 'Quantidade de Títulos'},
            title='Classificações Agrupadas',
            color_discrete_sequence=[netflix_colors['red']]
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.header('Evolução e Tendências')

    st.subheader('Crescimento do Catálogo ao Longo dos Anos')
    content_by_year = df.groupby('year_added')['type'].value_counts().unstack().fillna(0).reset_index()
    
    # CORREÇÃO: Obter dinamicamente as colunas a plotar com base no que está disponível nos dados filtrados
    columns_to_plot = [col for col in content_by_year.columns if col != 'year_added']
    
    fig3 = px.line(
        content_by_year, 
        x='year_added', 
        y=columns_to_plot, # Usar a lista dinâmica de colunas aqui
        labels={'year_added': 'Ano de Adição', 'value': 'Quantidade de Títulos', 'variable': 'Tipo de Conteúdo'},
        title='Conteúdo Adicionado à Netflix por Ano',
        markers=True,
        color_discrete_map={'Movie': netflix_colors['red'], 'TV Show': netflix_colors['black']}
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.header('Análise Geográfica e de Géneros')
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader('Top 10 Países Produtores')
        countries_exploded = df[df['country'] != 'Unknown']['country'].str.split(', ').explode()
        top_10_countries = countries_exploded.value_counts().head(10)
        fig5 = px.bar(
            y=top_10_countries.index, 
            x=top_10_countries.values,
            orientation='h',
            labels={'y': 'País', 'x': 'Quantidade de Títulos'},
            title='Top 10 Países por Produção de Conteúdo',
            color_discrete_sequence=px.colors.sequential.Reds_r
        )
        fig5.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig5, use_container_width=True)

    with col4:
        st.subheader('Top 10 Géneros Mais Comuns')
        genres_exploded = df['listed_in'].str.split(', ').explode()
        top_10_genres = genres_exploded.value_counts().head(10)
        fig6 = px.bar(
            y=top_10_genres.index, 
            x=top_10_genres.values,
            orientation='h',
            labels={'y': 'Género', 'x': 'Quantidade de Títulos'},
            title='Top 10 Géneros Mais Comuns',
            color_discrete_sequence=px.colors.sequential.Reds_r
        )
        fig6.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig6, use_container_width=True)

    # --- Secção 3: Análise da Duração do Conteúdo ---
    st.header('Análise da Duração do Conteúdo')
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader('Distribuição da Duração dos Filmes (Minutos)')
        df_movies = df[df['type'] == 'Movie'].dropna(subset=['duration_minutes'])
        fig7 = px.histogram(
            df_movies,
            x='duration_minutes',
            nbins=40,
            title='Histograma da Duração dos Filmes',
            labels={'duration_minutes': 'Duração (Minutos)', 'count': 'Quantidade de Filmes'},
            color_discrete_sequence=[netflix_colors['red']]
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col6:
        st.subheader('Contagem de Séries por Número de Temporadas')
        df_shows = df[df['type'] == 'TV Show'].dropna(subset=['duration_seasons'])
        
        # CORREÇÃO: Adicionada uma verificação de dados vazios para evitar erros
        if not df_shows.empty:
            seasons_count_df = df_shows['duration_seasons'].value_counts().sort_index().reset_index()
            seasons_count_df.columns = ['duration_seasons', 'count']

            fig8 = px.bar(
                seasons_count_df,
                x='duration_seasons',
                y='count',
                title='Contagem de Séries por Temporadas',
                labels={'duration_seasons': 'Número de Temporadas', 'count': 'Quantidade de Séries'},
                color_discrete_sequence=[netflix_colors['black']]
            )
            fig8.update_xaxes(type='category') # Para tratar as temporadas como categorias
            st.plotly_chart(fig8, use_container_width=True)
        else:
            st.info("Não há dados de Séries de TV disponíveis para os filtros selecionados para exibir a contagem de temporadas.")
        
    # --- Secção 4: Análise Temática das Descrições (Nuvem de Palavras) ---
    st.header('Análise Temática das Descrições')

    df_wordcloud = df.copy()
    if selected_genre != 'Todos':
        df_wordcloud = df[df['listed_in'].str.contains(selected_genre)]
    
    text = " ".join(review for review in df_wordcloud.description)
    if text:
        st.subheader(f'Nuvem de Palavras para o Género: {selected_genre}')
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(stopwords=stopwords, background_color="white", colormap='Reds', width=800, height=400).generate(text)
        
        fig9, ax9 = plt.subplots(figsize=(10, 5))
        ax9.imshow(wordcloud, interpolation='bilinear')
        ax9.axis("off")
        st.pyplot(fig9)
    else:
        st.warning(f"Não há descrições disponíveis para o género '{selected_genre}' com os filtros atuais.")


    # --- Secção 5: Visualização dos Dados Limpos ---
    st.sidebar.header('Amostra de Dados')
    
    # Slider para selecionar o número de linhas a exibir
    if not df.empty:
        num_rows = st.sidebar.slider(
            'Número de linhas para exibir:',
            min_value=5,
            max_value=len(df),
            value=min(10, len(df)), # Garante que o valor não seja maior que os dados
            step=5
        )
        
        st.header('Explorar Dados Filtrados')
        st.write(f"A exibir uma amostra aleatória de {num_rows} registos com base nos filtros aplicados.")
        st.dataframe(df.sample(num_rows))
    else:
        st.warning("Nenhum dado corresponde aos filtros selecionados. Por favor, ajuste os filtros.")

