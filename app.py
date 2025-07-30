import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

st.set_page_config(layout='wide', page_title='Maven Bookshelf Challenge', page_icon='📚')

# Load data
@st.cache_data(show_spinner='⏳ Filling bookshelves...')
def load_data():
    # Goodreads works dataset
    works = pd.read_csv("goodreads_works.csv")
    works['description'] = works['description'].fillna('')
    works['genres'] = works['genres'].fillna('')
    works['author'] = works['author'].fillna('')

    reviews = pd.read_csv("reviews_new.csv")
    
    return works, reviews

works, reviews = load_data()

def recommend_books(works_filtered, vectorizer, book_vectors, user_input:str, n_recs=5, existing_list:list=[]):
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, book_vectors).flatten()
    top_indices = similarities.argsort()[-(n_recs+len(existing_list)):][::-1]
    
    recommendations = []
    for i in top_indices:
        recommendation_data_temp = works_filtered.iloc[i]

        if recommendation_data_temp['work_id'] not in recommendations:
            recommendations.append(recommendation_data_temp['work_id'])

        print(len(recommendations))
        if len(recommendations) >= n_recs:
            break
        
    return recommendations

@st.cache_data(show_spinner='⏳ Flipping pages...')
def get_recommendation_detail(works, work_id:int):
        recommendation_data = works[works['work_id'] == work_id].to_dict('records')[0]

        recommendation_detail = {
            'work_id': recommendation_data['work_id'],
            'original_title': recommendation_data['original_title'],
            'author': recommendation_data['author'],
            'original_publication_year': recommendation_data['original_publication_year'],
            'num_pages': recommendation_data['num_pages'],
            'avg_rating': recommendation_data['avg_rating'],
            'description': recommendation_data['description'][:300] + '...',
            'genres': recommendation_data['genres'],
            'image_url': recommendation_data['image_url'],
            'reviews_count': recommendation_data['reviews_count'],
            'ratings_count': recommendation_data['ratings_count']
        }
        
        return recommendation_detail

@st.cache_data(show_spinner='⏳ Flipping pages...')
def get_reading_list_detail(works, work_id:int):
        reading_list_data = works[works['work_id'] == work_id].to_dict('records')[0]

        reading_list_detail = {
            'work_id': reading_list_data['work_id'],
            'original_title': reading_list_data['original_title'],
            'author': reading_list_data['author'],
            'original_publication_year': reading_list_data['original_publication_year'],
            'num_pages': reading_list_data['num_pages'],
            'avg_rating': reading_list_data['avg_rating'],
            'description': reading_list_data['description'],
            'genres': reading_list_data['genres'],
            'image_url': reading_list_data['image_url'],
            'reviews_count': reading_list_data['reviews_count'],
            'ratings_count': reading_list_data['ratings_count'],
            '5_star_ratings': reading_list_data['5_star_ratings'],
            '4_star_ratings': reading_list_data['4_star_ratings'],
            '3_star_ratings': reading_list_data['3_star_ratings'],
            '2_star_ratings': reading_list_data['2_star_ratings'],
            '1_star_ratings': reading_list_data['1_star_ratings']
        }
        
        return reading_list_detail

# Session state initialization
if 'reading_list' not in st.session_state:
    st.session_state.reading_list = []

if 'preferences' not in st.session_state:
    st.session_state.preferences = {}

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

if 'works_filtered' not in st.session_state:
    st.session_state.works_filtered = works.copy()

if 'vectorizer' not in st.session_state: #or 'book_vectors' not in st.session_state:
    with st.spinner('⏳ Summoning bookworms...'):
        st.session_state.vectorizer = TfidfVectorizer(stop_words='english')
        st.session_state.vectorizer.fit(works['description'])

if 'navigation_step' not in st.session_state:
    st.session_state.navigation_step = None

page = st.sidebar.radio('Select section', ['📚 Book Recommender', '📖 My Reading List'], label_visibility='collapsed')

if page == '📚 Book Recommender':
    st.title('📚 Book Recommender')

    if not st.session_state.navigation_step:
        st.write('Use our recommendation system to find books based on your preferences...')
        
        if st.button("Let's go!", use_container_width=True):
            st.session_state.navigation_step = 'recommender_start'
            st.rerun()
    
        st.write('... or choose a curated list!')

        col1, col2 = st.columns(2)

        with col1:        
            if st.button('📢 Talk of the town', help='Books that have received much attention and well-loved by many.', use_container_width=True):
                st.session_state.recommendations += [16482835, 25126749, 28323940, 53077006, 321174]
                st.session_state.navigation_step = 'show_recommendations'
                st.rerun()

            if st.button('✨ Contemporary hits', help="You've definitely heard of these before, and for good reason - the most popular releases in recent years.", use_container_width=True):
                st.session_state.recommendations += [2792775, 4640799, 16827462, 17763198, 14302659]
                st.session_state.navigation_step = 'show_recommendations'
                st.rerun()

        with col2:
            if st.button('💎 Hidden gems', help="Less of the spotlight, but much love amongst those who did chance upon these books. Now's your turn!", use_container_width=True):
                st.session_state.recommendations += [53397776, 55475818, 49731957, 40893151, 49340945]
                st.session_state.navigation_step = 'show_recommendations'
                st.rerun()
            
            if st.button('🏛️ All-time classics', help='Renowed works that left their cultural mark on the world.', use_container_width=True):
                st.session_state.recommendations += [3462456, 992628, 2445338, 376514, 3237312]
                st.session_state.navigation_step = 'show_recommendations'
                st.rerun()   
        
    elif st.session_state.navigation_step == 'recommender_start':
        works_filtered = st.session_state.works_filtered
        
        st.write('Which age group suits you?')
        
        col1, col2, = st.columns(2)

        with col1:
            if st.button("👦👧 I'm a kid", use_container_width=True):
                st.session_state.works_filtered = works_filtered[(works_filtered['genres'].str.contains('children')) & (~works_filtered['genres'].str.contains('graphic'))]
                st.session_state.navigation_step = 'Q1_complete'
                st.rerun()

            if st.button('👨👩 I want something more mature', use_container_width=True):
                st.session_state.works_filtered = works_filtered[~works_filtered['genres'].str.contains('children')]
                st.session_state.navigation_step = 'Q1_complete'
                st.rerun()

        with col2:
            if st.button("🌱 I prefer young adult stories", use_container_width=True):
                st.session_state.works_filtered = works_filtered[works_filtered['genres'].str.contains('young-adult')]
                st.session_state.navigation_step = 'Q1_complete'
                st.rerun()

            if st.button("❔ I'm okay with anything!", use_container_width=True):
                st.session_state.navigation_step = 'Q1_complete'
                st.rerun()

    elif st.session_state.navigation_step == 'Q1_complete':
        works_filtered = st.session_state.works_filtered

        st.write('Do you prefer fiction or non-fiction?')
        
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button('🎭 Fiction', use_container_width=True):
                st.session_state.works_filtered = works_filtered[~works_filtered['genres'].str.contains('non-fiction')]
                st.session_state.navigation_step = 'Q2_complete'
                st.rerun()

        with col2:
            if st.button('📊 Non-fiction', use_container_width=True):
                st.session_state.works_filtered = works_filtered[works_filtered['genres'].str.contains('non-fiction')]
                st.session_state.navigation_step = 'Q2_complete'
                st.rerun()

        with col3:
            if st.button("❔ I'm okay with either!", use_container_width=True):
                st.session_state.navigation_step = 'Q2_complete'
                st.rerun()

    elif st.session_state.navigation_step == 'Q2_complete':
        works_filtered = st.session_state.works_filtered

        st.write('Which era of books do you want to explore?')
        
        col1, col2, = st.columns(2)

        with col1:
            if st.button('🕰️ Older literature', help='Books released before 1900', use_container_width=True):
                st.session_state.works_filtered = works_filtered[works_filtered['original_publication_year'] < 1900]
                st.session_state.navigation_step = 'text_input'
                st.rerun()

            if st.button('🕓 Modern releases', help='Books released between 2000-2009', use_container_width=True):
                st.session_state.works_filtered = works_filtered[(works_filtered['original_publication_year'] >= 2000) & (works_filtered['original_publication_year'] <= 2009)]
                st.session_state.navigation_step = 'text_input'
                st.rerun()

        with col2:
            if st.button('🕗 Classic reads', help='Books released between 1900-1999', use_container_width=True):
                st.session_state.works_filtered = works_filtered[(works_filtered['original_publication_year'] >= 1900) & (works_filtered['original_publication_year'] <= 1999)]
                st.session_state.navigation_step = 'text_input'
                st.rerun()

            if st.button('🕒 Most recent releases', help='Books released 2010 onwards', use_container_width=True):
                st.session_state.works_filtered = works_filtered[works_filtered['original_publication_year'] >= 2010]
                st.session_state.navigation_step = 'text_input'
                st.rerun()

        if st.button("❔ I'm okay with anything!", use_container_width=True):
            st.session_state.navigation_step = 'text_input'
            st.rerun()

    elif st.session_state.navigation_step == 'text_input':
        works_filtered = st.session_state.works_filtered
        # st.dataframe(works_filtered)
        
        st.write('Finally, tell us more about what you are most interested in!')
        
        text_input = st.text_input(
            'User input',
            placeholder="(Optional) Tell us briefly about what you want to read most (e.g. a heroic epic).",
            max_chars=100,
            label_visibility='collapsed'
        )

        if st.button("Recommend me some books!", use_container_width=True):
            if len(text_input) > 0:
                book_vectors = st.session_state.vectorizer.transform(works_filtered['description'])
                st.session_state.recommendations += recommend_books(
                    works_filtered,
                    st.session_state.vectorizer,
                    book_vectors,
                    text_input,
                    n_recs=5,
                    existing_list=st.session_state.reading_list
                )
            
            else:
                st.session_state.recommendations += list(np.random.choice(works_filtered['work_id'], 5, replace=False))

            st.session_state.navigation_step = 'show_recommendations'
            st.rerun()

    if st.session_state.navigation_step == 'show_recommendations':
        st.write('Here are your recommendations!')
        # st.write(st.session_state.recommendations)
        st.markdown('---')
        
        for recommendation in st.session_state.recommendations:
            col1, col2, col3 = st.columns([1, 6, 3])
            
            recommendation_detail = get_recommendation_detail(works, work_id=recommendation)

            with col1:
                st.image(recommendation_detail['image_url'], use_column_width=True)
        
            with col2:
                st.markdown(f"**{recommendation_detail['original_title']}** ")
                st.markdown(f"*{recommendation_detail['author']}* ({int(recommendation_detail['original_publication_year'])})")

                if recommendation_detail['num_pages'] > 0:
                    st.markdown(f"{int(recommendation_detail['num_pages']):,} pages ⋅ *{recommendation_detail['genres']}*")
                else:
                    st.markdown(f"*{recommendation_detail['genres']}*")
                
                st.markdown(f"{recommendation_detail['avg_rating']:1}⭐ *({recommendation_detail['ratings_count']:,} ratings)* ⋅ *{recommendation_detail['reviews_count']:,} user reviews*")
                    
            with col3:
                if recommendation not in st.session_state.reading_list:
                    if st.button(f'➕ Add to Reading List', key=f'add_recommendation_{recommendation}', use_container_width=True):
                        st.session_state.reading_list.append(recommendation)
                        st.rerun()
                
                else:
                    if st.button(f'❌ Remove from Reading List', key=f'remove_recommendation_{recommendation}', use_container_width=True):
                        st.session_state.reading_list.remove(recommendation)
                        st.rerun()

            # st.markdown(f"> {recommendation_detail['description']}")
            st.markdown('---')
        
        if st.button('🔁 Restart Book Recommender', use_container_width=True):
            st.session_state.navigation_step = None
            st.session_state.works_filtered = works.copy()
            st.session_state.recommendations = []
            st.rerun()

if page == '📖 My Reading List':
    st.title('📖 My Reading List')

    if len(st.session_state.reading_list) == 0:
        st.warning('Your reading list is empty. Head over to ***Book Recommender*** to get started!')
    
    else:       
        reading_list_details = []
        for book in st.session_state.reading_list:
            reading_list_detail = get_reading_list_detail(works, work_id=book)
            reading_list_details.append(reading_list_detail)

        book_titles = [book['original_title'] for book in reading_list_details]
        tabs = st.tabs(book_titles)

        for i, tab in enumerate(tabs):
            reading_list_detail = reading_list_details[i]

            with tab:
                col1, col2, col3 = st.columns([8, 1, 33])
            
                with col1:
                    st.image(reading_list_detail['image_url'], use_column_width=True)
                    st.markdown(f"First published {int(reading_list_detail['original_publication_year'])}")

                    if reading_list_detail['num_pages'] > 0:
                        st.markdown(f"{int(reading_list_detail['num_pages']):,} pages")
            
                with col3:
                    col4, col5 = st.columns([3, 1])

                    with col4:
                        st.markdown(f"#### {reading_list_detail['original_title']}")
                        st.markdown(f"*{reading_list_detail['author']}*")
                        
                        st.markdown(f"{reading_list_detail['description']}", unsafe_allow_html=True)
                        st.markdown(' '.join([f":blue-background[*{genre}*]" for genre in reading_list_detail['genres'].split(', ')]))

                    with col5:
                        st.markdown(f"⭐ {reading_list_detail['avg_rating']:.1f} ({reading_list_detail['ratings_count']:,} ratings)")
                        st.markdown(f"💬 {reading_list_detail['reviews_count']:,} reviews")

                with st.expander('View reviews', expanded=False):
                    col6, col7 = st.columns(2)
                    
                    with col6:
                        review_counts = {
                            '1⭐': reading_list_detail['1_star_ratings'],
                            '2⭐': reading_list_detail['2_star_ratings'],
                            '3⭐': reading_list_detail['3_star_ratings'],
                            '4⭐': reading_list_detail['4_star_ratings'],
                            '5⭐': reading_list_detail['5_star_ratings']
                            }

                        fig = go.Figure(
                            data=[
                                go.Bar(
                                    x=list(review_counts.values()),
                                    y=list(review_counts.keys()),
                                    orientation='h',
                                    marker_color='#e9e5cd',
                                    text=[f'{value:,}' for value in review_counts.values()],
                                    textposition='inside',
                                    insidetextanchor='start',
                                    hovertemplate='%{x:,} users gave %{y} ratings',
                                    name=''
                                )
                            ]
                        )

                        fig.update_layout(
                            margin=dict(t=10, b=10, l=10, r=30),
                            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                            template='plotly_white',
                            height=300
                        )

                        st.markdown('Ratings breakdown')
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    
                    with col7:
                        review_choice = st.radio('Review type', ['Top reviews', 'Most recent reviews'], key=f"review_type{reading_list_detail['work_id']}", label_visibility='collapsed', horizontal=True)

                        if review_choice == 'Top reviews':
                            reviews_display = reviews[(reviews['work_id'] == reading_list_detail['work_id']) & (reviews['category'] == 'top')]
                        elif review_choice == 'Most recent reviews':
                            reviews_display = reviews[(reviews['work_id'] == reading_list_detail['work_id']) & (reviews['category'] == 'recent')]
                        reviews_display = reviews_display.to_dict('records')
                        
                        for review in reviews_display:
                            if review['rating'] > 0:
                                st.markdown(f"**{pd.to_datetime(review['date_added']).strftime('%d %B %Y')}** ⋅ {review['rating']:.1f}⭐")
                            else:
                                st.markdown(f"**{pd.to_datetime(review['date_added']).strftime('%d %B %Y')}**")

                            st.markdown(f"> {review['review_text']}")

                            if isinstance(review['started_at'], str) and isinstance(review['read_at'], str):
                                st.markdown(f"Started reading on *{pd.to_datetime(review['started_at']).strftime('%d %B %Y')}* and finished on *{pd.to_datetime(review['read_at']).strftime('%d %B %Y')}*")
                            elif isinstance(review['started_at'], str):
                                st.markdown(f"Started reading on *{pd.to_datetime(review['started_at']).strftime('%d %B %Y')}*")
                            elif isinstance(review['read_at'], str):
                                st.markdown(f"Finished reading on *{pd.to_datetime(review['read_at']).strftime('%d %B %Y')}*")

                            st.markdown('---')

                st.markdown('---')

                col8, col9 = st.columns(2)

                with col8:
                    url = f"https://www.amazon.com/s?k={reading_list_detail['original_title'].replace(' ', '+')}&ref=cs_503_search"
                    st.link_button('Find on Amazon', url, type='primary', use_container_width=True)

                with col9:
                    if st.button(f'❌ Remove from Reading List', key=f"remove_book_{reading_list_detail['work_id']}", use_container_width=True):
                        st.session_state.reading_list.remove(reading_list_detail['work_id'])
                        st.rerun()
