import seaborn as sns
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from model_train import train_all_models

@st.cache_resource #once per session
def load_models():
    return train_all_models()

match_model, meeting_model, relationship_model, mlb_classes, model_input_columns = load_models()

df = pd.read_csv('https://raw.githubusercontent.com/federicafiore/capstone/main/dating_app_behavior_dataset_extended1.csv')

#for visualization
match_dummies = pd.get_dummies(df['match_outcome'], prefix='match_outcome')

# Add the dummy columns
df = pd.concat([df, match_dummies], axis=1)

# Create the 'meeting' column as a combination of two 
df['meeting'] = df['match_outcome_Date Happened'] | df['match_outcome_Relationship Formed']

bins = [17, 24, 34, 44, 54, 64, 120]  # 120 upper bound age
labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

st.title('Dating App Outcome Insights')
mode = st.radio("Choose a feature to explore:", ["Dataset Deep Dive", "Outcome Predictor"])

if mode == "Dataset Deep Dive":
    
    st.subheader('Dating App Users')
    st.text('This section looks at users characteristics from the dataset - showing trends by gender, location, education, and age')
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plt.figure()
        sns.countplot(data=df, x='gender')
        plt.title('User Count by Gender')
        plt.tight_layout()
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        fig = plt.figure()
        sns.countplot(data=df, x='location_type')
        plt.title('User Count by Location')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plt.figure()
        sns.countplot(data=df, x='education_level')
        plt.title('User Count by Education Level')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig = plt.figure()
        sns.countplot(data=df, x='age_group', order=labels)
        plt.title('User Count by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig)
    
    st.subheader('Mutual Matches Deep Dive')
    st.text('This section looks at mutual matches from the dataset - showing trends by gender, location, education, and age')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plt.figure()
        sns.barplot(data=df, x='gender', y='mutual_matches')
        plt.title('Average Mutual Matches by Gender')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig = plt.figure()
        sns.barplot(data=df, x='location_type', y='mutual_matches')
        plt.title('Average Mutual Matches by Location')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plt.figure()
        sns.barplot(data=df, x='education_level', y='mutual_matches')
        plt.title('Average Mutual Matches by Education Level')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig = plt.figure()
        sns.barplot(data=df, x='age_group', y='mutual_matches')
        plt.title('Average Mutual Matches by Age Group')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    st.subheader('Dates Deep Dive')
    st.text('This section looks at dates reported from the dataset - showing trends by gender, location, education, and age')
    
    col1, col2 = st.columns(2)
    with col1:
        fig = plt.figure()
        sns.barplot(data=df, x='gender', y='meeting')
        plt.title('Average Dates by Gender')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig = plt.figure()
        sns.barplot(data=df, x='location_type', y='meeting')
        plt.title('Average Dates by Location')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = plt.figure()
        sns.barplot(data=df, x='education_level', y='meeting')
        plt.title('Average Dates by Education Level')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig = plt.figure()
        sns.barplot(data=df, x='age_group', y='meeting')
        plt.title('Average Date by Age Group')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    st.subheader('Relationships Formed Deep Dive')
    st.text('This section looks at relationships formed from the dataset - showing trends by gender, location, education, and age')
    
    col1, col2 = st.columns(2)
    with col1:
        fig = plt.figure()
        sns.barplot(data=df, x='gender', y='match_outcome_Relationship Formed')
        plt.title('Average Relationships Formed by Gender')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig = plt.figure()
        sns.barplot(data=df, x='location_type', y='match_outcome_Relationship Formed')
        plt.title('Average Relationships Formed by Location')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = plt.figure()
        sns.barplot(data=df, x='education_level', y='match_outcome_Relationship Formed')
        plt.title('Average Relationships Formed by Education Level')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig = plt.figure()
        sns.barplot(data=df, x='age_group', y='match_outcome_Relationship Formed')
        plt.title('Average Relationships Formed by Age Group')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

if mode == "Outcome Predictor":
    st.title('Dating App Outcome Predictor')
    st.text("This app helps predict one's chances of matching, meeting, and forming a relationship. Fill out the form below to get results.")
    
    # User input form
    with st.form("user_form"):
        gender = st.selectbox("Gender", ['Female','Male','Non-binary','Genderfluid','Transgender','Prefer Not to Say'])
        orientation = st.selectbox("Sexual Orientation", ['Asexual','Bisexual', 'Demisexual', 'Gay', 'Lesbian', 'Pansexual', 'Queer', 'Straight'])
        location = st.selectbox("Location", ['Metro', 'Remote Area', 'Rural', 'Small Town', 'Suburban', 'Urban'])
        zodiac = st.selectbox("Zodiac Sign", ['Aries', 'Taurus', 'Gemini','Cancer','Leo','Virgo','Libra','Scorpio','Sagittarius','Capricorn','Acquarius','Pisces'])
        education = st.selectbox("Education Level", ["High School","Diploma","Associate's","Bachelor's","Master's","MBA","PhD","Postdoc","No Formal Education"])
        interests = st.multiselect(
            "Interests",
            [
                "Anime", "Art", "Astrology", "Binge-Watching", "Board Games", "Cars",
                "Clubbing", "Coding", "Cooking", "Crafting", "Dancing", "DIY",
                "Fashion", "Fitness", "Foodie", "Gaming", "Gardening", "Hiking",
                "History", "Investing", "K-pop", "Languages", "Makeup", "Meditation",
                "Memes", "MMA", "Motorcycling", "Movies", "Music", "Painting",
                "Parenting", "Pets", "Photography", "Podcasts", "Poetry", "Politics",
                "Reading", "Reading Club", "Running", "Skating", "Sneaker Culture",
                "Social Activism", "Spirituality", "Stand-up Comedy", "Startups",
                "Tattoos", "Tech", "Traveling", "Writing", "Yoga"
            ]
        )
        swipe_time = st.selectbox("When does swiping happen?", ['Early Morning','Morning','Afternoon','Evening','Late Night','After Midnight'])
        body_type = st.selectbox("Body Type", ['Slim', 'Average', 'Curvy', 'Muscular', 'Athletic', 'Plus Size'])
        relationship_intent = st.selectbox("Dating App Used for:", ['Serious Relationship','Networking','Friends Only','Exploring','Casual Dating','Hookups'])
        age = st.number_input("Enter your age", min_value=0, max_value=100, value=30)
    
        unit = st.selectbox("Select weight unit", ["kg", "lb"])
        weight = st.number_input(f"Enter body weight ({unit})", min_value=0.0, max_value=1000.0)
    
        # Convert to kg
        if unit == "lb":
            weight_kg = weight * 0.453592
        else:
            weight_kg = weight
    
        height_unit = st.selectbox("Select height unit", ["cm", "in"])
        height = st.number_input(f"Enter height ({height_unit})", min_value=0.0, max_value=300.0)
    
        # Convert to cm
        if height_unit == "in":
            height_cm = height * 2.54
        else:
            height_cm = height
    
        app_usage_time_min = st.select_slider("How many minutes per day are being allocated into using this app?", options=range(1, 301), value=30)
        profile_pic_count = st.select_slider("How many profile pictures were added?", options=range(1,10),value=3)
    
        submit = st.form_submit_button("Predict Outcome")
    
    # Run prediction
    if submit:
        input_dictionary = {
            'age': [age],
            'weight': [weight_kg],
            'height': [height_cm],
            'app_usage_time_min': [app_usage_time_min]
            'profile_pic_count': [profile_pic_count]
        }
    
        df_input = pd.DataFrame(input_dictionary)
    
        # One-hot encode
        for col, val in {
            'gender': gender,
            'sexual_orientation': orientation,
            'location': location,
            'zodiac_sign': zodiac,
            'education_level': education,
            'swipe_time_of_day': swipe_time,
            'body_type': body_type,
            'relationship_intent': relationship_intent
    
        }.items():
            df_input[f'{col}_{val}'] = 1  # This will later be matched to training columns
    
        # Interest tags
        for tag in mlb_classes:  # mlb_classes = list of interest tag columns from training
            df_input[tag] = 1 if tag in interests else 0
    
        # Collect missing columns - so it matches expected df and model can work
        missing_cols = {col: 0 for col in model_input_columns if col not in df_input.columns}
    
        # Create a DataFrame with those columns
        missing_df = pd.DataFrame([missing_cols])
    
        # Concatenate all at once
        df_input = pd.concat([df_input, missing_df], axis=1)
    
        # Reorder columns
        df_input = df_input[model_input_columns]
    
        match_prob = match_model.predict_proba(df_input)[0][1]
        match_percent = round(match_prob * 100, 2)
        meeting_prob = meeting_model.predict_proba(df_input)[0][1]
        meeting_percent = round(meeting_prob * 100, 2)
        relationship_prob = relationship_model.predict_proba(df_input)[0][1]
        relationship_percent = round(relationship_prob * 100, 2)
    
        st.text(f"Chances to match with someone: {match_percent}%")
        st.text(f"Chances to go on a date: {meeting_percent}%")
        st.text(f"Chances to start a relationship: {relationship_percent}%")
        st.text(f"These results are not a guarantee and only probabilistic.")

