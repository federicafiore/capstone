import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_all_models():
    #load CSV dataset
    df = pd.read_csv('https://raw.githubusercontent.com/federicafiore/capstone/main/dating_app_behavior_dataset_extended1.csv')
    
    #make mutual matches 1 if it happened, 0 if it didn't (any number greater than 0 would be 1)
    df['mutual_matches'] = df['mutual_matches'].apply(lambda x: 1 if x > 0 else 0)
    
    #columns for dummy
    category_columns = [
        'gender', 'sexual_orientation', 'location_type',
        'education_level', 'swipe_time_of_day',
        'zodiac_sign', 'body_type', 'relationship_intent', 'match_outcome'
    ]
    
    df = pd.get_dummies(df, columns=category_columns)
    
    # Process interest_tags, separate them as there are three in one cell divided by comma
    df['interest_tags'] = df['interest_tags'].apply(lambda x: [tag.strip() for tag in x.split(',')])
    
    mlb = MultiLabelBinarizer()
    interest_dummies = pd.DataFrame(mlb.fit_transform(df['interest_tags']), columns=mlb.classes_)
    
    # Combine with df
    df = pd.concat([df, interest_dummies], axis=1)
    df.drop('interest_tags', axis=1, inplace=True)
    
    #date happened whether date happened or relationship formed as outcome - assumption
    df['match_outcome_Date Happened'] = df['match_outcome_Date Happened'] | df['match_outcome_Relationship Formed']
    
    # Choose X and Y
    X = df.drop(['mutual_matches','income_bracket','app_usage_time_label','swipe_right_ratio','swipe_right_label','likes_received','bio_length','message_sent_count',	'emoji_usage_rate',	'last_active_hour', 'match_outcome_Blocked', 'match_outcome_Catfished',
        'match_outcome_Chat Ignored', 'match_outcome_Ghosted', 'match_outcome_Instant Match', 'match_outcome_Mutual Match','match_outcome_No Action', 'match_outcome_One-sided Like', 'match_outcome_Date Happened', 'match_outcome_Relationship Formed'], axis=1)
    #outcome for mutual match model
    y_match = df['mutual_matches']
    #outcome for date model
    y_meeting = df['match_outcome_Date Happened']
    #outcome for relationship model
    y_relationship = df['match_outcome_Relationship Formed']
    
    X_train, X_test, y_match_train, y_match_test = train_test_split(X, y_match, test_size=0.2, random_state=42)
    _, _, y_meeting_train, y_meeting_test = train_test_split(X, y_meeting, test_size=0.2, random_state=42)
    _, _, y_relationship_train, y_relationship_test = train_test_split(X, y_relationship, test_size=0.2, random_state=42)
    
    # Train all 3 models
    match_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    match_model.fit(X_train, y_match_train)
    
    meeting_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    meeting_model.fit(X_train, y_meeting_train)
    
    relationship_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    relationship_model.fit(X_train, y_relationship_train)
    
    # 5-fold cross-validation for match prediction
    cv_scores_match = cross_val_score(match_model, X, y_match, cv=5, scoring='accuracy')
    st.write(f"Match Model - 5-Fold CV Accuracy: {cv_scores_match.mean():.2f} (+/- {cv_scores_match.std():.2f})")
    
    # 5-fold cross-validation for meeting prediction
    cv_scores_meeting = cross_val_score(meeting_model, X, y_meeting, cv=5, scoring='accuracy')
    st.write(f"Meeting Model - 5-Fold CV Accuracy: {cv_scores_meeting.mean():.2f} (+/- {cv_scores_meeting.std():.2f})")
    
    # 5-fold cross-validation for relationship prediction
    cv_scores_relationship = cross_val_score(relationship_model, X, y_relationship, cv=5, scoring='accuracy')
    st.write(f"Relationship Model - 5-Fold CV Accuracy: {cv_scores_relationship.mean():.2f} (+/- {cv_scores_relationship.std():.2f})")
    
    y_match_pred = match_model.predict(X_test)
    y_match_prob = match_model.predict_proba(X_test)[:, 1]
    
    match_accuracy = accuracy_score(y_match_test, y_match_pred)
    match_precision = precision_score(y_match_test, y_match_pred)
    match_recall = recall_score(y_match_test, y_match_pred)
    match_auc = roc_auc_score(y_match_test, y_match_prob)
    
    st.write("Match Model Metrics:")
    st.write(f"Accuracy: {match_accuracy:.2f}")
    st.write(f"Precision: {match_precision:.2f}")
    st.write(f"Recall: {match_recall:.2f}")
    st.write(f"AUC: {match_auc:.2f}")
    
    y_meeting_pred = meeting_model.predict(X_test)
    y_meeting_prob = meeting_model.predict_proba(X_test)[:, 1]
    
    meeting_accuracy = accuracy_score(y_meeting_test, y_meeting_pred)
    meeting_precision = precision_score(y_meeting_test, y_meeting_pred)
    meeting_recall = recall_score(y_meeting_test, y_meeting_pred)
    meeting_auc = roc_auc_score(y_meeting_test, y_meeting_prob)
    
    st.write("Meeting Model Metrics:")
    st.write(f"Accuracy: {meeting_accuracy:.2f}")
    st.write(f"Precision: {meeting_precision:.2f}")
    st.write(f"Recall: {meeting_recall:.2f}")
    st.write(f"AUC: {meeting_auc:.2f}")
    
    y_relationship_pred = relationship_model.predict(X_test)
    y_relationship_prob = relationship_model.predict_proba(X_test)[:, 1]
    
    relationship_accuracy = accuracy_score(y_relationship_test, y_relationship_pred)
    relationship_precision = precision_score(y_relationship_test, y_relationship_pred)
    relationship_recall = recall_score(y_relationship_test, y_relationship_pred)
    relationship_auc = roc_auc_score(y_relationship_test, y_relationship_prob)
    
    st.write("Relationship Model Metrics:")
    st.write(f"Accuracy: {relationship_accuracy:.2f}")
    st.write(f"Precision: {relationship_precision:.2f}")
    st.write(f"Recall: {relationship_recall:.2f}")
    st.write(f"AUC: {relationship_auc:.2f}")
    
    #losigtic regression, lower accuracy, discarded
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # # Train
    # model = LogisticRegression(max_iter=1000, class_weight='balanced')
    # model.fit(X_train, y_train)
    #
    # # Evaluate
    # y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # st.write(f"Model Accuracy: {accuracy:.2f}")
    
    # # Save test data and predictions for app
    # joblib.dump(y_match_test, 'y_match_test.pkl')
    # joblib.dump(y_match_pred, 'y_match_pred.pkl')
    # joblib.dump(y_match_prob, 'y_match_prob.pkl')
    
    # joblib.dump(y_meeting_test, 'y_meeting_test.pkl')
    # joblib.dump(y_meeting_pred, 'y_meeting_pred.pkl')
    # joblib.dump(y_meeting_prob, 'y_meeting_prob.pkl')
    
    # joblib.dump(y_relationship_test, 'y_relationship_test.pkl')
    # joblib.dump(y_relationship_pred, 'y_relationship_pred.pkl')
    # joblib.dump(y_relationship_prob, 'y_relationship_prob.pkl')
    
    # # Save models
    # joblib.dump(match_model, 'match_likelihood_random_forest.pkl')
    # joblib.dump(meeting_model, 'meeting_likelihood_random_forest.pkl')
    # joblib.dump(relationship_model, 'relationship_likelihood_random_forest.pkl')
    
    # # Save the interest tag column names
    # joblib.dump(mlb.classes_.tolist(), 'interest_tags_classes.pkl')
    
    # # Save the model input column order
    # joblib.dump(X.columns.tolist(), 'model_input_columns.pkl')

    return match_model, meeting_model, relationship_model, mlb.classes_.tolist(), X.columns.tolist()
