import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
@st.cache
def load_data():
    data = pd.read_csv('uterine_carcinoma_data.csv')
    return data.copy()

data = load_data()

# Data Preparation
# Handle missing values
# Fill missing values for numeric columns with their median
numeric_cols = data.select_dtypes(include=['number']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Fill missing values for categorical columns with the mode
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Define features and target
X = data.drop(['Patient ID', 'Sample ID', 'Overall Survival Status'], axis=1)
y = data['Overall Survival Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Streamlit App
st.title('Uterine Carcinoma Survival Prediction')

st.write('## Model Performance')
st.write(f'Accuracy: {accuracy}')
st.write('Confusion Matrix:')
st.write(conf_matrix)
st.write('Classification Report:')
st.write(class_report)

st.write('## Make Predictions')

# Input form for new data
def user_input_features():
    cancer_type = st.selectbox('Cancer Type Detailed', data['Cancer Type Detailed'].unique())
    disease_free_status = st.selectbox('Disease Free Status', data['Disease Free Status'].unique())
    disease_specific_status = st.selectbox('Disease-specific Survival status', data['Disease-specific Survival status'].unique())
    mutation_count = st.slider('Mutation Count', int(data['Mutation Count'].min()), int(data['Mutation Count'].max()), int(data['Mutation Count'].median()))
    fraction_genome_altered = st.slider('Fraction Genome Altered', float(data['Fraction Genome Altered'].min()), float(data['Fraction Genome Altered'].max()), float(data['Fraction Genome Altered'].median()))
    diagnosis_age = st.slider('Diagnosis Age', int(data['Diagnosis Age'].min()), int(data['Diagnosis Age'].max()), int(data['Diagnosis Age'].median()))
    msi_mantis_score = st.slider('MSI MANTIS Score', float(data['MSI MANTIS Score'].min()), float(data['MSI MANTIS Score'].max()), float(data['MSI MANTIS Score'].median()))
    msisensor_score = st.slider('MSIsensor Score', float(data['MSIsensor Score'].min()), float(data['MSIsensor Score'].max()), float(data['MSIsensor Score'].median()))
    race_category = st.selectbox('Race Category', data['Race Category'].unique())
    subtype = st.selectbox('Subtype', data['Subtype'].unique())
    tumor_type = st.selectbox('Tumor Type', data['Tumor Type'].unique())

    data_dict = {
        'Cancer Type Detailed': cancer_type,
        'Disease Free Status': disease_free_status,
        'Disease-specific Survival status': disease_specific_status,
        'Mutation Count': mutation_count,
        'Fraction Genome Altered': fraction_genome_altered,
        'Diagnosis Age': diagnosis_age,
        'MSI MANTIS Score': msi_mantis_score,
        'MSIsensor Score': msisensor_score,
        'Race Category': race_category,
        'Subtype': subtype,
        'Tumor Type': tumor_type
    }

    features = pd.DataFrame(data_dict, index=[0])
    return features

input_df = user_input_features()

# Encode input data
for column in input_df.select_dtypes(include=['object']).columns:
    input_df[column] = label_encoders[column].transform(input_df[column])

# Standardize input data
input_df = scaler.transform(input_df)

# Predict
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
st.write('Overall Survival Status:', 'LIVING' if prediction[0] == 0 else 'DECEASED')

st.subheader('Prediction Probability')
st.write(prediction_proba)
