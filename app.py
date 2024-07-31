import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the trained model and pre-fitted scaler, and label encoders
model = joblib.load('voting_classifier_model.pkl')
scaler = joblib.load('minmax_scaler.pkl')  # Load your fitted MinMaxScaler
label_encoders = joblib.load('label_encoders.pkl')  # Load your fitted LabelEncoders

# Define the expected columns based on the training phase
expected_columns = [
    'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate',
    'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel_0', 'BusinessTravel_1',
    'BusinessTravel_2', 'Department_0', 'Department_1', 'Department_2', 'Education_0',
    'Education_1', 'Education_2', 'Education_3', 'Education_4', 'EducationField_0',
    'EducationField_1', 'EducationField_2', 'EducationField_3', 'EducationField_4',
    'EducationField_5', 'EnvironmentSatisfaction_0', 'EnvironmentSatisfaction_1',
    'EnvironmentSatisfaction_2', 'EnvironmentSatisfaction_3', 'Gender_0', 'Gender_1',
    'JobInvolvement_0', 'JobInvolvement_1', 'JobInvolvement_2', 'JobInvolvement_3',
    'JobRole_0', 'JobRole_1', 'JobRole_2', 'JobRole_3', 'JobRole_4', 'JobRole_5',
    'JobRole_6', 'JobRole_7', 'JobRole_8', 'JobSatisfaction_0', 'JobSatisfaction_1',
    'JobSatisfaction_2', 'JobSatisfaction_3', 'MaritalStatus_0', 'MaritalStatus_1',
    'MaritalStatus_2', 'NumCompaniesWorked_0', 'NumCompaniesWorked_1', 'NumCompaniesWorked_2',
    'NumCompaniesWorked_3', 'NumCompaniesWorked_4', 'NumCompaniesWorked_5', 'NumCompaniesWorked_6',
    'NumCompaniesWorked_7', 'NumCompaniesWorked_8', 'NumCompaniesWorked_9', 'OverTime_0',
    'OverTime_1', 'PerformanceRating_0', 'PerformanceRating_1', 'RelationshipSatisfaction_0',
    'RelationshipSatisfaction_1', 'RelationshipSatisfaction_2', 'RelationshipSatisfaction_3',
    'StockOptionLevel_0', 'StockOptionLevel_1', 'StockOptionLevel_2', 'StockOptionLevel_3',
    'TrainingTimesLastYear_0', 'TrainingTimesLastYear_1', 'TrainingTimesLastYear_2',
    'TrainingTimesLastYear_3', 'TrainingTimesLastYear_4', 'TrainingTimesLastYear_5',
    'TrainingTimesLastYear_6', 'WorkLifeBalance_0', 'WorkLifeBalance_1', 'WorkLifeBalance_2',
    'WorkLifeBalance_3'
]

# Custom CSS for styling
st.markdown("""
    <style>
    .stButton button {
        background-color: #6c757d;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 10px;
    }
    .stMarkdown h3 {
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to scale and preprocess input data
def preprocess_input(data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])

    # Specify the columns
    numeric_features = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 
                        'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 
                        'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                        'YearsWithCurrManager', 'YearsSinceLastPromotion']

    label_encoding_features = ['BusinessTravel', 'Department', 'EducationField', 'OverTime', 'JobRole']

    # Apply Label Encoding
    for column in label_encoding_features:
        le = label_encoders[column]
        input_df[column] = le.transform(input_df[column])

    # One-hot encode categorical features
    input_df = pd.get_dummies(input_df, columns=label_encoding_features)

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    # Scale numeric features
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    return input_df.values

# Function to get user input from Streamlit interface
def get_user_input():
    with st.sidebar:
        st.header("Employee Details")
        
        age = st.number_input('Age', min_value=18, max_value=60, value=30)
        daily_rate = st.number_input('Daily Rate', min_value=100, max_value=1500, value=800)
        distance_from_home = st.number_input('Distance From Home', min_value=1, max_value=30, value=10)
        hourly_rate = st.number_input('Hourly Rate', min_value=30, max_value=100, value=50)
        monthly_income = st.number_input('Monthly Income', min_value=1000, max_value=20000, value=5000)
        monthly_rate = st.number_input('Monthly Rate', min_value=1000, max_value=25000, value=10000)
        percent_salary_hike = st.number_input('Percent Salary Hike', min_value=0, max_value=30, value=15)
        total_working_years = st.number_input('Total Working Years', min_value=0, max_value=40, value=10)
        years_at_company = st.number_input('Years at Company', min_value=0, max_value=30, value=5)
        years_in_current_role = st.number_input('Years in Current Role', min_value=0, max_value=20, value=5)
        years_since_last_promotion = st.number_input('Years Since Last Promotion', min_value=0, max_value=20, value=2)
        years_with_curr_manager = st.number_input('Years with Current Manager', min_value=0, max_value=20, value=3)

        business_travel = st.selectbox('Business Travel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
        department = st.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources'])
        education_field = st.selectbox('Education Field', ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'])
        job_role = st.selectbox('Job Role', [
            'Sales Executive', 'Research Scientist', 'Laboratory Technician', 
            'Manufacturing Director', 'Healthcare Representative', 
            'Manager', 'Sales Representative', 'Research Director', 'Human Resources'
        ])
        overtime = st.selectbox('OverTime', ['Yes', 'No'])

    data = {
        'Age': age,
        'DailyRate': daily_rate,
        'DistanceFromHome': distance_from_home,
        'HourlyRate': hourly_rate,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': monthly_rate,
        'PercentSalaryHike': percent_salary_hike,
        'TotalWorkingYears': total_working_years,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager,
        'BusinessTravel': business_travel,
        'Department': department,
        'EducationField': education_field,
        'JobRole': job_role,
        'OverTime': overtime,
    }
    
    return data

# Main function to run the Streamlit app
def main():
    st.title("Employee Attrition Prediction")
    st.write("Enter the employee details to predict attrition using the sidebar.")
    
    # Get user input
    user_input = get_user_input()
    
    # Button to make prediction
    if st.button('Predict'):
        # Preprocess the user input
        input_data = preprocess_input(user_input)
        
        # Convert input_data to a numpy array and reshape for prediction
        input_array = np.array(input_data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)
        prediction_proba = model.predict_proba(input_array)
        
        # Show only the probability of attrition
        attrition_probability = prediction_proba[0][1]
        
        # Determine the color based on the probability
        color = "red" if attrition_probability > 0.5 else "green"
        
        # Display the prediction result with color
        st.markdown(f"<h3 style='color:{color};'>Probability of Attrition: {attrition_probability:.2%}</h3>", unsafe_allow_html=True)

        if prediction[0] == 1:
            st.error("The employee is likely to leave the company.")
        else:
            st.success("The employee is likely to stay in the company.")

if __name__ == '__main__':
    main()
