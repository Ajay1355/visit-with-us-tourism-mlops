from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'deployment' / 'model.joblib'

st.set_page_config(page_title='Visit with Us — Package Prediction', page_icon='✈️', layout='centered')
st.title('Visit with Us — Wellness Tourism Package Prediction')
st.write('Enter customer details to estimate the likelihood of purchasing the package.')

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

age = st.number_input('Age', min_value=18, max_value=100, value=35)
type_contact = st.selectbox('Type of Contact', ['Self Enquiry', 'Company Invited'])
city_tier = st.selectbox('City Tier', [1, 2, 3])
duration = st.number_input('Duration of Pitch', min_value=0, max_value=60, value=10)
occupation = st.selectbox('Occupation', ['Salaried', 'Small Business', 'Large Business', 'Free Lancer', 'Government Service'])
gender = st.selectbox('Gender', ['Male', 'Female'])
persons = st.number_input('Number of Persons Visiting', min_value=1, max_value=20, value=2)
followups = st.number_input('Number of Followups', min_value=0, max_value=20, value=3)
product = st.selectbox('Product Pitched', ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
property_star = st.selectbox('Preferred Property Star', [3, 4, 5])
marital = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
trips = st.number_input('Number of Trips', min_value=0, max_value=30, value=3)
passport = st.selectbox('Passport', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
satisfaction = st.selectbox('Pitch Satisfaction Score', [1, 2, 3, 4, 5])
own_car = st.selectbox('Own Car', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
children = st.number_input('Number of Children Visiting', min_value=0, max_value=10, value=1)
designation = st.selectbox('Designation', ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
income = st.number_input('Monthly Income', min_value=0, max_value=1000000, value=25000)

if st.button('Predict Package Purchase', type='primary'):
    input_df = pd.DataFrame([{
        'Age': age,
        'TypeofContact': type_contact,
        'CityTier': city_tier,
        'DurationOfPitch': duration,
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': persons,
        'NumberOfFollowups': followups,
        'ProductPitched': product,
        'PreferredPropertyStar': property_star,
        'MaritalStatus': marital,
        'NumberOfTrips': trips,
        'Passport': passport,
        'PitchSatisfactionScore': satisfaction,
        'OwnCar': own_car,
        'NumberOfChildrenVisiting': children,
        'Designation': designation,
        'MonthlyIncome': income
    }])

    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(probability >= 0.50)

    st.metric('Purchase Probability', f'{probability:.1%}')
    if prediction == 1:
        st.success('Likely to purchase the Wellness Tourism Package.')
    else:
        st.info('Less likely to purchase the Wellness Tourism Package.')