import streamlit as st
import pandas as pd
import joblib 

encoded_features = joblib.load("encoded_features.pkl")

model = joblib.load("model.pkl")

loan_df = pd.read_csv(r"updated_loan_data.csv")
loan_df.columns = loan_df.columns.str.lower().str.replace(" ","_")

# App title
st.title("Bank Loan Defaulter Prediction")

# User input form
st.header("Enter Loan Details Below")


# Numerical Features
loan_amount = st.number_input("Loan Amount", min_value=1000,max_value=35000, value=9895, step=1000)
funded_amount = st.number_input("Funded Amount", min_value=1000,max_value=35000, value=9204, step=1000)
interest_rate = st.number_input("Interest Rate", min_value=5.0, max_value=27.0, value=10.57, step=0.1)
income = st.number_input("Income", min_value=14500.0,max_value=400000.0, value=29113.0, step=1000.0)
debit_to_income = st.number_input("Debit to Income", min_value=0.0, max_value=23.10, value=0.2, step=0.1)
open_account = st.number_input("Open Account", min_value=2, max_value=37,value=9, step=1)
total_accounts = st.number_input("Total Accounts", min_value=4,max_value=72, value=16, step=1)
revolving_balance = st.number_input("Revolving Balance", min_value=0,max_value=115000, value=5140, step=1000)
revolving_utilities = st.number_input("Revolving Utilities (%)", min_value=0.0, max_value=100.0, value=59.84, step=2.0)
total_revolving_credit_limit = st.number_input("Total Revolving Credit Limit", min_value=1000,max_value=200000, value=1589, step=500)
last_week_pay = st.number_input("Last Week Pay",min_value=0, max_value=160,value=25,step=10 )  
total_received_interest = st.number_input("Total Received Interest", min_value=4.0,max_value=1487.38, value=5.0, step=100.0)
total_received_late_fee = st.number_input("Total Received Late Fee", min_value=0.0,max_value=4.0 ,value=0.059, step=2.0)
recoveries = st.number_input("Recoveries", min_value=0.0, max_value=4000.0,value=6.24, step=100.0)
collection_recovery_fee = st.number_input("Collection Recovery Fee", min_value=0.0,max_value=160.0, value=1.08, step=10.0)
total_collection_amount = st.number_input("Total Collection Amount", min_value=1,max_value=16000,value=37, step=100)
total_current_balance = st.number_input("Total Current Balance", min_value=600,max_value=1000000, value=118057, step=500)


#discrete numerical features
term = st.selectbox("Term", options=[36,58, 59]) 
delinquency_two_years = st.number_input("Delinquency - Two Years", min_value=0, max_value=8 ,value=0, step=1)
inquires_six_months = st.number_input("Inquires - Six Months", min_value=0,max_value =5 ,value=0, step=1)
public_record = st.number_input("Public Record", min_value=0,max_value=4, value=0, step=1)
collection_12_months_medical = st.selectbox("Collection 12 Months Medical",options=[0,1] )

# Categorical Features
grade = st.selectbox("Grade", options=["A", "B", "C", "D", "E", "F", "G"])
home_ownership = st.selectbox("Home Ownership", options=["RENT", "OWN", "MORTGAGE"])
verification_status = st.selectbox("Verification Status", options=["Verified", "Source Verified", "Not Verified"])
loan_title = st.selectbox("Loan Title",loan_df["loan_title"].unique())
initial_list_status = st.selectbox("Initial List Status", options=["f", "w"])
application_type = st.selectbox("Application Type", options=["INDIVIDUAL","JOINT"])

input_data = pd.DataFrame(
    {
        "loan_amount": [loan_amount],
        "funded_amount": [funded_amount],
        "term": [term],
        "interest_rate": [interest_rate],
        "grade": [grade],
        "home_ownership": [home_ownership],
        "income": [income],
        "verification_status": [verification_status],
        "loan_title": [loan_title],
        "debit_to_income": [debit_to_income],
        "delinquency_two_years": [delinquency_two_years],
        "inquires_six_months": [inquires_six_months],
        "open_account": [open_account],
        "public_record": [public_record],
        "revolving_balance": [revolving_balance],
        "revolving_utilities": [revolving_utilities],
        "total_accounts": [total_accounts],
        "initial_list_status": [initial_list_status],
        "total_received_interest": [total_received_interest],
        "total_received_late_fee": [total_received_late_fee],
        "recoveries": [recoveries],
        "collection_recovery_fee": [collection_recovery_fee],
        "collection_12_months_medical": [collection_12_months_medical],
        "application_type": [application_type],
        "last_week_pay": [last_week_pay],
        "total_collection_amount": [total_collection_amount],
        "total_current_balance": [total_current_balance],
        "total_revolving_credit_limit": [total_revolving_credit_limit]
    }
) 

# Encoding the training data for matching the index with test data
train_encoded = pd.get_dummies(loan_df.drop(columns="loan_status"), drop_first=True)

# Extracting the feature names for matching the index with test data
train_feature_names = train_encoded.columns.tolist()

# Encode the input data
input_data_encoded = pd.get_dummies(input_data, drop_first=True)

# Reindex to match the training features
input_data_encoded = input_data_encoded.reindex(columns=train_feature_names, fill_value=0)



if st.button("classification"):
    # Make prediction using the pipeline model 
    classification = model.predict(input_data_encoded[encoded_features])
    
    loan_status = "Defaulter" if classification[0]== 1 else "Non-Defaulter"
    # Display the result
    st.subheader(f"Bank loan status: {loan_status}")



