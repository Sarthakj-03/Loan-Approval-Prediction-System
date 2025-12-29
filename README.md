# Loan-Prediction
Designed and developed a full-stack machine learning web application that predicts loan approval eligibility with 82% accuracy.
Built an interactive Streamlit interface that analyzes 11 financial features (income, credit history, employment, loan amount, etc.) and provides interpretable predictions with confidence scores and feature importance visualization. Implemented end-to-end data preprocessing, categorical encoding, feature engineering, and proper train-test splitting following ML best practices. The application is production-ready and deployable to Streamlit Cloud, demonstrating proficiency in Python, scikit-learn, data science, and web development. The emphasis on model interpretability and user experience showcases understanding that effective ML systems must balance accuracy with explainability—especially critical in financial decision-making contexts.

Predicts loan approval status (Approved / Not Approved).
Uses 11+ financial and personal features (income, credit history, employment, loan amount, dependents, property area, etc.).
Shows approval probability / confidence scores.
Displays key factors affecting the decision.
Clean, beginner-friendly UI with tabs: Predict, Model Info, How It Works.

Tech stack
  -Python
  -Streamlit
  -Pandas
  -NumPy
  -Scikit-learn (RandomForestClassifier)
  -Matplotlib
  -Seaborn
  
Model
  -RandomForestClassifier

What this app does
  -Shows a simple overview of the dataset (rows, features, class balance of approved vs rejected loans).
  -Predicts whether a loan is likely to be Approved or Not Approved based on user inputs.
  -Displays approval and rejection probabilities as confidence scores.
  -Highlights key factors like credit history, income–loan ratio, and employment type that influence the prediction.
  -Provides a small “Model Info” section with feature importance and basic model details.
  -Includes a “How It Works” / FAQ style page explaining the model and factors in simple language.

I mainly built it to learn how to:
  -Load and clean tabular data using Pandas.
  -Do basic data analysis and feature engineering for a classification problem.
  -Train a simple ML model using scikit-learn (Random Forest).
  -Make charts and feature importance plots with Matplotlib / Seaborn.
  -Create an interactive ML dashboard with Streamlit where users can change inputs and see predictions.

How to run locally
  -Clone this repo:
    -git clone https://github.com/<your-username>/loan-approval-predictor.git
    -cd loan-approval-predictor
  -Create a virtual environment (optional but recommended):
    -python -m venv venv
    -venv\Scripts\activate      # on Windows
    -# or
    -source venv/bin/activate   # on Linux / Mac
  -Install the requirements:
    -pip install -r requirements.txt
    -# or, if no file:
    -pip install streamlit pandas numpy scikit-learn matplotlib seaborn
    -Put your dataset file in the project folder with name loan_data.csv (or keep the built‑in synthetic data if your script generates it).

  Run the Streamlit app:
    -streamlit run loan_app.py
    -Open the link shown in the terminal (usually http://localhost:8501).

Pages in the app
  -Predict
    -Main page where the user enters income, loan amount, credit history, employment type, dependents, etc.
    -Shows prediction (Approved / Not Approved) and confidence scores.
    -Displays a short explanation of key factors that affected the decision.
  -Model Info
    -Shows basic information about the model (Random Forest, number of trees, features used).
    -Displays a feature importance chart to show which inputs matter most.
  -How It Works
    -Simple explanation of how the model was trained.
    -Lists the main factors that usually affect loan approval.
    -Tips on how a user could improve their chances in a real scenario.

Notes
  -This is a beginner / practice project, so the code is not perfect.
  -The main goal was to learn the basics of data preprocessing, model training and building interactive ML dashboards with Streamlit.
  -You can fork this repo and improve it by:
    -Adding more filters and advanced metrics.
    -Trying different ML models or hyperparameter tuning.
    -Using a real Kaggle loan dataset and extending the analysis with more visualizations.
