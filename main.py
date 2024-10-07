import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import io


# Define the report generation function
def generate_report(y_pred, y_test, feature_importance_df, target_variable):
    report = "### Agricultural Yield Report\n\n"
    report += f"The model has predicted the yield based on the selected features.\n\n"

    # Analyze predictions
    if np.mean(y_pred) > np.mean(y_test):
        report += "The predicted yield is higher than the actual yield, which suggests good conditions for crop production.\n"
    else:
        report += "The predicted yield is lower than the actual yield, indicating potential areas for improvement.\n"

    # Feature importance analysis
    report += "\n### Feature Importance Analysis:\n"
    for index, row in feature_importance_df.iterrows():
        report += f"- **{row['Feature']}**: Importance Score: {row['Importance']:.2f}\n"

    report += "\n### Recommendations:\n"

    # Recommendations based on feature importance
    if 'average_rain_fall_mm_per_year' in feature_importance_df['Feature'].values:
        report += "- Ensure adequate rainfall or irrigation for optimal yield.\n"
    if 'avg_temp' in feature_importance_df['Feature'].values:
        report += "- Monitor and manage temperatures to ensure they are within the optimal range for crops.\n"
    if 'pesticides_tonnes' in feature_importance_df['Feature'].values:
        report += "- Use pesticides judiciously to maintain crop health.\n"

    report += "\n### Practices to Avoid:\n"
    report += "- Avoid over-reliance on chemical fertilizers which can degrade soil health over time.\n"
    report += "- Do not ignore soil health indicators; regularly test soil and amend accordingly.\n"
    report += "- Avoid monoculture practices, as they can lead to pest outbreaks and soil depletion.\n"

    report += "\n### Additional Resources:\n"
    report += "- For more information on sustainable agriculture practices, visit:\n"
    report += "  - [Sustainable Agriculture Research and Education (SARE)](https://www.sare.org/)\n"
    report += "  - [Rodale Institute](https://rodaleinstitute.org/)\n"
    report += "  - [FAO - Food and Agriculture Organization](https://www.fao.org/sustainable-agriculture/en/)\n"

    return report


# Set up the app
st.set_page_config(page_title="AI for Sustainable Agriculture", layout="wide")
st.title("🌱 Generative AI for Sustainable Agriculture")

# Description of the solution
st.subheader("🌟 Proposed Solution")
solution_description = """
Our Generative AI platform addresses the challenges of sustainable agriculture by leveraging data-driven insights to optimize farming practices. The platform analyzes diverse datasets, including soil health, weather patterns, and crop performance, enabling farmers to make informed decisions tailored to their unique environments. By modeling the impacts of various farming techniques and inputs, our AI fosters innovation in sustainable practices, promoting the adoption of eco-friendly methods. Additionally, community engagement is encouraged through shared insights and best practices, creating a collaborative ecosystem for continuous improvement.
"""
st.write(solution_description)

# How the innovation accelerates change
st.subheader("⚡ Accelerating Change with Technology")
technology_description = """
Our Generative AI platform accelerates change by employing advanced algorithms to analyze vast datasets in real-time, providing actionable insights for farmers. This technology enables precision farming, allowing for tailored solutions that maximize crop yield while minimizing environmental impact. Through continuous learning and adaptation, the platform not only predicts outcomes based on current practices but also simulates potential improvements, driving a shift towards sustainable agriculture. This data-centric approach equips farmers with the knowledge to implement innovative, resource-efficient practices, thereby enhancing productivity and environmental stewardship.
"""
st.write(technology_description)

# How the solution is unique
st.subheader("💎 Unique Features of Our Solution")
unique_description = """
Our Generative AI platform distinguishes itself by offering a comprehensive, integrated approach to sustainable agriculture. While many solutions focus on isolated aspects like crop monitoring or soil analysis, our platform combines real-time data from multiple sources into a unified framework. This holistic view enables farmers to simulate and visualize the outcomes of various practices, promoting innovation in sustainable methods. Additionally, the platform fosters community engagement, allowing farmers to share insights and best practices, creating a collaborative ecosystem that enhances learning and adaptation. This unique combination of features positions our solution as a leader in advancing sustainable agriculture.
"""
st.write(unique_description)

# Sidebar for file upload
st.sidebar.header("Upload Your Agricultural Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Styling options for better UI
st.markdown(""" 
    <style> 
    .reportview-container { 
        background: #F0F2F6; 
        font-family: 'Arial'; 
        color: black; 
    } 
    .sidebar .sidebar-content { 
        background: #D8E1E8; 
    } 
    .stButton>button { 
        color: white; 
        background-color: #5A9; 
        border-radius: 10px; 
        font-size: 18px; 
        margin: 20px; 
        padding: 10px; 
    } 
    </style> 
""", unsafe_allow_html=True)

# Check if a file is uploaded
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Show preview of the dataset
    st.subheader("📊 Data Preview")
    st.write(data.head())

    # Show dataset statistics
    st.subheader("📈 Dataset Statistics")
    st.write(data.describe())

    # Automated detection of target variable
    st.subheader("🎯 Automated Target Variable Detection")
    target_variable = st.selectbox("Select Target Column (for Yield Prediction)", data.columns)

    # Handle missing values
    if data.isnull().sum().sum() > 0:
        st.warning("⚠️ Your dataset contains missing values. These will be automatically handled.")
        data = data.fillna(data.mean())  # Simple imputation for missing values

    # Feature selection
    st.subheader("🔍 Select Features for Training")
    features = st.multiselect("Select the Features", [col for col in data.columns if col != target_variable])

    if st.button("Train Model"):
        if features:
            X = data[features]
            y = data[target_variable]

            # Handle categorical data by applying LabelEncoder
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.success(f"Model trained successfully! R² score: {r2:.2f}, MSE: {mse:.2f}")

            # Feature importance
            importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            st.subheader("🔑 Feature Importance")
            fig = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h',
                         title="Feature Importance")
            st.plotly_chart(fig)

            # Generate report
            report = generate_report(y_pred, y_test, feature_importance_df, target_variable)
            st.subheader("📄 Generated Report")
            st.write(report)

            # Download predictions
            st.subheader("📄 Download Predictions")
            results_df = pd.DataFrame({"Actual Yield": y_test, "Predicted Yield": y_pred})
            st.write(results_df)

            # CSV download button
            output = io.StringIO()
            results_df.to_csv(output, index=False)
            st.download_button(label="Download Predictions as CSV", data=output.getvalue(), file_name="predictions.csv",
                               mime="text/csv")
        else:
            st.warning("Please select at least one feature to train the model.")
else:
    st.sidebar.info("Please upload a CSV file to begin.")

# Advanced Data Visualization
if uploaded_file is not None:
    st.header("📊 Advanced Data Visualizations")

    # Correlation heatmap: Only use numerical columns
    st.subheader("🔗 Correlation Matrix")
    num_columns = data.select_dtypes(include=['float64', 'int64']).columns  # Only numeric columns
    if len(num_columns) > 1:
        corr = data[num_columns].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("The dataset doesn't have enough numerical columns for a correlation matrix.")

    # Pair plot: Only use numeric columns
    st.subheader("🌐 Pair Plot")
    if len(num_columns) > 1:
        fig = sns.pairplot(data[num_columns])
        st.pyplot(fig)
    else:
        st.warning("The dataset doesn't have enough numerical columns for a pair plot.")

    # Distribution of Target Variable
    st.subheader(f"📊 Distribution of {target_variable}")
    if data[target_variable].dtype in ['float64', 'int64']:
        fig = px.histogram(data, x=target_variable, nbins=50, title=f"{target_variable} Distribution")
        st.plotly_chart(fig)
    else:
        st.warning(f"The target variable '{target_variable}' is not numeric and cannot be plotted as a distribution.")
