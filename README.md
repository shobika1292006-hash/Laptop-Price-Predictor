# Laptop Price Predictor 💻📈

Laptop Price Predictor is an interactive machine learning web dashboard powered by **Streamlit**. It fits a Linear Regression model using **scikit-learn** to estimate the market price of laptops based on key specifications: **RAM (GB)**, **SSD Storage (GB)**, and **Processor Generation**.

The application features a dark-themed premium design, dynamic inputs, custom hover metrics, Plotly visualization plots, and an interactive dataset examiner.

---

## 🌟 Key Features

- **Dynamic Prediction**: Adjust hardware specifications (RAM, SSD, and Processor Generation) in real-time to get instant laptop price estimates.
- **Model Mathematical Formula**: Displays the live computed regression coefficients and intercept in mathematical LaTeX format, detailing the exact price contribution per unit of hardware spec.
- **Interactive Plotly Visualizations**:
  - **Component Breakdown Chart**: Shows how much each selected hardware specification contributes to the estimated price relative to the baseline price.
  - **Actual vs. Predicted Plot**: Visualizes predictions against target validation data with a regression trendline.
  - **Feature Importance Chart**: Graphs the model coefficients directly, showing the dollar value added per GB of RAM, GB of SSD, or Generation of Processor.
- **Model Performance Metrics**: Displays $R^2$ score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to gauge accuracy.
- **Dataset Explorer**: Raw data viewer, summary statistics, and direct options to download the dataset as a CSV.
- **Custom CSV Upload**: Allows users to import external laptop data files to train custom predictors.

---

## 🚀 Local Setup Instructions

Follow these steps to run the application locally on your computer.

### 1. Prerequisites
Ensure you have **Python 3.8+** installed. You can check your version by running:
```bash
python --version
```

### 2. Install Dependencies
Navigate to this project directory in your terminal and install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
Start the Streamlit development server:
```bash
streamlit run app.py
```

Streamlit will launch a local server and automatically open the application in your default web browser (usually at `http://localhost:8501`).

---

## 📂 Project Structure

- `app.py`: Main Streamlit application containing layout configurations, UI styling, model training, and analytical charts.
- `laptop_prices.csv`: Sample laptop pricing dataset containing specifications (RAM, SSD, Processor_Gen) and price records.
- `requirements.txt`: Python package requirements.
- `README.md`: Project documentation and guides.
"# Laptop-Price-Predictor" 
