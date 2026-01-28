# ✈️ Indian Airlines – Flight Analytics & Price Prediction Web Application

A comprehensive **Airline Management & Analytics Web Application** for analyzing Indian airline flight data and predicting ticket prices using **machine learning** and **data pipelines**.

---

## 📌 Project Overview

This project focuses on end-to-end airline data analysis, structured data pipelines, and ticket price prediction. It transforms raw airline data into meaningful insights using exploratory analysis, visual dashboards, and a machine learning–powered price predictor.

The system is designed as a **Data Engineering + Machine Learning mini-project**, suitable for academic evaluation, GitHub portfolios, and interviews.

---

## 🎯 Objectives

- Analyze airline and flight data  
- Build structured data pipelines  
- Predict airline ticket prices using ML  
- Understand factors affecting airfare  
- Demonstrate real-world data preprocessing and modeling skills  

---

## ✨ Features

### 📊 Dashboard
- Real-time statistics (Total Flights, Average Price, Airlines, Routes)
- Interactive charts for average price by airline
- Price range insights (Min / Avg / Max)

### 🎯 Price Predictor
- ML-powered ticket price prediction
- Predict price category: **Low / Medium / High**
- Based on **Random Forest Classification**
- Uses flight details such as airline, route, duration, stops, class, and days left

### 📋 Flight Data Explorer
- Browse all available flight data
- Paginated table view (20 flights per page)
- Color-coded price categories
- Detailed flight information

### 📈 Analysis
- Economy vs Business class price comparison
- Impact of stops on ticket pricing
- Key airline pricing insights

---

## 🗂️ Project Files

- **`Indian_Airlines.csv`**  
  Dataset containing Indian airline flight and pricing data

- **`DE_miniproject.ipynb`**  
  Jupyter Notebook implementing:
  - Data ingestion
  - Data cleaning & preprocessing
  - Exploratory Data Analysis (EDA)
  - Feature engineering
  - ML model training
  - Price prediction pipeline

- **`indian_airlines_app.html`**  
  Standalone frontend web application

- **`app.py`** *(optional)*  
  Flask backend for server-side processing

---

## 🔄 Data Pipeline Workflow

1. **Data Ingestion** – Load CSV dataset  
2. **Data Cleaning** – Handle missing & inconsistent values  
3. **Feature Engineering** – Encode categorical features  
4. **EDA** – Analyze trends and pricing patterns  
5. **Model Training** – Train Random Forest classifier  
6. **Prediction & Evaluation** – Predict ticket price category  

---

## 📊 Price Prediction Model

- **Algorithm**: Random Forest Classifier  
- **Features**:
  - Airline
  - Source City
  - Destination City
  - Departure & Arrival Time
  - Stops
  - Class
  - Duration
  - Days Left
- **Target**:
  - Low (< ₹3000)
  - Medium (₹3000–₹7000)
  - High (> ₹7000)
- **Training Split**: 80% Train / 20% Test  

---

## 🛠️ Technologies Used

- **Frontend**: React.js, Chart.js  
- **Styling**: Custom CSS (gradients & animations)  
- **Backend (Optional)**: Flask + Python  
- **Machine Learning**: scikit-learn  
- **Data Processing**: pandas, numpy  
- **Visualization**: Matplotlib, Seaborn  
- **Notebook**: Jupyter Notebook  

---

## 🚀 How to Run the Project

### Option 1: Standalone Web Application
1. Open `indian_airlines_app.html` in your browser  
2. Ensure `Indian_Airlines.csv` is in the same directory  
3. No server setup required  

---

### Option 2: Jupyter Notebook (Data Pipeline & ML)

## 1. Clone the repository
```bash
git clone https://github.com/your-username/airline-management-system.git
Navigate to the project directory
cd airline-management-system

## 2.Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn

## 3.Open Jupyter Notebook
jupyter notebook

## Run DE_miniproject.ipynb cell by cell

## Option 3: Flask Backend (Optional)
pip install -r requirements.txt
python app.py

Open:
http://localhost:5000

## 📚 Key Insights

✓ Business class tickets cost 2–3x more than Economy
✓ Non-stop flights are generally more expensive
✓ Booking early (14+ days) can reduce prices by 20–40%

## 🌐 Browser Compatibility

Chrome 90+
Firefox 88+
Safari 14+
Edge 90+

## 🔮 Future Enhancements

Add advanced ML models (XGBoost, Neural Networks)
Real-time price tracking
Flight search & filters
PDF report export
User authentication
Live airline API integration

## 🧑‍💻 Author

Siddhi Agarwal
B.Tech Computer Science & Engineering
MIT World Peace University

## 📄 License
This project is developed for educational and analytical purposes.
