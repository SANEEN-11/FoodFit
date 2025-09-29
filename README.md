<p align="center">
  <img src="frontend\public\images\logo5.png" alt="FoodFit Logo" width="200"/>
</p>

## 📋 Overview

FoodFit is a comprehensive platform that helps users optimize their food consumption, reduce waste, and improve nutritional outcomes through data-driven insights. This was developed as a part of Hacksus'25 following the theme of UN SDG 12, "RESPONSIBLE CONSUMPTION AND PRODUCTION". By combining machine learning analytics with user-friendly tracking tools, FoodFit creates personalized recommendations for healthier eating habits and more sustainable food practices.

## ✨ Features

### Core Functionality
- **Food Consumption Analysis**: Track and analyze your eating habits over time
- **Waste Reduction**: Get insights on how to minimize food waste
- **Nutritional Optimization**: Receive personalized portion recommendations
- **Data Visualization**: View your progress through intuitive charts and graphs
- **ML-Powered Insights**: Benefit from machine learning algorithms that improve over time

### Smart Features
- **Personalized Portion Recommendations**: AI recommends ideal portion sizes based on user history, appetite, and dietary needs
- **Waste Forecast Meter**: Predicts the likelihood of leftovers for a dish using feedback from similar users
- **Smart Meal Suggestions**: Alternative meal/item combinations suggested to reduce over-ordering
- **Consumption Feedback Loop**: Users can rate portion accuracy and report food leftover levels to improve prediction models
- **Restaurant Integration**: Portion size display and recommendations for restaurant meals

## 🛠️ Tech Stack

### Backend
- **Python 3.12+** - Core backend language
- **Flask** - Web framework with CORS support
- **Machine Learning Stack**:
  - **scikit-learn** - For RandomForest and regression models
  - **XGBoost** - Advanced gradient boosting
  - **pandas & numpy** - Data manipulation and analysis
  - **matplotlib & seaborn** - Data visualization
- **RESTful API** architecture

### Frontend
- **TypeScript** - Type-safe JavaScript development
- **React 19** - Modern React with latest features
- **Vite** - Fast build tool and dev server
- **Chakra UI** - Component library for consistent styling
- **React Router** - Client-side routing
- **React Query** - Server state management and caching
- **Zustand** - Lightweight state management
- **Modern HTML/CSS** - Responsive design

### Frontend Architecture
The frontend is organized into specialized modules:
- **Authentication System** (`src/pages/auth/`) - User login and authentication
- **Food Scanner** (`src/pages/scanner/`) - Food recognition and scanning
- **Restaurant Management** (`src/pages/restaurants/`) - Restaurant listings and menus
- **Cart & Donation System** (`src/pages/cart/`, `src/pages/donation/`) - Shopping cart and food donation features
- **Shared Components** (`src/components/`) - Reusable UI components

## 🤖 Machine Learning Models

### 1. Portion Prediction Model
- **Algorithm**: Gradient Boosting Regressor
- **Purpose**: Predicts optimal portion sizes for individual users
- **Features**: User demographics (age, weight, height, activity level), food nutritional data, meal timing, hunger levels
- **Training Data**: [`optimal_portions.csv`](backend/datasets/optimal_portions.csv)

### 2. Waste Prediction Model
- **Algorithm**: Random Forest Classifier
- **Purpose**: Predicts likelihood of food waste (Small, Medium, Large categories)
- **Features**: Portion size, user characteristics, meal context, time factors
- **Output**: Probability score and waste likelihood classification

### 3. Recommendation System
- **Algorithm**: Collaborative Filtering with K-Nearest Neighbors
- **Purpose**: Suggests alternative meals based on similar user preferences
- **Method**: User-food rating matrix with cosine similarity
- **Features**: User consumption history, satisfaction ratings, food categories

### Model Performance & Features
- **Feature Engineering**: Includes BMR calculation, macronutrient ratios, temporal features, and user consumption patterns
- **Data Processing**: Automated feature scaling, one-hot encoding for categorical variables
- **Validation**: Train-test split with performance metrics (MSE, R², accuracy, classification reports)

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the development server:
   ```bash
   python app.py
   ```

The backend will start on `http://localhost:5000` with the following endpoints:
- `GET /health` - Health check
- `POST /predict_portion` - Get portion recommendations
- `POST /predict_waste` - Get waste likelihood predictions  
- `POST /recommend_meals` - Get alternative meal suggestions

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to the local development URL shown in your terminal (typically `http://localhost:5173`)

### Quick Start with Sample Data
The system comes pre-loaded with sample datasets including:
- User profiles with demographic and dietary information
- Food items database with nutritional data
- Historical consumption patterns for model training

## 📂 Project Structure

```
FoodFit/
├── backend/                # Python backend code
│   ├── app.py              # Flask API server with ML endpoints
│   ├── MLcode.py           # Machine learning algorithms and models
│   ├── test_api.py         # API testing utilities
│   ├── datasets/           # Training data and food databases
│   │   ├── consumption_history.csv
│   │   ├── food_items.csv
│   │   ├── user_profiles.csv
│   │   └── optimal_portions.csv
│   └── requirements.txt    # Python dependencies
│
├── frontend/               # TypeScript/React frontend
│   ├── public/             # Static assets and images
│   ├── src/                # Source code
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Page components (auth, scanner, restaurants, cart)
│   │   ├── lib/            # Utilities and store management
│   │   └── main.tsx        # Application entry point
│   ├── package.json        # Node dependencies and scripts
│   └── vite.config.ts      # Vite configuration
│
├── misc/                   # Additional project resources
│   ├── abstract.txt        # Project abstract and features
│   ├── arch.ts             # Architecture documentation
│   └── user_1_waste_analysis.png  # Sample ML analysis output
│
└── README.md               # This file
```

## 📊 Data Analysis & Insights

FoodFit leverages multiple datasets to power its recommendations:

### Core Datasets
- **User Profiles**: Demographics, dietary preferences, activity levels, TDEE calculations
- **Food Database**: Nutritional information, categories, standard portions
- **Consumption History**: User eating patterns, portion sizes, satisfaction ratings
- **Optimal Portions**: ML training data for portion size predictions

### Analytics Features
- **Waste Pattern Visualization**: Per-user waste analysis by meal type, food category, and time
- **Consumption Trends**: Track eating habits and portion accuracy over time
- **Recommendation Analytics**: Performance metrics for suggested meals and portions
- **Feedback Loop Integration**: Continuous model improvement through user feedback

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

Project Link: [https://github.com/RonnMath03/FoodFit](https://github.com/RonnMath03/FoodFit)

---

<p align="center">Made with ❤️ for better nutrition and sustainable eating</p>

