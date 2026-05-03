🧠 Breast Cancer Classification using Artificial Neural Network (ANN)




📌 Overview

This project builds an Artificial Neural Network (ANN) model to classify breast cancer tumors into:

🟢 Benign (Non-cancerous)
🔴 Malignant (Cancerous)

The goal is to support early detection using machine learning techniques.

📊 Dataset Information

The dataset contains medical features extracted from breast mass images:

Radius
Texture
Perimeter
Area
Smoothness
Compactness
Concavity
Symmetry
🎯 Target Variable:
M → Malignant (1)
B → Benign (0)
⚙️ Project Pipeline
1️⃣ Data Loading
data = pd.read_csv("breast-cancer.csv")

2️⃣ Data Preprocessing
Removed unnecessary columns (e.g., id)
Encoded target variable:
data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

3️⃣ Data Visualization

Understanding feature relationships using correlation heatmap:

sns.heatmap(data.corr(), annot=True)

4️⃣ Feature Engineering
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

5️⃣ Train-Test Split
80% Training
20% Testing
6️⃣ Feature Scaling

Applied StandardScaler for better ANN performance.

🧠 Model Architecture (ANN)
model = Sequential()

model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

🧩 Layers:
Input Layer: 16 neurons
Hidden Layer: 8 neurons
Output Layer: 1 neuron (Sigmoid)
🏋️ Training Configuration
Optimizer: Adam
Loss Function: Binary Crossentropy
Epochs: 50
Batch Size: 16
📈 Evaluation Metrics

The model is evaluated using:

Accuracy Score
Confusion Matrix
Classification Report
ROC Curve
📊 Results Visualization

Training vs Validation accuracy/loss graphs were used to analyze model performance and detect overfitting.

🎯 Objective

To build a reliable AI model capable of assisting in early breast cancer detection by classifying tumors accurately.

🛠 Tech Stack
Python 🐍
Pandas
NumPy
Matplotlib 📊
Seaborn 🌊
Scikit-learn 🤖
TensorFlow / Keras 🧠
🚀 Future Improvements
Hyperparameter tuning
Cross-validation
Feature selection optimization
Deployment as a web app (Flask / Streamlit)

⭐ If you like this project

Give it a ⭐ on GitHub — it motivates me to build more AI projects!
