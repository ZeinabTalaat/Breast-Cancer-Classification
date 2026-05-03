🧠 Breast Cancer Classification using ANN
📌 Overview

This project uses an Artificial Neural Network (ANN) to classify breast cancer tumors into:

🟢 Benign (Non-cancerous)
🔴 Malignant (Cancerous)

The goal is early and accurate cancer detection using machine learning.

📊 Dataset

The dataset contains medical features such as:

Radius
Texture
Perimeter
Area
Smoothness
Compactness
Concavity
Symmetry


🎯 Target:
M → Malignant (1)
B → Benign (0)



⚙️ Workflow
1. Load Data
data = pd.read_csv("breast-cancer.csv")

2. Data Cleaning
Remove unnecessary columns
Encode target variable:
data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

3. Visualization

Correlation heatmap:

sns.heatmap(data.corr(), annot=True)

4. Features & Target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

5. Train-Test Split
80% Training
20% Testing
6. Feature Scaling

Used StandardScaler for normalization.






🧠 Model (ANN)
model = Sequential()

model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

Layers:
Input → 16 neurons
Hidden → 8 neurons
Output → Sigmoid




🏋️ Training
Optimizer: Adam
Loss: Binary Crossentropy
Epochs: 50
Batch Size: 16






📊 Evaluation
Accuracy
Confusion Matrix
Classification Report
ROC Curve




🎯 Goal

Build a model that helps in early detection of breast cancer with high accuracy.




🛠 Tech Stack

Python | Pandas | NumPy | Matplotlib | Seaborn | Scikit-learn | TensorFlow
