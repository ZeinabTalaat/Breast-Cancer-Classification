# Breast-Cancer-Classification
AI-based breast cancer prediction model using data preprocessing, visualization, and deep learning techniques.
Breast Cancer Classification using Artificial Neural Network
📌 Project Overview
This project aims to build a machine learning model to classify breast cancer tumors as Malignant (M) or Benign (B) using an Artificial Neural Network (ANN).
The model is trained on a breast cancer dataset and goes through several steps including data preprocessing, visualization, training, and evaluation.
📂 Dataset
The dataset used in this project contains features extracted from breast mass images.
Each sample includes measurements such as:
radius
texture
perimeter
area
smoothness
compactness
concavity
symmetry
The target column diagnosis contains:
M = Malignant (1) → Cancerous tumor
B = Benign (0) → Non-cancerous tumor
⚙️ Project Workflow
1️⃣ Import Libraries
The project uses the following Python libraries:
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
TensorFlow / Keras
2️⃣ Data Loading
The dataset is loaded using Pandas:
Python
data = pd.read_csv("breast-cancer.csv")
3️⃣ Data Preprocessing
Several preprocessing steps are performed:
Removing unnecessary columns (e.g., id)
Converting diagnosis labels:
Label
Value
M
1
B
0
Python
data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
4️⃣ Data Visualization
A correlation heatmap is generated to understand relationships between features.
Python
sns.heatmap(corr, annot=True)
This helps identify important features affecting the diagnosis.
5️⃣ Feature & Target Separation
Python
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
6️⃣ Train-Test Split
The dataset is divided into training and testing sets.
Python
train_test_split(X, y, test_size=0.2)
80% Training
20% Testing
7️⃣ Feature Scaling
Standardization is applied using:
Python
StandardScaler()
This improves neural network performance.
🧠 Artificial Neural Network (ANN)
The model architecture:
Layer
Neurons
Activation
Input Layer
16
ReLU
Hidden Layer
8
ReLU
Output Layer
1
Sigmoid
Python
model = Sequential()
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
🏋️ Model Training
The model is trained using:
Optimizer: Adam
Loss Function: Binary Crossentropy
Epochs: 50
Batch Size: 16
📊 Model Evaluation
The model performance is evaluated using:
Accuracy
Confusion Matrix
Classification Report
ROC Curve
Example:
Python
confusion_matrix(y_test, y_pred)
📈 Accuracy Visualization
Training and validation accuracy are plotted to monitor model learning.
🎯 Project Goal
The goal of this project is to build a model capable of accurately predicting whether a tumor is malignant or benign, helping in early breast cancer detection.
🛠 Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
TensorFlow / Keras
