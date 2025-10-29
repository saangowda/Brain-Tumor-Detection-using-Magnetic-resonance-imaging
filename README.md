# 🧠 Brain Tumor Detection from MRI

This project uses **Deep Learning (CNN)** and **Flask** to detect brain tumors from MRI images.  
It helps in classifying MRI scans into **tumor** and **non-tumor** categories with high accuracy.

---

## 🚀 Features
- Upload MRI images through a simple web interface  
- Preprocessing of MRI scans (resizing, normalization, augmentation)  
- CNN-based classification using **TensorFlow/Keras**  
- Accuracy and loss visualization using **Matplotlib**  
- Flask-based web deployment for real-time prediction  

---

## 🧩 Tech Stack
- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Python, Flask  
- **Deep Learning:** TensorFlow, Keras  
- **Libraries:** NumPy, OpenCV, Matplotlib  
- **Database (optional):** SQLite/MySQL  

---

## 🧠 Model Overview
The Convolutional Neural Network (CNN) model is trained on MRI datasets containing brain tumor images.  
It performs feature extraction and classification to identify tumor presence in new MRI inputs.

Model Steps:
1. Data Preprocessing (resize, normalize)
2. CNN Model Training (Conv2D, MaxPooling, Flatten, Dense)
3. Model Evaluation (accuracy and loss metrics)
4. Flask Integration for real-time web predictions

---

## 🖥️ How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/sahana-kr/Brain-Tumor-Detection.git
Navigate to the project folder:

cd Brain-Tumor-Detection


Install dependencies:

pip install -r requirements.txt


Run the Flask app:

python app.py


Open your browser and go to:

http://127.0.0.1:5000/

📊 Results

Achieved high accuracy on validation dataset

Visualized training and validation loss curves

Simple and user-friendly interface for MRI prediction

🧾 Project Structure
Brain-Tumor-Detection/
│
├── static/              # CSS, JS, and image assets
├── templates/           # HTML templates (home, result pages)
├── model/               # Trained CNN model file (.h5)
├── app.py               # Flask web app
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
└── dataset/             # MRI dataset (not uploaded due to size)

🧑‍💻 Developer

Sahana KR
📍 Bangalore, Karnataka
📧 sanugowda87@gmail.com

🔗 LinkedIn: linkedin.com/in/sahana-kr-s42
 | GitHub :

🏅 Acknowledgment

Dataset used from Kaggle Brain MRI Images for Brain Tumor Detection.
Thanks to the open-source community for valuable libraries and resources.

⭐ If you found this project helpful, don’t forget to star the repository!

