#Gastrointestinal Cancer Detection & Classification Using Deep Learning

## 📌 Project Overview

Gastrointestinal Cancer Detection & Classification is a deep learning-based system designed to classify gastrointestinal tract images into **cancerous and non-cancerous categories**.

The project uses **transfer learning with MobileNetV2** to automatically learn important visual features from gastrointestinal images and perform image classification. The system includes image preprocessing, data augmentation, model training, evaluation, and prediction.

## 🎯 Objectives

* Develop an automated gastrointestinal cancer image classification system.
* Classify gastrointestinal images into cancerous and non-cancerous categories.
* Apply deep learning and transfer learning for medical image analysis.
* Improve model performance using image preprocessing and augmentation.
* Evaluate the model using accuracy, precision, recall, and F1-score.

## 🧠 Methodology

The project follows these major steps:

1. **Dataset Collection**

   * Gastrointestinal images are used for model development and evaluation.

2. **Image Preprocessing**

   * Image resizing
   * Pixel normalization
   * Data augmentation
   * Regularization

3. **Model Development**

   * MobileNetV2 is used as the transfer learning backbone.
   * The pretrained network extracts relevant image features.
   * Additional classification layers are used to classify the images.

4. **Model Training**

   * The model is trained using the prepared gastrointestinal image dataset.
   * Training and validation performance are monitored using accuracy and loss.

5. **Model Evaluation**

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * Confusion matrix

6. **Prediction**

   * The trained model can be used to classify new gastrointestinal images.

## 🛠️ Technologies Used

* **Python**
* **TensorFlow / Keras**
* **MobileNetV2**
* **NumPy**
* **OpenCV**
* **Matplotlib**
* **Deep Learning**
* **Transfer Learning**

## 📊 Model Performance

The developed model achieved the following evaluation results:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 80.14% |
| Precision | 80.07% |
| Recall    | 80.14% |
| F1-Score  | 79.87% |

These results demonstrate the model's ability to distinguish between cancerous and non-cancerous gastrointestinal images.

## 📁 Project Structure

```text
Gastrointestinal-Cancer-Detection-and-Classificatioon-using-Deep-Learning/
│
├── Gastrovsion/
│   ├── app.py
│   ├── train_model.py
│   ├── metadata.json
│   ├── results.png
│   ├── final_model.keras
│   ├── final_model_finetuned.keras
│   ├── fold_1_best.keras
│   ├── fold_2_best.keras
│   ├── fold_3_best.keras
│   └── gi_cancer_final.keras
│
└── README.md
```

> The original dataset is not included in this repository because of its large file size.

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/SaiHariB/Gastrointestinal-Cancer-Detection-and-Classificatioon-using-Deep-Learning.git
```

### 2. Navigate to the project

```bash
cd Gastrointestinal-Cancer-Detection-and-Classificatioon-using-Deep-Learning
```

### 3. Install required dependencies

Install the required Python libraries according to the project configuration.

### 4. Run the application

```bash
python Gastrovsion/app.py
```

## 📌 Important Note

The gastrointestinal image dataset is **not included in this repository** because of GitHub file-size limitations.

The trained model files and project source code are included for demonstration and development purposes.

## 🔬 Future Enhancements

* Improve classification accuracy with larger and more diverse datasets.
* Experiment with additional transfer learning architectures such as ResNet and DenseNet.
* Integrate Grad-CAM for visual explanation of model predictions.
* Develop a more user-friendly web interface.
* Deploy the trained model as a cloud-based medical image analysis application.

## 👨‍💻 Author

**Sai Hari B**

B.E. Computer Science Engineering
Sathyabama Institute of Science and Technology

## ⚠️ Disclaimer

This project is developed for **educational and research purposes**. It is not intended to replace professional medical diagnosis or clinical decision-making.
