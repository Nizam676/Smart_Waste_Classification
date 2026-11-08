# Smart Waste Classification and Recycling Recommendation System Using CNN

## Project Overview

The **Smart Waste Classification and Recycling Recommendation System** is a Deep Learning-based project that automatically identifies different types of waste from uploaded images.

The system uses Convolutional Neural Networks (CNN) and Transfer Learning with MobileNetV2 to classify waste into six categories and provide appropriate recycling recommendations.

The project aims to encourage proper waste segregation and promote environmental sustainability through intelligent image classification.

## Objectives

- Automatically classify waste using image processing and Deep Learning.
- Identify six different categories of waste.
- Compare the performance of Custom CNN and MobileNetV2.
- Display prediction results and confidence scores.
- Provide recycling and disposal recommendations.

## Waste Categories

The system classifies waste into the following six categories:

1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic
6. Trash

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep Learning framework |
| Keras | Model development and training |
| Custom CNN | Waste classification model |
| MobileNetV2 | Transfer Learning model |
| Flask | Web application backend |
| HTML | Webpage structure |
| CSS | Webpage styling |
| JavaScript | Frontend interaction |
| Pillow | Image processing |
| Matplotlib | Data visualization |
| Scikit-learn | Model evaluation |

## Dataset

The project uses the **TrashNet dataset**, containing 2,527 images distributed across six waste categories.

| Category | Number of Images |
|---|---:|
| Cardboard | 403 |
| Glass | 501 |
| Metal | 410 |
| Paper | 594 |
| Plastic | 482 |
| Trash | 137 |
| **Total** | **2,527** |

## Deep Learning Models

Two Deep Learning approaches were developed and compared:

### 1. Custom CNN

A Convolutional Neural Network developed from scratch using convolutional layers, max-pooling layers, flattening, dense layers and dropout.

### 2. MobileNetV2

A transfer learning approach using the pretrained MobileNetV2 architecture, with a custom classification layer for the six waste categories.

### Model Performance Comparison

| Model | Validation Accuracy |
|---|---:|
| Custom CNN | 61.58% |
| MobileNetV2 | 69.11% |

**Best Performing Model:** MobileNetV2

MobileNetV2 achieved better validation accuracy in the training comparison.

## Project Features

- Image upload functionality.
- Automatic waste classification.
- Prediction confidence score.
- Recycling recommendations.
- User-friendly web interface.
- Comparison of two Deep Learning models.

## Project Structure

```text
DL_Waste_Classification/
│
├── app.py / main.py
├── model.py
├── predictor.py
├── train.py
├── check_dataset.py
│
├── templates/
│
├── static/
│
├── models/
│   └── (Trained model files)
│
├── dataset/
│   └── (TrashNet dataset)
│
├── results/
│
├── requirements.txt
└── README.md
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd DL_Waste_Classification
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the trained model

Download the required trained model from the Google Drive link provided below and place it inside the `models` folder.

Ensure the filename matches the one expected by the application.

### 7. Run the application

```bash
python main.py
```

Open the local URL displayed in the terminal to access the application.

## Trained Model Downloads

The trained model files are hosted on Google Drive because of GitHub's individual file upload size limitation.

**Custom CNN Model**

[Download Custom CNN Model](https://drive.google.com/file/d/1yaFOW-9NSCNLrltr11jeAH7a1eaEgVfn/view?usp=drive_link)

**MobileNetV2 Model**

[Download MobileNetV2 Model](https://drive.google.com/file/d/1732vdYK0u2WtHZ-VuMxjV__t9_2rzXwc/view?usp=drive_link)

*Note: Replace the placeholder links above with the actual Google Drive sharing links.*

## Future Enhancements

- Improve classification accuracy.
- Expand the number of waste categories.
- Integrate real-time camera-based waste detection.
- Develop a mobile application.
- Improve recycling recommendations.

## Conclusion

This project demonstrates the application of Deep Learning and Computer Vision in solving a real-world environmental problem.

By comparing a Custom CNN model with MobileNetV2, the project explores different approaches to image classification and demonstrates how intelligent systems can support effective waste segregation and recycling.

---

**Project:** Smart Waste Classification and Recycling Recommendation System Using CNN  
**Domain:** Deep Learning | Computer Vision | Environmental Sustainability
