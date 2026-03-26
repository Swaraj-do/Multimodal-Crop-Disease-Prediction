# 🌾 Multimodal Crop Disease Prediction Using Image and Environmental Data

This project presents a multimodal machine learning framework for crop disease prediction by combining leaf image analysis with simulated environmental parameters such as temperature, humidity, rainfall, wind speed, and seasonal variation. The system not only predicts crop disease classes but also provides disease probability, environmental influence score, risk level estimation, and rule-based management recommendations, making it a simple agricultural decision-support prototype.

---

## 📌 Features

- Image-only crop disease classification using MobileNetV2
- Multimodal disease prediction using image + environmental data
- Environmental influence scoring
- Risk level estimation (Low / Medium / High)
- Rule-based recommendation generation
- Saved confusion matrices and training history plots
- Terminal-based prediction pipeline for inference

---

## 🧠 Method Overview

The proposed system uses a dual-branch multimodal architecture. The first branch processes crop leaf images using a pretrained MobileNetV2 convolutional neural network to extract visual features. The second branch processes environmental parameters including temperature, humidity, rainfall, wind speed, and season code using dense neural layers. The extracted image and environmental features are fused through concatenation and passed to fully connected layers for final disease classification. Based on prediction probability and environmental influence score, the system further estimates disease risk and provides management recommendations.

---

## 🏗 Architecture

- **Image Branch:** Leaf Image → MobileNetV2 → Global Average Pooling → Dense Layer
- **Environmental Branch:** Temperature, Humidity, Rainfall, Wind Speed, Season → Dense Layers
- **Fusion Layer:** Concatenation of image and environmental features
- **Output Layer:** Softmax classification for crop disease prediction
- **Post-Processing:** Environmental influence score + Risk estimation + Recommendations

---

## 📊 Results Comparison

| Model | Inputs | Validation Accuracy |
|------|--------|---------------------|
| Image-only | Leaf image | ~97.2% – 97.3% |
| Multimodal | Leaf image + environmental data | **~97.5%** |

The multimodal model achieved a slight improvement over the image-only baseline while also enabling environmental reasoning, risk estimation, and recommendation support.

---

## 📈 Saved Results

### Image-only model artifacts
- `results/image_only_confusion_matrix.png`
- `results/image_only_results.txt`

### Multimodal model artifacts
- `results/multimodal_confusion_matrix.png`
- `results/multimodal_results.txt`
- `results/multimodal_training_history_acc.png`
- `results/multimodal_training_history_loss.png`

### Sample prediction artifacts
- `results/sample_prediction_output.txt`
- `results/sample_prediction_output.png`

---

## 🖥 Sample Prediction Output

Example output from the terminal inference pipeline:

- Predicted Disease: `Tomato___Late_blight`
- Disease Probability: `100.00%`
- Environmental Influence: `0.672`
- Risk Level: `HIGH`

The system also returns top predictions and rule-based recommendations such as improving airflow, reducing leaf wetness, and consulting agricultural guidelines for suitable treatment options.

---

## 📂 Dataset

The project uses the **PlantVillage** dataset for crop leaf images. Since PlantVillage does not contain real environmental metadata, simulated environmental parameters were attached to each image sample to create a multimodal dataset for research experimentation.

Environmental features used:
- Temperature
- Humidity
- Rainfall
- Wind Speed
- Season Code

---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## ⚠️ Limitations

- Environmental data is simulated rather than collected from field sensors or weather APIs.
- Soil moisture and soil chemistry parameters are not included.
- PlantVillage is a clean benchmark dataset and does not fully represent real-field image noise and variability.
- Recommendations are rule-based and generalized, not crop-location specific.

---

## 🚀 Future Work

- Incorporate real environmental and sensor data
- Include soil moisture and soil health parameters
- Build a web or mobile deployment interface
- Add explainable AI visualization for image regions
- Extend to real-field disease datasets

---

## 📜 Research Relevance

This project demonstrates how multimodal learning can improve crop disease prediction systems by combining visual and environmental cues. In addition to classification, the framework supports risk-aware and recommendation-oriented outputs, making it suitable as a prototype for smart agriculture decision-support systems.
