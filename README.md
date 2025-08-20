# Mango Leaf Disease Classifier 🥭🩺

A **deep learning–powered web app** that identifies **mango leaf diseases** from images using a trained **TensorFlow model** and an interactive **Streamlit UI**.

---

## ✨ Features

* 🔬 **Disease Detection**: Classifies mango leaf images into 6 categories:

  * Anthracnose
  * Bacterial Canker
  * Cutting Weevil
  * Die Back
  * Gall Midge
  * Healthy

* 📸 **Image Upload**: Upload `.jpg` or `.png` images of mango leaves.

* ⚡ **Real-Time Prediction**: Get instant classification with confidence scores.

* 🎨 **User-Friendly UI**: Built with Streamlit for simplicity and accessibility.

* 🧠 **Pretrained Model**: Loads a TensorFlow model (`.h5`) for accurate inference.

---

## 🚀 How It Works

### 🔐 Classification Flow

```
Upload Image → Preprocessing → TensorFlow Model → Predicted Class + Confidence
```

### 🧾 Example Output

> ✅ The image is classified as **Anthracnose** with 92.15% confidence.

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/MangoDiseaseClassification.git
cd MangoDiseaseClassification
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 🎯 Use Cases

* Helping **farmers** and **agriculture researchers** detect mango leaf diseases early.
* Demonstrating the power of **deep learning + computer vision** in agriculture.
* Educational showcase for **TensorFlow model deployment with Streamlit**.

---

## 📸 Screenshots

![Image](https://github.com/user-attachments/assets/3f4d417f-197a-4ea2-9281-07a1486bd96e)
*Upload Mango Image*

![Image](https://github.com/user-attachments/assets/a30fb93f-4931-4382-ac11-bde2b6d1d88a)
*Choose Mango Image*

![Image](https://github.com/user-attachments/assets/14d4d3ca-c089-48a5-8166-69ed5d339d0e)
*Result of the Mango Disease Detection*

---

## ⚠️ Disclaimer

This project is for **educational and experimental purposes**.
While the model performs well on sample data, it should **not replace expert agricultural advice** for critical decision-making.

---

## 🏷️ Topics

`python` `machine-learning` `deep-learning` `computer-vision` `tensorflow` `keras` `neural-networks` `streamlit` `image-classification` `cnn`
