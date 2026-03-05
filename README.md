# 🧠 EmotiSense AI

### Multi-Class Text Emotion Detection System

EmotiSense AI is a Natural Language Processing (NLP) based machine learning application that analyzes textual input and predicts the underlying emotion expressed in the text. The system classifies emotions into multiple categories such as **joy, sadness, anger, fear, love, and surprise** while also providing probability scores, confidence levels, and sentiment comparison.

This project demonstrates the use of **TF-IDF feature extraction and Logistic Regression** for multi-class text classification, along with an interactive **Streamlit web application** for real-time emotion detection and visualization.

---

## 🚀 Features

* 📝 Detects emotions from user-input text

* 😊 Supports **6 emotion classes**

  * Joy
  * Sadness
  * Anger
  * Fear
  * Love
  * Surprise

* 📊 Displays **probability distribution** for each emotion

* 🟢🟡🔴 Shows **confidence level** of predictions

* 💬 Performs **sentiment analysis** (positive / negative / neutral)

* 📈 Interactive **visualization charts** using Plotly

* 🌐 User-friendly **Streamlit web interface**

---

## 🛠️ Technologies Used

* **Python**
* **Natural Language Processing (NLP)**
* **Scikit-learn**
* **TF-IDF Vectorization**
* **Logistic Regression**
* **Streamlit**
* **Plotly**
* **Pandas & NumPy**

---

## 📂 Project Structure

```
HexSoftwares_EmotiSenseAI
│
├── app.py
├── train_model.py
├── predict.py
├── download_dataset.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
│
├── models/
│   └── emotion_pipeline.pkl
│
└── assets/
```

---

## ⚡ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/HexSoftwares_EmotiSenseAI.git
cd HexSoftwares_EmotiSenseAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📥 Download Dataset

Run the dataset download script:

```bash
python download_dataset.py
```

This downloads the **dair-ai/emotion dataset** used for training the model.

---

## 🧠 Train the Model

```bash
python train_model.py
```

This will:

* Train the TF-IDF + Logistic Regression model
* Generate evaluation metrics
* Save the trained model in the **models/** folder

---

## 🌐 Run the Web Application

```bash
streamlit run app.py
```

Open the link shown in the terminal (usually):

```
http://localhost:8501
```

Then enter text to detect emotions.

---

## 🎯 Example

**Input**

```
I am extremely excited about this new opportunity!
```

**Output**

```
Emotion: Joy 😄
Confidence: High
Sentiment: Positive
```

The application also displays a **probability chart** showing the likelihood of each emotion.

---

## 📊 Dataset

This project uses the **dair-ai/emotion dataset** from HuggingFace which contains thousands of labeled text samples for emotion classification.

Emotion classes include:

* joy
* sadness
* anger
* fear
* love
* surprise

---

## 📌 Future Improvements

* Deep Learning models (LSTM / Transformers)
* Emotion detection from speech
* Real-time social media emotion analysis
* API deployment for integration with other applications

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

## 👨‍💻 Author

**Pranav Anand**

AI / Machine Learning Enthusiast
MSc Artificial Intelligence, Machine Learning & Data Science

---

⭐ If you like this project, feel free to **star the repository**.
