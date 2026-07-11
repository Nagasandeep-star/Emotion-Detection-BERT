# 😊 Emotion Detection Using BERT

A Machine Learning and Natural Language Processing (NLP) project that detects human emotions from text using the **BERT (Bidirectional Encoder Representations from Transformers)** model. This application is built using **PyTorch**, **Hugging Face Transformers**, and **Flask**, providing a simple and interactive web interface for real-time emotion prediction.

---

## 🚀 Features

- 🔍 Detects emotions from user input text
- 🤖 Fine-tuned BERT model for emotion classification
- 📊 Displays confidence score for predictions
- 🌐 Flask-based web application
- 🎨 Responsive and user-friendly interface
- ⚡ Real-time emotion prediction

---

## 🧠 Supported Emotions

- 😊 Joy
- 😢 Sadness
- ❤️ Love
- 😠 Anger
- 😨 Fear
- 😲 Surprise

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Flask
- HTML5
- CSS3
- JavaScript
- Pandas
- Scikit-learn

---

## 📂 Project Structure

```text
Emotion-Detection-BERT/
│
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── data/
│   ├── raw/
│   └── processed/
│
├── inference/
│   └── predictor.py
│
├── model/
│   ├── bert_model.py
│   ├── dataset.py
│   ├── dataloader.py
│   └── tokenizer.py
│
├── preprocessing/
│
├── saved_model/
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
└── tests/
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Nagasandeep-star/Emotion-Detection-BERT.git
```

### 2️⃣ Navigate to the Project

```bash
cd Emotion-Detection-BERT
```

### 3️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

### 4️⃣ Activate the Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

### 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 6️⃣ Run the Application

```bash
python app.py
```

### 7️⃣ Open in Your Browser

```text
http://127.0.0.1:5000
```

---

## 📸 Application Preview

Enter a sentence like:

> **I am very happy today.**

The application predicts the emotion and displays the confidence score.

Example Output:

```text
😊 Joy

Confidence: 98.42%
```

---

## 🎯 Future Enhancements

- 🌙 Dark Mode
- 📈 Emotion Probability Visualization
- 📝 Prediction History
- 🔗 REST API
- 🐳 Docker Support
- ☁️ Cloud Deployment (Render/AWS)
- 📱 Mobile Responsive UI

---

## 📚 Learning Outcomes

This project demonstrates:

- Data preprocessing for NLP
- Text tokenization using BERT Tokenizer
- Fine-tuning BERT for emotion classification
- PyTorch Dataset and DataLoader
- Model training and evaluation
- Flask web application development
- Frontend integration using HTML, CSS, and JavaScript
- Git and GitHub version control

---

## 👨‍💻 Authors

### Thiruveedula Naga Sandeep

- 📧 Email: nagasandeepthiru@gmail.com
- 🐙 GitHub: https://github.com/Nagasandeep-star
- 💼 LinkedIn: https://www.linkedin.com/in/thiruveedula-naga-sandeep

---

### Pagadala Ishwarya

- 📧 Email: ishwaryapagadala131@gmail.com
- 🐙 GitHub: https://github.com/Ishwaryapagadala
- 💼 LinkedIn: https://www.linkedin.com/in/pagadala-ishwarya-7a83b4331

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push your branch.
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the **MIT License**.

---

⭐ **If you found this project helpful, please consider giving it a Star on GitHub!**