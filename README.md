# 🚀 DeepFake Detection API

A **DeepFake Detection Web App** that classifies images as **Real or Fake** using a deep learning model.\
Deployed on **Render** with an **interactive web interface**.

---

## 📌 Features

✅ **Upload an image** through a simple web UI\
✅ **Get instant predictions** (Real or Fake) with confidence score\
✅ **Powered by PyTorch-based DeepFake detection model**\
✅ **API endpoint available** for programmatic access\
✅ **Live Deployment on Render**

---

## ⚙️ Installation

1️⃣ **Clone the Repository**

```sh
git clone https://github.com/anand7as/dfdc_project_7as.git
cd dfdc_project_7as
```

2️⃣ **Install Dependencies**

```sh
pip install -r requirements.txt
```

3️⃣ **Run the Application**

```sh
python app.py
```

🔗 **Access the Web UI:** [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🎯 API Usage

📍 **Endpoint:** `POST /predict`\
📤 **Request:** Send an image file using `multipart/form-data`

```sh
curl -X POST -F "image=@test_image.jpg" "https://dfdc-project-7as.onrender.com/predict"
```

📥 **Response Example:**

```json
{ "prediction": "Fake", "confidence": 92.5 }
```

---

## 🚀 Live Demo

🔗 **Try it now:** [DeepFake Detector](https://dfdc-project-7as.onrender.com/)

---

## 🔄 Deployment & Updates

To update the project and trigger redeployment on Render:

```sh
git add .
git commit -m "Updated app"
git push origin main
```

---

## 🖼️ UI Preview

(Add a screenshot or GIF of your web UI here!)

---

## 🤝 Contributing

Want to improve this project? Feel free to **open a PR** or suggest new features!

---

## ⚖️ License

This project is **open-source** under the **MIT License**.

