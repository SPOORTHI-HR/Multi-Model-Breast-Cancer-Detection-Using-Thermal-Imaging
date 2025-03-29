# **Multi-Model AI-Based Breast Cancer Detection Using Thermal Imaging**  

## **Overview**  
This project presents an **AI-powered breast cancer detection system** using a **multi-model approach** with **thermal imaging** to provide a **non-invasive, radiation-free, and comfortable** alternative to traditional mammography.  

The system performs **two-stage classification**:  
1. **Model 1:** Classifies an input thermal image as **NORMAL or SICK** (General breast health classification).  
2. **Model 2:** If classified as **SICK**, the second model further classifies it into **Benign or Malignant**.  

The system is deployed using **Flask**, allowing users to upload thermal images and receive predictions.  

## **Motivation**  
Breast cancer is the **leading cause of cancer-related deaths among Indian women**, with **late detection being a major issue**. Many women hesitate to undergo traditional screening due to discomfort, fear, and social stigma.  

This model provides an **accessible, AI-driven alternative**, encouraging early detection and consultation without exposure to radiation.  

## **Multi-Model Approach (Multimodeling)**  
This project follows a **multi-model pipeline** to enhance classification accuracy:  

1. **First Model (RGB-Based Classification)**  
   - Takes a **thermal RGB image** as input.  
   - **Classifies it as NORMAL or SICK.**  
   - If **NORMAL**, no further processing is needed.  

2. **Second Model (Grayscale-Based Classification for SICK Cases)**  
   - Converts the thermal image to **grayscale**.  
   - **Enhances features** by repeating the grayscale channel to match the RGB model.  
   - **Classifies the abnormality as Benign or Malignant.**  

This two-step **multimodeling** process ensures a **more reliable classification**, improving early detection efficiency.  

## **Dataset**  
- **Source:** Mendeley (Thermal breast cancer image dataset)  
- **Classes:**  
  - **NORMAL** (Healthy cases)  
  - **SICK - Benign**  
  - **SICK - Malignant**  

## **Model Architecture**  
This project utilizes **ResNet18**, a pre-trained deep learning model, for classification.  

### **Pipeline:**  
1. **Step 1:** Classifies an input thermal image as **NORMAL or SICK** using an **RGB-trained model**.  
2. **Step 2:** If classified as **SICK**, the image is **converted to grayscale**, repeated as a 3-channel input, and passed to the **second model** for **Benign or Malignant classification**.  

## **Technologies Used**  
- **Deep Learning:** PyTorch, ResNet18  
- **Image Processing:** OpenCV, Pillow  
- **Web Framework:** Flask  
- **Deployment:** Local server for testing  

## **Installation & Usage**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/breast-cancer-detection.git
cd breast-cancer-detection
```  

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```  

### **3. Run the Flask App**  
```bash
python BCD.py
```  

### **4. Use the Web Interface**  
- Open **http://127.0.0.1:5000/** in your browser.  
- Upload a **thermal breast image**.  
- Get predictions instantly.  


## **How This Helps Women**  
✔ **Multi-Model Precision:** Uses two specialized models to improve accuracy.  
✔ **Non-Invasive & Comfortable:** **Thermal imaging** replaces painful mammograms.  
✔ **Early Detection:** Encourages women to **screen themselves** without fear.  
✔ **AI-Powered Analysis:** Provides **fast and radiation-free** predictions.  
✔ **Bridges the Early Detection Gap:** Helps in **awareness and accessibility**, especially in India.  

## **Contributing**  
Feel free to contribute by improving the model, optimizing performance, or enhancing the web UI!
 

