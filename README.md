# 📈 Corporate Profit Prediction API & ML Pipeline

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Random Forest](https://img.shields.io/badge/Model-Random_Forest-success?style=for-the-badge)

## 📖 Project Overview
The **Corporate Profit Prediction API** is an end-to-end machine learning project that estimates a startup's financial profit based on its departmental expenditures (R&D, Administration, and Marketing) and geographic location. 

This project demonstrates a rigorous data science workflow: performing automatic Exploratory Data Analysis (EDA), engineering financial ratio features, benchmarking multiple regression algorithms (Linear Regression, Random Forest, XGBoost, LightGBM) via `RandomizedSearchCV`, and deploying the winning model as a high-performance RESTful API using **FastAPI**.

## ✨ Key Features
*   **Automated EDA:** Generates correlation heatmaps and feature distribution histograms on the fly using `seaborn` and `matplotlib` to validate data integrity before training.
*   **Dynamic Feature Engineering:** Computes critical business metrics (`RD_to_Admin_ratio`, `Marketing_to_Admin_ratio`, and `Total_Spend`) with mathematical safeguards (epsilon `1e-5`) to prevent division-by-zero errors.
*   **Algorithmic Benchmarking:** Systematically compares Linear Regression against advanced tree-based models (Random Forest, XGBoost, LightGBM) using cross-validated random search for hyperparameter tuning.
*   **Synchronized Inference Logic:** The FastAPI deployment perfectly mirrors the training feature engineering, dynamically calculating the required ratios from raw JSON payloads before passing them to the `.joblib` pipeline.
*   **Strict Data Validation:** Utilizes `Pydantic` to enforce strict type-checking and validation on incoming API requests.

## 📊 Data Description
The model is trained on a dataset of 50 corporate startups (`dataframe.csv`).

**Input Features:**
*   `R&D Spend`: Research and Development expenditure.
*   `Administration`: Administrative and operational overhead.
*   `Marketing Spend`: Marketing and advertising expenditure.
*   `State`: The state where the startup operates (e.g., 'New York', 'California', 'Florida').

**Target Variable:**
*   `Profit`: The overall calculated profit of the startup.

## 🛠️ Project Architecture

```text
├── train_model.py                             # EDA, Feature Engineering, Model Training & Tuning
├── main.py                                    # FastAPI application and prediction endpoint
├── dataframe.csv                              # Startup financial dataset[Not included, download required]
├── best_model.joblib                          # Serialized Scikit-Learn pipeline (Generated Output)
└── README.md                                  # Project documentation
```

## 🚀 Installation & Prerequisites

To run this data pipeline and API locally, ensure you have Python 3.8+ installed.

1. **Clone the repository:**
   ```bash
   git clone [Insert Repository Link Here]
   cd [Insert Repository Directory Name]
   ```

2. **Install the required dependencies:**
   It is highly recommended to use a virtual Python environment.
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn fastapi uvicorn joblib pydantic
   ```

3. **Add the Dataset:**
   Ensure the `dataframe.csv` file is downloaded and placed in the root directory.

## 💻 Usage / How to Run

### Step 1: Train the Model
Run the training script to perform EDA, tune the models, and export the best-performing pipeline. 
*(Note: You will need to close the pop-up EDA plot windows for the script to continue training).*

```bash
python train_model.py
```

### Step 2: Launch the FastAPI Server
Once `best_model.joblib` is generated, start the FastAPI server using Uvicorn.

```bash
uvicorn main:app --reload
```

### Step 3: Test the API Endpoint
Navigate to `http://127.0.0.1:8000/docs` in your browser to use the interactive Swagger UI, or test the `/predict` endpoint via cURL:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "rd_spend": 165349.20,
  "administration": 136897.80,
  "marketing_spend": 471784.10,
  "state": "New York"
}'
```

## 📈 Results / Model Evaluation

During the automated benchmarking phase, the models achieved the following Mean Absolute Errors (MAE) on the test set:

*   **Linear Regression:** `6554.21`
*   **Random Forest:** `6530.12` 🏆 *(Winner)*
*   **XGBoost:** `9080.88`
*   **LightGBM:** `24699.42`

**Why did Random Forest win?**
Given the extremely small dataset size (50 samples), highly complex gradient boosting algorithms like LightGBM and XGBoost struggled with overfitting and lack of depth, whereas Random Forest (and standard Linear Regression) mapped the generalized patterns much more effectively.

**Example API Response:**
```json
{
  "prediction": 192261.83
}
```

## 🤝 Contributing
Contributions are highly encouraged! If you'd like to scale this project:
1. Fork the repository
2. Create your Feature Branch (`git checkout -b feature/DockerizeAPI`)
3. Commit your Changes (`git commit -m 'Add Dockerfile for FastAPI deployment'`)
4. Push to the Branch (`git push origin feature/DockerizeAPI`)
5. Open a Pull Request

## 📜 License
This project is open-source and available under the MIT License. See `LICENSE` for more information.

---

## 📬 Contact
**Selim Najaf**

*   **LinkedIn:** [linkedin.com/in/selimnajaf-data-analyst](https://www.linkedin.com/in/selimnajaf/)
*   **GitHub:** [github.com/SelimNajaf](https://github.com/SelimNajaf)

*Developed as a continuous learning initiative in advanced Data Science and ML Engineering.*
