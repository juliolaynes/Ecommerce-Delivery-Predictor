# E-Commerce Predictive Logistics: Ingesting Relational Data to Model Delivery Delays

An end-to-end data science project utilizing a relational dataset of **110,000+ real Brazilian e-commerce orders** (Olist) to build a robust data pipeline and predict logistics bottlenecks.

---

## 🎯 Business Problem & Objective
In e-commerce, delivery delays directly damage customer trust, increase churn, and inflate customer support costs. 
* **The Situation:** This store faces a base delivery delay rate of **6.59%**.
* **The Goal:** Build a pipeline to extract data spread across 5 relational tables, clean the records, and deploy a machine learning classifier capable of flagging high-risk shipments before they leave the warehouse, allowing operations to proactively mitigate delays.

---

## 🛠️ Tech Stack & Architecture
* **Data Extraction:** SQL (via DuckDB)
* **Data Wrangling & Analytics:** Python, Pandas, NumPy
* **Data Visualization:** Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)

---

## 📂 Project Structure & Workflow

The workflow represents a realistic enterprise data pipeline, broken down into sequential steps:

### 🔹 Phase 1: Relational Data Extraction (SQL)
Instead of simply importing a single flat file, data was extracted directly from 5 core relational tables (`orders`, `customers`, `items`, `products`, `sellers`) using a optimized ANSI SQL script via **DuckDB** to build the initial operational, financial, and geographical master dataset.

### 🔹 Phase 2: Missing Values & Advanced Feature Engineering
* **Data Cleaning:** Imputed physical variables (weight/dimensions) using medians and filtered out non-delivered status anomalies.
* **Feature Engineering:** 
  * `is_late` (**Target Variable**): Binary marker calculated via actual delivery dates vs. courier estimates.
  * `is_same_state`: Spatial proxy mapping origin vs. target destination (Local vs. Interstate shipping).
  * `product_volume_cm3`: Calculated physical dimensional constraints.
  * `freight_ratio`: Proportional cost of shipping relative to product price.
  * `log_weight`, `log_volume`, `log_freight`: Applied logarithmic transformations to smooth highly skewed distributions and penalize heavy outliers.

---

## 📊 Key Analytical Insight (EDA)

Exploratory Data Analysis revealed that **Interstate Shipping** dramatically inflates delivery bottlenecks. Interstate orders have a **7.40%** delay rate compared to only **2.43%** for shipments bound to the same local state, confirming geographical scope as a critical predictor for the model.

---

## 🤖 Machine Learning Performance

The initial model baseline was implemented using a linear approach (`LogisticRegression` achieving a 0.5844 ROC-AUC). Due to non-linear complexities in logistical variables, the architecture was evolved to a **Random Forest Classifier** with balanced class weights to compensate for severe target imbalance.

### Final Results Comparison:

| Model Architecture | Accuracy | ROC-AUC Score | Target Class (Delayed) Recall |
| :--- | :---: | :---: | :---: |
| Baseline (Logistic Regression) | 40% | 0.5844 | **76%** |
| **Optimized Random Forest** | **52%** | **0.6275** | 66% |

**Strategic Trade-off:** By deploying the non-linear Random Forest model, the framework achieved a significantly better balance. It mitigated false alarms (boosting overall accuracy by 12% and ROC-AUC to 0.6275) while still maintaining a robust **66% recall rate** on capturing delayed shipments.

---

## 📈 Feature Importance (What Drives Delays?)

The tree-based classification algorithm revealed that physical and financial features dominate logistics friction:
1. `log_freight` (Shipping cost distribution) — *Highest impact*
2. `freight_ratio` (Freight price proportional to product value)
3. `price` (Base product cost)
4. `log_volume` (Package dimensions)

---

## 🚀 How to Reproduce
1. Clone this repository.
2. Download the Olist dataset from Kaggle and place the CSVs inside the `data/` folder.
3. Install dependencies: `pip install pandas numpy duckdb scikit-learn seaborn matplotlib`
4. Run the Jupyter Notebook in your environment.
