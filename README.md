# Store Sales Prediction

This project uses a Random Forest Regressor to predict the sales of items at various Big Mart outlets. The model is built from a dataset of item and outlet properties and has been deployed as an interactive web app using Streamlit.

This project was developed as a data science assignment.

---

## Live Demo

You can view and interact with the deployed Streamlit app here:

**[Live App Link](https://your-streamlit-app-url.streamlit.app/)** *(You will get this URL after you deploy it on Streamlit Community Cloud)*

---

## Project Findings

The final **Random Forest Regressor** model was chosen after comparing it with a Linear Regression model. It performed better, achieving:

* **R² Score:** **0.56** (explains 56% of the variance in sales)
* **Root Mean Squared Error (RMSE):** **1090.13**

**Key Feature Engineering:**
* **`Item_Category`**: Item types were grouped into 'Food', 'Non-Consumable', and 'Drink' to create a stronger feature.
* **`Outlet_Years`**: Created from `Outlet_Establishment_Year` to represent the age of the store.
* **Imputation**: Missing `Item_Weight` was filled with the mean, and missing `Outlet_Size` was filled with the mode.

---

## How to Run This Project Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/Store-Sales-Prediction.git](https://github.com/your-username/Store-Sales-Prediction.git)
cd Store-Sales-Prediction