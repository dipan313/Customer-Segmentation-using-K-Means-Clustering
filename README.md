# PRODIGY_ML_02
Customer Segmentation using K-Means Clustering
📌 Project Overview
This project applies K-Means Clustering to segment customers of a retail store based on their purchase behavior. By analyzing spending patterns and annual income, we can group customers into meaningful clusters, helping businesses personalize marketing strategies and improve customer engagement.

🚀 Features
Data Preprocessing: Standardizes the dataset for better clustering results.

Optimal Cluster Selection: Uses the Elbow Method to determine the best number of clusters.

K-Means Clustering: Groups customers based on their spending score and income.

Visualization: Generates plots to understand customer segmentation.

Clustered Data Export: Saves the results for further analysis.

📂 Project Structure
ml_task2/
│── data/
│   ├── Mall_Customers.csv          # Raw dataset
│   ├── clustered_customers.csv     # Processed dataset with cluster labels
│── notebooks/
│   ├── kmeans_customer_segmentation.ipynb  # Google Colab notebook
│── src/
│   ├── kmeans_clustering.py        # Python script for clustering
│── images/
│   ├── elbow_method.png            # Elbow method plot
│   ├── customer_clusters.png       # Cluster visualization
│── README.md                       # Project documentation
│── requirements.txt                 # Required libraries
│── .gitignore                       # Ignore unnecessary files
🔧 Setup & Installation
Prerequisites
Ensure you have Python 3.x installed along with the required libraries.

Install Dependencies
Run the following command to install the required libraries:

pip install -r requirements.txt
Run the Script
Execute the following command to perform clustering:

python src/kmeans_clustering.py
This will generate clustered results and visualizations in the images/ folder.

📊 Understanding the Results
The Elbow Method plot helps identify the optimal number of clusters.

The Customer Segmentation plot shows grouped customers based on their spending and income.

The Processed dataset includes the cluster labels for each customer.

📜 Dataset Information
Annual Income (k$): Customer’s yearly income.

Spending Score (1-100): Customer’s spending behavior score assigned by the store.

🤝 Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

📜 License
This project is open-source and available under the MIT License.

🔍 Need Help? Feel free to reach out or open an issue. Happy coding! 🚀


