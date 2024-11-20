# Text Classification of BBC News Dataset

## Overview
This project is focused on classifying BBC news articles into one of the following five categories:
- `tech`
- `business`
- `sport`
- `entertainment`
- `politics`

The dataset contains 2225 articles, each labeled with a category. The project implements a pipeline that includes:
1. Data preprocessing
2. Feature extraction using TF-IDF
3. Classification using Logistic Regression

The model achieves a classification accuracy of **96.85%**, demonstrating strong performance across all categories.

---

## Project Structure
The project files are organized as follows:

|-- bbc-text.csv # Dataset file 
|-- Part2.py # Main script containing the code 
|-- confusion_matrix.png # Confusion matrix visualization 
|-- report.pdf # Detailed project report 
|-- README.md # Project documentation

---

## Setup Instructions
### Prerequisites
Make sure you have Python 3.7 or higher installed. The following Python libraries are required:
- `pandas`
- `scikit-learn`
- `matplotlib`
- `re` (standard library)

## Installation
1. Clone or download this repository.
2. Navigate to the project directory.
3. Install the required packages:
   ```bash
   pip install pandas scikit-learn matplotlib


## Running the Project
### Steps to Execute
1.Ensure that the dataset file bbc-text.csv is located in the same directory as Part2.py.
2.Run the Python script to execute the entire classification process:
	python Part2.py
3.The script will:
	Load the dataset (bbc-text.csv).
	Preprocess the text data (remove non-alphabetic characters, convert to lowercase, etc.).
	Apply TF-IDF vectorization to extract features from the text.
	Train a Logistic Regression classifier on the dataset.
	Evaluate the classifier and print the classification accuracy and detailed classification report.

## Results
### The Logistic Regression classifier achieved the following results:

	Overall Accuracy: 96.85%
	Category-wise Performance (Precision | Recall | F1-score):
		tech: 0.98 | 0.98 | 0.98
		business: 0.95 | 0.93 | 0.94
		sport: 0.98 | 1.00 | 0.99
		entertainment: 1.00 | 0.96 | 0.98
		politics: 0.94 | 0.98 | 0.96