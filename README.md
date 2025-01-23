 ## Sentiment Analysis Model - AWS Deployment
## Project Overview
This project demonstrates deploying a Sentiment Analysis model fine-tuned using DistilBERT-base-uncased, making it accessible to users via a web application built with Gradio. The deployment uses AWS services such as S3, EC2, and RDS.

## Objective
To deploy a sentiment analysis model for real-time predictions, storing user logs and results in an AWS RDS database while leveraging S3 for model storage and EC2 for hosting the application.

## Features
Sentiment Prediction: Analyze tweets and classify sentiments as Positive, Neutral, or Negative.
Web Application: Built using Gradio, with a user-friendly interface.
AWS Integration:
S3: Store the trained model and application script (App.py).
EC2: Host the application.
RDS: Log user interactions and predictions.
Security: Implements IAM roles and AWS security best practices.
## Dataset
Source: Entity-level sentiment analysis dataset of Twitter.
Classes: Positive, Neutral, Negative.
## Installation
Prerequisites
Python 3.8 or later
AWS CLI configured with appropriate permissions
An active AWS account
Install Dependencies
boto3
gradio
torch
transformers
pymysql

## How to Run the Project
1. Model Training and Deployment
Fine-tune the DistilBERT-base-uncased model using the dataset.

Save the fine-tuned model and App.py script to an S3 bucket.

2. Set Up AWS Infrastructure
S3: Upload model files and App.py script to a designated S3 bucket.

EC2: Launch an instance, assign an IAM role with S3 full access, and configure security groups (port 8501 for Gradio, 3306 for RDS).

RDS: Set up a database to store user interaction logs.

3. Deploy the Application
SSH (secure shell) into the EC2 instance.

Download the model and App.py from S3.

Run the application:
python App.py 

4. Access the Web Application
Use the generated public URL to access the application.

## Key Functionalities
Prediction API: Enter text to receive sentiment predictions and probabilities.

Logging: Automatically logs input text, predictions, and user IP addresses to the RDS database.

## Security Measures
Configured IAM roles for secure S3 and RDS access.

Security groups to control inbound traffic.

## Project Structure
├── README.md # Project overview and setup instructions

├──  Python dependencies

├── App.py # Gradio application script

├── Cleaned_Tweet_dataset.csv # Preprocessed dataset

├── model/ # Directory for the fine-tuned model files

## Evaluation Metrics
Accuracy: Measures overall prediction correctness.

Precision/Recall/F1-Score: Evaluate model performance on individual classes.

Latency: Average time per prediction.

## Acknowledgements
Hugging Face Transformers for the pre-trained model.

AWS for infrastructure services.

Gradio for the user interface.
