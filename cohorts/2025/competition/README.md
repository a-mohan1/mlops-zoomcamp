# MLOps Zoomcamp 2025 Competition

Welcome to the MLOps Zoomcamp 2025 Competition!

We're hosting this competition to help you practice everything you learn in the course.

Unlike traditional Kaggle competitions that focus only on model accuracy, this competition emphasizes the complete MLOps lifecycle, from model development to deployment and monitoring.

## Quick Links
- **Competition Platform**: https://youare.bot/ 
  - Register your classifier in the "Register your API" section
  - View leaderboards and track your performance
- [Detailed Overview (Slides)](https://docs.google.com/presentation/d/1-DIRH8wh-Rc251oZLjFNzQF0Fm0yUYe9PmLm00b3SyM/edit?usp=sharing)
- [Competition Overview Video](https://www.youtube.com/watch?v=ZxUVBG4z5uE)

## Competition Overview

### What You'll Build


**Classifier**: Build an ML model that detects whether a chat participant is a bot or human

### Scoring

**Classifiers**: Evaluated on both:
  - ML metrics: accuracy, ROC, precision, recall, F-score
  - Technical metrics: latency, uptime, response rate

### Timeline
- Competition period: June 2025-August 2025
- Final submission deadline: September 1st, 2025
- Evaluation: Best consecutive 7-day performance
  - Deploy anytime during the competition period
  - No need to keep your service running the entire time

### Benefits
- Practical MLOps experience in a real-world scenario
- Portfolio-worthy project showcasing end-to-end ML system
- Course points for participation
- $500 of AWS vouchers divided equally among top 10 participants

## Implementation Guide

Start with these core components:
- FastAPI/Flask for serving your model
- PostgreSQL for message history (optional)
- MLflow for experiment tracking
- Prometheus/Grafana for monitoring

As you advance, consider adding:
- CI/CD pipeline for automated deployment
- Evidently for model quality monitoring
- Airflow for automated retraining
- Cloud deployment (AWS, Fly.io, etc.)

### Getting Started
1. **Example Implementations**:
   - [YouAreBot Quickstart Template](https://github.com/open-cu/youarebot-quickstart): Simple Echo Bot with FastAPI endpoint
   - [YouAreBot Bot Example](https://github.com/open-cu/youarebot-bot): Advanced GPT-based bot (requires OpenAI/DeepSeek API key and additional setup)
   - [YouAreBot Classifier Example](https://github.com/open-cu/youarebot-classifier): FastAPI service with PostgreSQL storage and random classifier (baseline for you to improve)

2. **Deployment Options**:
   - Testing: Use quickstart repo with SSH tunneling
   - Production: Deploy to any cloud provider
     - AWS (recommended, covered in course)
     - Fly.io
     - Your preferred cloud platform

## Support
For technical questions, contact Misha:
- Telegram: https://t.me/bearcolonel
- Email: mikhail.semchinov@gmail.com
