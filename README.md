# 🧠 AI Task Management Dashboard

An interactive web dashboard built with **Streamlit** to:
- Explore and visualize task data (EDA, boxplots, wordcloud)
- Perform NLP preprocessing on task descriptions
- Train and use an ML model to **predict task priority**
- Upload/download processed data

🌱 **Live App**: [👉 Click here to open the dashboard](https://ai-task-management-dashboard.streamlit.app/)

---

## ✨ **Features**
✅ Data overview and summary statistics  
✅ Visualizations:
- Completion time vs priority (boxplot)
- User sentiment distribution (countplot)
- Progress distribution (histogram)
- Word cloud of task descriptions
✅ NLP text preprocessing (stopwords removal, lemmatization using spaCy)
✅ Train & save ML model (TF-IDF + RandomForest)
✅ Live prediction: enter new task description → get predicted priority
✅ Upload your own CSV and explore it
✅ Download processed dataset

---

## 📦 **Project Structure**
├── app.py # Streamlit app
├── notebook.ipynb # Data analysis & model training
├── requirements.txt # Python dependencies
├── data/
│ └── processed_task_dataset.csv # Cleaned dataset
├── models/
│ └── priority_model.pkl # Saved trained model
├── completion_time_vs_priority.png # Plot images used in app
├── progress_distribution.png
├── user_sentiment_distribution.png
└── wordcloud.png
