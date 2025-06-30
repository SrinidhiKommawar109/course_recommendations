🎓 Course Recommendation System

This is a simple yet effective **Course Recommendation System** built using **Streamlit** and **scikit-learn**. It helps users find Udemy courses based on a keyword, available time, course type (paid or free), and difficulty level.

🚀 Features

- Recommends Udemy courses based on a keyword
- Filters by:
  - Available learning time
  - Free or paid courses
  - Course difficulty level
- Beautiful background image using custom CSS
- Outputs clickable course links
🧠 How It Works
1. **TF-IDF** vectorization is used on the course titles.
2. A **cosine similarity** score is computed between the input keyword and the course titles.
3. Filters (duration, paid/free, level) are applied to shortlist relevant courses.
4. Top 10 recommendations are displayed with links.
 📁 Dataset

The system expects a dataset named `udemy_course_data.csv` with at least the following columns:
- `course_title`
- `content_duration`
- `is_paid`
- `level`
- `url`
- `published_time` (optional)

## 📦 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/course-recommender.git
    cd course-recommender
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
 🧾 Requirements

streamlit
pandas
scikit-learn

