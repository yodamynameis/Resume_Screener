# Import necessary libraries
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import pickle
import base64
import io
from PyPDF2 import PdfReader
import docx2txt


import nltk
from nltk.corpus import stopwords

# Try to load stopwords, download if missing
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Download NLTK resources
nltk.download('punkt', download_dir='nltk_data')
nltk.download('stopwords', download_dir='nltk_data')
nltk.download('wordnet', download_dir='nltk_data')

# Add this at the top of your script to tell NLTK where to look
nltk.data.path.append('nltk_data')


# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Data preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    
    # Join tokens back to string
    cleaned_text = ' '.join(cleaned_tokens)
    
    return cleaned_text

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    text = docx2txt.process(file)
    return text

# Function to extract text from resume
def extract_text_from_resume(file):
    file_extension = file.name.split('.')[-1]
    if file_extension == 'pdf':
        return extract_text_from_pdf(file)
    elif file_extension == 'docx':
        return extract_text_from_docx(file)
    elif file_extension == 'txt':
        return file.getvalue().decode('utf-8')
    else:
        return "Unsupported file format"

# Function to generate feedback based on predicted category and resume text
def generate_feedback(predicted_category, resume_text, category_skills):
    # Get required skills for the predicted category
    required_skills = category_skills.get(predicted_category, [])
    
    # Check which skills are present in the resume
    present_skills = [skill for skill in required_skills if skill.lower() in resume_text.lower()]
    missing_skills = [skill for skill in required_skills if skill.lower() not in resume_text.lower()]
    
    # Generate feedback with emojis
    feedback = f"🎯 Based on your resume, you are most suitable for the **'{predicted_category}'** role.\n\n"
    
    if present_skills:
        feedback += "💪 **Strengths:**\n"
        for skill in present_skills:
            feedback += f"✅ Good knowledge of **{skill}**\n"
    
    if missing_skills:
        feedback += "\n🛠️ **Areas for improvement:**\n"
        for skill in missing_skills:
            feedback += f"❌ Consider developing skills in **{skill}**\n"
    
    return feedback

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="💼 Resume Screnner", layout="wide")
    st.title("👔 Resume Screening and Job Role Predictor")
    st.write("Upload your resume to find the most suitable job role and get personalized feedback 🎉")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    
    # Download sample data if not present
    try:
        # Try to load the dataset
        df = pd.read_csv('UpdatedResumeDataSet.csv')
    except FileNotFoundError:
        st.write("⏳ Downloading dataset for first-time setup...")
        st.error("Please download the UpdatedResumeDataSet.csv from Kaggle and place it in the same directory as this script.")
        st.write("👉 You can find it here: [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)")
        return
    
    # Train or load model
    model_file = 'resume_classifier_model.pkl'
    vectorizer_file = 'tfidf_vectorizer.pkl'
    encoder_file = 'label_encoder.pkl'
    
    try:
        # Try to load the model and related components
        model = pickle.load(open(model_file, 'rb'))
        vectorizer = pickle.load(open(vectorizer_file, 'rb'))
        encoder = pickle.load(open(encoder_file, 'rb'))
        
        st.success("✅ Loaded pre-trained model.")
        
    except FileNotFoundError:
        st.write("⌛ Training new model...")
        
        # Preprocess the data
        df['cleaned_resume'] = df['Resume'].apply(preprocess_text)
        
        # Encode the target variable
        encoder = LabelEncoder()
        df['Category_id'] = encoder.fit_transform(df['Category'])
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['cleaned_resume'])
        y = df['Category_id']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model (Multinomial Naive Bayes)
        model = OneVsRestClassifier(MultinomialNB())
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"📊 Model accuracy: **{accuracy:.2f}**")
        
        # Save the model and related components
        pickle.dump(model, open(model_file, 'wb'))
        pickle.dump(vectorizer, open(vectorizer_file, 'wb'))
        pickle.dump(encoder, open(encoder_file, 'wb'))
    
    # Dictionary of skills for each job category
    category_skills = {
        'Data Science': ['Python', 'R', 'SQL', 'Machine Learning', 'Deep Learning', 'Statistics', 'Data Visualization', 'Pandas', 'NumPy', 'TensorFlow', 'PyTorch', 'Scikit-learn'],
        'HR': ['Recruitment', 'Employee Relations', 'Performance Management', 'Onboarding', 'HR Policies', 'HRIS', 'Talent Management', 'Compensation', 'Benefits'],
        'Advocate': ['Legal Research', 'Litigation', 'Contract Review', 'Case Management', 'Client Counseling', 'Legal Writing', 'Court Procedures'],
        'Arts': ['Creativity', 'Design', 'Visual Arts', 'Performing Arts', 'Adobe Creative Suite', 'Portfolio Development', 'Artistic Direction'],
        'Web Designing': ['HTML', 'CSS', 'JavaScript', 'UI/UX', 'Responsive Design', 'Figma', 'Adobe XD', 'Web Development', 'WordPress'],
        'Mechanical Engineer': ['CAD', 'AutoCAD', 'SolidWorks', 'Thermodynamics', 'Machine Design', 'Material Science', 'Manufacturing Processes'],
        'Sales': ['Negotiation', 'CRM Software', 'Lead Generation', 'Sales Strategy', 'Customer Service', 'Business Development', 'Market Analysis'],
        'Accountant': ['Financial Analysis', 'Bookkeeping', 'Tax Preparation', 'Auditing', 'Financial Reporting', 'QuickBooks', 'Excel', 'Account Reconciliation'],
        'Software Engineer': ['Programming', 'Java', 'Python', 'C++', 'JavaScript', 'Software Development', 'Version Control', 'Agile Methodology', 'Data Structures', 'Algorithms'],
        'Blockchain': ['Smart Contracts', 'Solidity', 'Ethereum', 'Consensus Mechanisms', 'Cryptography', 'DApps', 'Blockchain Architecture'],
        'DevOps Engineer': ['CI/CD', 'Docker', 'Kubernetes', 'Jenkins', 'AWS', 'Cloud Infrastructure', 'Automation', 'Linux', 'Scripting'],
        'Database': ['SQL', 'Database Design', 'Database Administration', 'NoSQL', 'MongoDB', 'MySQL', 'PostgreSQL', 'Data Modeling', 'Query Optimization'],
        'Hadoop': ['Big Data', 'MapReduce', 'HDFS', 'Spark', 'Hive', 'Data Processing', 'Distributed Computing', 'Yarn'],
        'ETL Developer': ['Data Integration', 'ETL Tools', 'Data Warehousing', 'Data Migration', 'SQL', 'Data Mapping', 'Informatica', 'Talend'],
        'DBA': ['Database Administration', 'Performance Tuning', 'Backup and Recovery', 'SQL', 'Database Security', 'Oracle', 'MySQL', 'Database Design'],
        'Business Analyst': ['Requirements Gathering', 'Business Process Modeling', 'Data Analysis', 'Process Improvement', 'SWOT Analysis', 'Use Case Development', 'Stakeholder Management'],
        'SAP Developer': ['SAP ABAP', 'SAP HANA', 'SAP Fiori', 'SAP S/4HANA', 'SAP ERP', 'Business Process Integration', 'SAP Modules'],
        'Automation Testing': ['Selenium', 'Test Automation', 'Test Scripts', 'QA Methodologies', 'CI/CD', 'Test Planning', 'Regression Testing', 'JUnit', 'TestNG'],
        'Network Security Engineer': ['Network Protocols', 'Firewalls', 'IDS/IPS', 'VPN', 'Vulnerability Assessment', 'Security Audits', 'Penetration Testing', 'Cybersecurity'],
        'PMO': ['Project Management', 'Program Management', 'Stakeholder Management', 'Risk Management', 'Resource Planning', 'MS Project', 'Project Scheduling', 'Budget Management'],
        'Python Developer': ['Python', 'Django', 'Flask', 'REST APIs', 'OOP', 'SQLAlchemy', 'Web Development', 'Data Analysis'],
        'Java Developer': ['Java', 'Spring', 'Hibernate', 'J2EE', 'REST APIs', 'SQL', 'OOP', 'Microservices', 'Maven', 'JUnit'],
        'Operations Manager': ['Operations Management', 'Process Improvement', 'Team Leadership', 'Budget Management', 'KPI Tracking', 'Resource Allocation', 'Strategic Planning'],
        'Tableau': ['Data Visualization', 'Dashboard Creation', 'Business Intelligence', 'Data Analysis', 'Data Interpretation', 'Tableau Desktop', 'Tableau Server'],
    }
    
    if uploaded_file is not None:
        # Extract text from resume
        resume_text = extract_text_from_resume(uploaded_file)
        
        # Preprocess the resume text
        cleaned_resume = preprocess_text(resume_text)
        
        # Transform the resume text using the fitted vectorizer
        resume_vector = vectorizer.transform([cleaned_resume])
        
        # Predict the category
        category_id = model.predict(resume_vector)[0]
        predicted_category = encoder.inverse_transform([category_id])[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(resume_vector)[0]
        category_probs = [(encoder.inverse_transform([i])[0], prob) for i, prob in enumerate(probabilities)]
        category_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Display results
        st.header("🔍 Results")
        st.subheader(f"Best Match: **{predicted_category}**")
        
        # Show top 3 job role matches with probabilities
        st.write("### Top job role matches:")
        for i, (category, prob) in enumerate(category_probs[:3]):
            st.write(f"**{i+1}. {category}** — {prob*100:.2f}%")
        
        # Generate and display feedback
        st.header("📝 Personalized Feedback")
        feedback = generate_feedback(predicted_category, resume_text, category_skills)
        st.write(feedback)
        
        # Show word cloud of skills mentioned in resume (optional)
        st.header("🔧 Skills Analysis")
        top_skills = []
        for skill_list in category_skills.values():
            for skill in skill_list:
                if skill.lower() in resume_text.lower():
                    top_skills.append(skill)
        
        if top_skills:
            st.write("**Skills identified in your resume:**")
            st.write("🏆 " + ", ".join(top_skills))
        else:
            st.write("No specific skills were identified in your resume. Consider adding more technical skills related to your target role.")
        
        # Resume improvement suggestions
        st.header("🚀 Resume Improvement Suggestions")
        # General suggestions
        st.write("**General suggestions:**")
        st.write("1. ✍️ Use action verbs to start bullet points")
        st.write("2. 💡 Quantify achievements when possible")
        st.write("3. 🎯 Tailor your resume to the specific job role")
        st.write("4. 📐 Keep formatting consistent")
        st.write("5. 🔍 Proofread for grammar and spelling errors")
        
        # Word count analysis
        word_count = len(re.findall(r'\w+', resume_text))
        st.write(f"📝 Your resume contains approximately **{word_count}** words.")
        if word_count < 300:
            st.write("⚠️ Your resume seems quite short. Consider adding more details about your experiences and skills.")
        elif word_count > 1000:
            st.write("⚠️ Your resume is quite lengthy. Consider condensing it to highlight the most relevant information.")

# Run the app
if __name__ == "__main__":
    main()
