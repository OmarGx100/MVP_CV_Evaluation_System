import streamlit as st
import os
import pandas as pd
import requests
import json
import logging
from pathlib import Path
import shutil
from datetime import datetime
import re
import time
import requests

# PDF and DOCX processing libraries
try:
    import PyPDF2
    import docx
except ImportError:
    st.error("Required libraries not found. Please install: pip install PyPDF2 python-docx")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cv_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CVEvaluationSystem:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "qwen3:1.7b"
        self.failed_folder = "failed_cvs"
        
    def check_ollama_connection(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if self.model_name in model_names:
                    return True, f"✅ Ollama is running with {self.model_name}"
                else:
                    return False, f"❌ Model {self.model_name} not found. Available models: {model_names}"
            else:
                return False, f"❌ Ollama responded with status {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"❌ Cannot connect to Ollama: {str(e)}"
                             
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")
    
    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"DOCX extraction failed: {str(e)}")
    
    def extract_text_from_file(self, file_path):
        """Extract text from PDF or DOCX file"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return self.extract_text_from_docx(file_path)
        else:
            raise Exception(f"Unsupported file format: {file_extension}")
    
    def create_failed_folder(self, base_path):
        """Create failed folder if it doesn't exist"""
        failed_path = os.path.join(base_path, self.failed_folder)
        os.makedirs(failed_path, exist_ok=True)
        return failed_path
    
    def move_failed_file(self, file_path, failed_folder_path, error_message):
        """Move failed file to failed folder and log error"""
        try:
            filename = os.path.basename(file_path)
            destination = os.path.join(failed_folder_path, filename)
            shutil.move(file_path, destination)
            
            # Log the error
            logger.error(f"Failed to process {filename}: {error_message}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to move file {filename}: {str(e)}")
            return False

    def evaluate_cv_with_llm_openrouter(self, cv_text, job_description):
        """Evaluate CV using OpenRouter LLM"""

        prompt = f"""
    You are an experienced HR professional evaluating a candidate's CV for a specific job position. 
    Be STRICT and CRITICAL in your evaluation as if you're competing for limited positions.
    IMPORTANT YOU MUST INCLUDE IN YOUR RESPONSE THE TWO ATTRIBUTES LIKE BELOW `SCORE:`, `EVALUATION:`

    JOB DESCRIPTION:
    {job_description}

    CANDIDATE'S CV:
    {cv_text}

    Please provide a STRICT evaluation with the following format:

    SCORE: [0-100]%

    EVALUATION:
    - Skills Match: [Detailed analysis of how candidate's skills match job requirements]
    - Experience Level: [Analysis of years and quality of experience vs requirements]
    - Education: [Educational background assessment]
    - Achievements: [Notable accomplishments relevant to the role]
    - Gaps/Weaknesses: [Areas where candidate falls short]
    - Overall Assessment: [Summary and recommendation]

    Be harsh but fair. Only exceptional candidates should score above 85%. 
    Average candidates should score 60-75%. Below-average candidates should score below 60%.
    """

        try:
            headers = {
                "Authorization": f"Bearer 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",  # OpenRouter API Key
                "Content-Type": "application/json"
            }

            payload = {
                "model": "mistralai/mistral-small-3.2-24b-instruct:free",  # e.g., "mistralai/mistral-7b-instruct" or "openai/gpt-4"
                "messages": [
                    {"role": "system", "content": "You are an expert HR evaluator."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 1000
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                raise Exception(f"OpenRouter request failed with status {response.status_code}: {response.text}")

        except Exception as e:
            raise Exception(f"LLM evaluation failed: {str(e)}")

    def evaluate_cv_with_llm(self, cv_text, job_description):
        """Evaluate CV using Ollama LLM"""
        prompt = f"""
        You are an experienced HR professional evaluating a candidate's CV for a specific job position. 
        Be STRICT and CRITICAL in your evaluation as if you're competing for limited positions.
        IMPORTANT YOU MUST INCLUDE IN YOU RESPONSE THE TWO ATTRIBUTES LIKE BELOW `SCORE:`, `EVALUATION:`

        JOB DESCRIPTION:
        {job_description}
        
        CANDIDATE'S CV:
        {cv_text}
        
        Please provide a STRICT evaluation with the following format:
        
        SCORE: [0-100]%
        
        EVALUATION:
        - Skills Match: [Detailed analysis of how candidate's skills match job requirements]
        - Experience Level: [Analysis of years and quality of experience vs requirements]
        - Education: [Educational background assessment]
        - Achievements: [Notable accomplishments relevant to the role]
        - Gaps/Weaknesses: [Areas where candidate falls short]
        - Overall Assessment: [Summary and recommendation]
        
        Be harsh but fair. Only exceptional candidates should score above 85%. 
        Average candidates should score 60-75%. Below-average candidates should score below 60%.
        """
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response received')
            else:
                raise Exception(f"LLM request failed with status {response.status_code}")
                
        except Exception as e:
            raise Exception(f"LLM evaluation failed: {str(e)}")
    
    def parse_llm_response(self, llm_response):
        """Parse LLM response to extract score and evaluation"""
        try:
            # Extract score using regex
            score_match = re.search(r'SCORE:\s*(\d+)%?', llm_response, re.IGNORECASE)
            if score_match:
                score = int(score_match.group(1))
            else:
                # Fallback: look for any percentage in the response
                percentage_match = re.search(r'(\d+)%', llm_response)
                score = int(percentage_match.group(1)) if percentage_match else 0
            
            # Extract evaluation text (everything after EVALUATION:)
            eval_match = re.search(r'EVALUATION:\s*(.*)', llm_response, re.DOTALL | re.IGNORECASE)
            evaluation = eval_match.group(1).strip() if eval_match else llm_response
            
            return score, evaluation
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            return 0, llm_response
    
    def extract_candidate_name(self, cv_text):
        """Try to extract candidate name from CV text"""
        try:
            lines = cv_text.split('\n')
            # Usually the name is in the first few lines
            for line in lines[:5]:
                line = line.strip()
                if len(line) > 2 and len(line) < 50:
                    # Basic heuristic: if it looks like a name (2-4 words, mostly alphabetic)
                    words = line.split()
                    if 2 <= len(words) <= 4 and all(word.replace('.', '').isalpha() for word in words):
                        return line
            return "Unknown"
        except:
            return "Unknown"
    
    def process_cv_folder(self, folder_path, job_description, threshold, progress_callback=None):
        """Process all CVs in the specified folder"""
        results = []
        failed_files = []
        
        # Get all PDF and DOCX files
        pdf_files = list(Path(folder_path).glob("*.pdf"))
        docx_files = list(Path(folder_path).glob("*.docx"))
        all_files = pdf_files + docx_files
        
        if not all_files:
            return results, failed_files, "No PDF or DOCX files found in the specified folder."
        
        # Create failed folder
        failed_folder_path = self.create_failed_folder(folder_path)
        
        total_files = len(all_files)
        processed_files = 0
        
        for file_path in all_files:
            try:
                if progress_callback:
                    progress_callback(processed_files, total_files, f"Processing {file_path.name}...")
                
                # Extract text from CV
                cv_text = self.extract_text_from_file(str(file_path))
                
                if not cv_text.strip():
                    raise Exception("No text could be extracted from the file")
                
                # Evaluate with LLM
                # llm_response = self.evaluate_cv_with_llm(cv_text, job_description)
                llm_response = self.evaluate_cv_with_llm_openrouter(cv_text, job_description)
                score, evaluation = self.parse_llm_response(llm_response)
                
                # Extract candidate name
                candidate_name = self.extract_candidate_name(cv_text)
                
                # Determine qualification status
                qualification_status = "Qualified" if score >= threshold else "Not Qualified"
                
                results.append({
                    'Filename': file_path.name,
                    'Candidate_Name': candidate_name,
                    'Score_Percentage': score,
                    'Qualification_Status': qualification_status,
                    'Detailed_Explanation': evaluation,
                    'Ranking': 0  # Will be set after sorting
                })
                
                processed_files += 1
                
            except Exception as e:
                error_message = str(e)
                failed_files.append({
                    'filename': file_path.name,
                    'error': error_message
                })
                
                # Move failed file and log error
                self.move_failed_file(str(file_path), failed_folder_path, error_message)
                processed_files += 1
        
        # Sort results by score (descending) and assign rankings
        results.sort(key=lambda x: x['Score_Percentage'], reverse=True)
        for i, result in enumerate(results):
            result['Ranking'] = i + 1
        
        return results, failed_files, None

def main():

    
    st.set_page_config(
        page_title="HR CV Evaluation System",
        page_icon="📋",
        layout="wide"
    )
    
    
    st.title("🎯 HR CV Evaluation System")
    st.markdown("Evaluate candidate CVs using AI-powered analysis with Ollama")
    
    # Initialize the system
    cv_system = CVEvaluationSystem()
    
    # Check Ollama connection
    st.subheader("🔌 System Status")
    connection_status, message = cv_system.check_ollama_connection()
    
    if connection_status:
        st.success(message)
    else:
        st.error(message)
        st.stop()
    
    # Main interface
    st.subheader("📝 Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        job_description = st.text_area(
            "Job Description",
            placeholder="Enter the complete job description including required skills, experience, education, etc.",
            height=200
        )
    
    with col2:
        folder_path = st.text_input(
            "CV Folder Path",
            placeholder="Enter the path to folder containing CVs",
            help="Path to folder containing PDF and DOCX files"
        )
        
        threshold = st.slider(
            "Qualification Threshold (%)",
            min_value=0,
            max_value=100,
            value=70,
            help="Minimum score percentage for qualification"
        )
    
    # Process button
    if st.button("🚀 Start Evaluation", type="primary"):
        if not job_description.strip():
            st.error("Please enter a job description")
        elif not folder_path.strip():
            st.error("Please enter a folder path")
        elif not os.path.exists(folder_path):
            st.error("Folder path does not exist")
        else:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, message):
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress)
                status_text.text(f"{message} ({current}/{total})")
            
            # Process CVs
            with st.spinner("Processing CVs..."):
                start_time = time.time()
                results, failed_files, error = cv_system.process_cv_folder(
                    folder_path, job_description, threshold, update_progress
                )
                end_time = time.time()
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if error:
                st.error(error)
            else:
                # Display results summary
                st.subheader("📊 Results Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Processed", len(results))
                with col2:
                    qualified_count = len([r for r in results if r['Qualification_Status'] == 'Qualified'])
                    st.metric("Qualified Candidates", qualified_count)
                with col3:
                    st.metric("Failed Files", len(failed_files))
                with col4:
                    st.metric("Processing Time", f"{end_time - start_time:.1f}s")
                
                if results:
                    # Convert to DataFrame
                    df = pd.DataFrame(results)
                    
                    # Display results table
                    st.subheader("🏆 Candidate Rankings")
                    st.dataframe(
                        df[['Ranking', 'Candidate_Name', 'Score_Percentage', 'Qualification_Status', 'Filename']],
                        use_container_width=True
                    )
                    
                    # Qualified candidates section
                    qualified_df = df[df['Qualification_Status'] == 'Qualified']
                    if not qualified_df.empty:
                        st.subheader(f"✅ Qualified Candidates (Score ≥ {threshold}%)")
                        st.dataframe(qualified_df, use_container_width=True)
                    
                    # Download CSV
                    csv_data = df.to_csv(index=False)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"cv_evaluation_results_{timestamp}.csv"
                    
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        type="primary"
                    )
                
                # Display failed files if any
                if failed_files:
                    st.subheader("❌ Failed Files")
                    failed_df = pd.DataFrame(failed_files)
                    st.dataframe(failed_df, use_container_width=True)
                    st.info(f"Failed files have been moved to: {os.path.join(folder_path, cv_system.failed_folder)}")

if __name__ == "__main__":
    main()