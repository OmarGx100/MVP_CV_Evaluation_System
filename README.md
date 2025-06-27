# 🎯 HR CV Evaluation System

An AI-powered HR tool that automatically evaluates candidate CVs against job descriptions using local LLM models through Ollama. Built with Streamlit for an intuitive web interface.

## 📖 Description

The HR CV Evaluation System is designed to streamline the recruitment process by leveraging AI to perform initial CV screening. The system evaluates candidates with the same rigor as an experienced HR professional, providing detailed assessments and rankings to help identify the most suitable candidates for any position.

### Key Capabilities:
- **Automated CV Analysis**: Processes PDF and DOCX files to extract and analyze candidate information
- **AI-Powered Evaluation**: Uses Ollama's local LLM models (optimized for Qwen2.5:4b) for comprehensive candidate assessment
- **Strict Scoring System**: Implements competitive evaluation criteria similar to real HR practices
- **Comprehensive Reporting**: Generates detailed explanations for each candidate's score and ranking
- **Flexible Filtering**: Allows customizable qualification thresholds for different positions
- **Error Handling**: Robust system that manages failed files and provides detailed error logging

## 🚀 Features

### Core Functionality
- ✅ **Multi-format Support**: Handles both PDF and DOCX CV files
- ✅ **Real-time Processing**: Live progress tracking during evaluation
- ✅ **Intelligent Ranking**: Automatic candidate ranking based on scores
- ✅ **Detailed Analysis**: Comprehensive evaluation covering skills, experience, education, and achievements
- ✅ **Customizable Thresholds**: Adjustable qualification percentage for different roles
- ✅ **Export Results**: Download results as CSV for further analysis

### Technical Features
- ✅ **Local AI Processing**: Uses Ollama for privacy-focused, local LLM evaluation
- ✅ **Robust Error Handling**: Failed files are quarantined with detailed error logs
- ✅ **System Health Checks**: Automatic Ollama connectivity verification
- ✅ **Streamlit Web Interface**: User-friendly web application
- ✅ **Comprehensive Logging**: Detailed logs for troubleshooting and audit trails

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running
- Qwen2.5:4b model (or modify code for your preferred model)

### Step 1: Install Ollama
```bash
# Install Ollama (visit https://ollama.ai for platform-specific instructions)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the Qwen2.5:4b model
ollama pull qwen2.5:4b
```

### Step 2: Clone and Setup Project
```bash
# Clone the repository (or download the files)
git clone <your-repository-url>
cd hr-cv-evaluation-system

# Install Python dependencies
pip install streamlit pandas requests PyPDF2 python-docx pathlib
```

### Step 3: Start Ollama Service
```bash
# Start Ollama server
ollama serve
```

## 🎮 Usage

### Starting the Application
```bash
# Run the Streamlit application
streamlit run cv_evaluation_app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Interface

1. **System Status Check**: The app automatically verifies Ollama connectivity
2. **Job Description**: Enter the complete job description including:
   - Job title and role requirements
   - Required years of experience
   - Essential skills and qualifications
   - Educational requirements
   - Any specific criteria

3. **Configuration**:
   - **CV Folder Path**: Specify the directory containing candidate CVs
   - **Qualification Threshold**: Set minimum score percentage (default: 70%)

4. **Processing**: Click "Start Evaluation" to begin automated CV analysis

5. **Results**: View rankings, download CSV reports, and analyze candidate evaluations

### Expected Folder Structure
```
your_cv_folder/
├── candidate1_resume.pdf
├── candidate2_cv.docx
├── candidate3_resume.pdf
└── ...
```

### Output Structure
After processing, you'll get:
```
your_cv_folder/
├── failed_cvs/                    # Failed files (if any)
│   └── problematic_cv.pdf
├── cv_evaluation_results_YYYYMMDD_HHMMSS.csv
└── cv_evaluation.log              # Detailed processing logs
```

## 📊 Sample Output

The system generates a comprehensive CSV report with the following columns:

| Column | Description |
|--------|-------------|
| `Filename` | Original CV filename |
| `Candidate_Name` | Extracted candidate name |
| `Score_Percentage` | AI-generated score (0-100%) |
| `Qualification_Status` | Qualified/Not Qualified based on threshold |
| `Detailed_Explanation` | Comprehensive evaluation breakdown |
| `Ranking` | Overall candidate ranking |

### Sample Evaluation Output:
```
SCORE: 78%

EVALUATION:
- Skills Match: Strong technical skills in Python, Java, and cloud technologies. 
  Matches 85% of required technical competencies.
- Experience Level: 5 years experience meets the 3-5 year requirement. 
  Previous roles show progressive responsibility.
- Education: Computer Science degree from accredited university aligns well.
- Achievements: Led 3 successful projects, increased team efficiency by 25%.
- Gaps/Weaknesses: Limited experience with specific framework mentioned in job requirements.
- Overall Assessment: Strong candidate with solid foundation, minor skill gaps can be addressed through training.
```

## ⚙️ Configuration

### Model Configuration
To use a different Ollama model, modify the `model_name` in the `CVEvaluationSystem` class:

```python
self.model_name = "your-preferred-model"  # Change from "qwen2.5:4b"
```

### Evaluation Strictness
The system is configured for strict evaluation by default. To adjust:
- Modify the prompt in `evaluate_cv_with_llm()` method
- Adjust temperature and other LLM parameters
- Change scoring thresholds in the evaluation logic

### Logging Configuration
Logs are saved to `cv_evaluation.log`. To modify logging behavior, adjust the logging configuration in the code.

## 🔧 Troubleshooting

### Common Issues

**❌ "Cannot connect to Ollama"**
```bash
# Ensure Ollama is running
ollama serve

# Check if model is available
ollama list
```

**❌ "Model not found"**
```bash
# Pull the required model
ollama pull qwen2.5:4b
```

**❌ "No text could be extracted"**
- Check if CV files are text-based (not scanned images)
- Verify file format is PDF or DOCX
- Check file permissions

**❌ "Permission denied" errors**
- Ensure Python has write permissions to the CV folder
- Check if files are not opened in other applications

### Log Analysis
Check `cv_evaluation.log` for detailed error information:
```bash
tail -f cv_evaluation.log
```

## 📈 Performance Optimization

### For Large CV Batches:
- Increase Ollama timeout in requests (currently 120s)
- Process files in smaller batches
- Consider using GPU acceleration for Ollama

### For Better Accuracy:
- Use larger, more capable models (if hardware allows)
- Fine-tune prompts for specific job types
- Adjust temperature and other LLM parameters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) for providing local LLM capabilities
- [Streamlit](https://streamlit.io) for the excellent web framework
- [Qwen Team](https://github.com/QwenLM/Qwen) for the efficient language model

## 📞 Support

For questions, issues, or feature requests:
1. Check the troubleshooting section
2. Review existing issues in the repository
3. Create a new issue with detailed information
4. Check logs for error details

---

**Made with ❤️ for HR professionals who want to leverage AI for better recruitment decisions.**