# Language Detection using Machine Learning

## Overview

This project implements a **Language Detection** system using classical Machine Learning techniques on multilingual text samples. The goal is to automatically identify the language of any given input text.

The entire project is designed to be run on **Google Colab**, leveraging free cloud resources and popular NLP/ML Python libraries.

## Features

- Creation of a small custom multilingual dataset or use of public datasets like [papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification).
- Text preprocessing: cleaning, normalization.
- Feature engineering using TF-IDF on character n-grams.
- Training and comparison of multiple classifiers:
  - Multinomial Naive Bayes
  - Support Vector Machines (SVM)
  - Logistic Regression
- Model evaluation via accuracy scores, classification reports, and confusion matrix visuals.
- Prediction utility function for new input texts with confidence scores.
- Interactive command-line interface for language prediction.
- Model saving and loading for deployment readiness.

## Supported Languages

The model currently supports detecting the following languages (expandable with more data):

- English
- French
- Spanish
- German
- Italian
- Portuguese
- Russian
- Japanese
- Hindi
- Arabic

## Setup and Usage

### Prerequisites

- Python 3.x environment (Google Colab recommended)
- Required Python libraries (installed automatically in Colab notebook):
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`
  - `langdetect`, `polyglot`, `textblob`, `datasets`

### Running the Project

1. Open `Language_Detection.ipynb` in Google Colab.
2. Run all notebook cells sequentially to:
   - Install dependencies
   - Prepare and preprocess dataset
   - Extract features and vectorize text
   - Train and evaluate multiple models
   - Save the trained model and vectorizer
3. Use the provided prediction functions to detect language on new inputs.
4. Optionally, activate the interactive testing CLI for hands-on predictions.

## Results and Notes

- Sample dataset in the notebook is minimal and for demonstration only.
- Due to the small dataset size, model accuracy may be initially low.
- For practical use, expand the dataset with more and diverse text samples.
- Classical ML models provide a solid baseline; improving with deep learning or transformer-based models is a good next step.

## Future Enhancements

- Integrate larger and more balanced multilingual datasets.
- Use deep learning models such as LSTM or Transformers for improved accuracy.
- Add a web UI using frameworks like Streamlit or Gradio.
- Deploy on cloud services or as an API for real-world applications.
- Support additional languages and dialects.
- Include confidence thresholding and handling of mixed-language inputs.

## File Structure

- `Language_Detection.ipynb` — Complete Google Colab notebook with full implementation.
- `language_detector_model.pkl` — Saved trained ML model.
- `language_detector_vectorizer.pkl` — Saved TF-IDF vectorizer.
- `README.md` — This documentation file.

## References

- Hugging Face Dataset: papluca/language-identification
- Python libraries: scikit-learn, pandas, numpy, matplotlib, seaborn, langdetect, polyglot, textblob
- Google Colab platform for free GPU/TPU access

## Contributing and Support

Contributions, bug reports, or feature requests are welcome via GitHub issues or pull requests.  

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Thank you for exploring this language detection project! Happy coding! 🚀
