# AI Text Detector

The AI Detector project is a robust machine-learning system designed to classify text as either human-written or AI-generated. Built to improve upon the accuracy of existing services like GPT Zero, this tool minimizes false positives and negatives to ensure fairness in academic and professional environments.

## Table of Contents
1. [Architecture & Approach](#architecture--approach)
2. [Data Collection](#data-collection)
3. [Limitations & Future Work](#limitations--future-work)
4. [Setup & Installation](#setup--installation)
5. [Usage](#usage)
6. [Contributors](#contributors)

## Architecture & Approach

This project utilizes an ensemble approach, combining three distinct machine-learning models to evaluate text from multiple angles and create one comprehensive system:

### 1. Content-Based Model (Logistic Regression)
This model acts as a foundational baseline by analyzing the vocabulary and content of the text. 
* **Feature Extraction**: It uses a `CountVectorizer` to transform the text into numerical features based on the occurrences of specific words.
* **Classification**: A Logistic Regression model assigns probabilities to the text based on these numerical counts to classify the text. 

### 2. Structural-Based Model (Random Forest)
To understand how the text is built, this model analyzes the structural and syntactic features of the input.
* **Feature Extraction**: Using the `spaCy` NLP library, the system extracts structural metrics such as the total number of tokens, number of sentences, and average sentence length. It also counts part-of-speech distributions, including nouns, verbs, adjectives, and punctuation.
* **Classification**: A Random Forest classifier utilizes these structural patterns to differentiate between human and AI writing, leveraging multiple decision trees to effectively handle complex patterns and noise.

### 3. Deep Learning Model (LSTM)
To capture nuanced contextual and long-range language patterns, the system employs a Long Short-Term Memory (LSTM) neural network.
* **Embedding Layer**: The tokenized text is fed into an embedding layer that maps tokens to dense vector representations, capturing both semantic and syntactic information.
* **Sequential Processing**: The LSTM processes the input sequence token-by-token, allowing it to understand the broader context of each word. 
* **Classification**: The final output passes through a dense layer with a sigmoid activation function to output a final probability between 0 and 1 indicating the likelihood of AI generation.

During live evaluation, the system extracts features for the user's input, runs predictions across all three models, and calculates a weighted, combined probability to make a final classification.

## Data Collection

Training a highly accurate model requires a balanced and consistent labeled dataset. 
* **Human-Written Data**: A script scraped Wikipedia articles to build a large dataset of human-authored text with a uniform, academic style.
* **AI-Generated Data**: To ensure the topics were identical and the text structure remained comparable, a local LLM (Ollama) was prompted to generate articles on the exact same Wikipedia subjects.

## Limitations & Future Work

**Current Limitations:**
* **Style Bias**: Because the model was trained primarily on Wikipedia data, it is heavily tailored to formal, academic writing and may disproportionately misclassify alternative writing styles.
* **Text Length Adjustments**: Classification remains difficult for extremely short texts (where indicators are spread too thin) and extremely long texts.

**Future Improvements:**
* Implementing sentence-by-sentence analysis to highlight exactly which parts of a text the model thinks is AI-generated.
* Expanding the dataset to include diverse writing styles and patterns.
* Developing a front-end visualization website to display the percentage of AI use.

## Setup & Installation

1. Ensure your system has Python 3.8 or later installed.
2. Clone this repository:
   ```bash
   git clone [https://github.com/helloswayamshah/AI-Detector.git](https://github.com/helloswayamshah/AI-Detector.git)


3. Create and activate a Python virtual environment:
* **Windows**:
```console
python -m venv env
.\env\Scripts\activate

```


* **Linux/Mac**:
```bash
python -m venv env
source ./env/bin/activate

```




4. Install the required dependencies (`pandas`, `scikit-learn`, `numpy`, `spacy`, `tensorflow`):
* **Windows**:
```console
pip install -r .\requirements.txt

```


* **Linux/Mac**:
```bash
pip install -r ./requirements.txt

```




5. Download the necessary `spaCy` English language model:
```bash
python -m spacy download en_core_web_sm

```



## Usage

To train the models and launch the interactive predictor, run the main Python script:

* **Windows**:
```console
python .\ai_detector.py

```


* **Linux/Mac**:
```bash
python ./ai_detector.py

```



The script will automatically train the Logistic Regression, Random Forest, and LSTM models using the `TRAIN.csv` dataset in the `data` directory. Once training is complete, the terminal will enter a **Live Demo** mode, prompting you to paste any text string to evaluate whether it is human-written or AI-generated.

## Contributors

* **Swayam Shah**
* personal-email: [helloswayamshah@gmail.com](mailto:helloswayamshah@gmail.com)
* school-email: [sshah36@ucsc.edu](mailto:sshah36@ucsc.edu)


* **Atharva Tawde**
* **Jiancheng Xiong**
* **Karthik Chaparala**

```

```
