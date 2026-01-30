# 📊 Python – Text Analytics & Media Analysis 

## Description
- In this assignment, I applied Python and text analytics techniques to analyze media content across U.S. and international news sources. The project focused on data collection, natural language processing (NLP), and visualization using histograms to uncover patterns in word usage. Through hands-on implementation, I strengthened my ability to transform unstructured text data into meaningful insights while addressing real-world business and media analysis questions.

## Skills Developed 
- Leave blank

## 📰 Question 1 – Media Article Collection
Question Overview
- For this question, I selected a single news topic covered within the last three months and scraped 10 total articles—7 from U.S.-based media channels and 3 from international outlets. Each article included the title, author(s), publication date, and full text. Non-coding scraping tools were primarily used, with Yahoo permitted as one optional coding-based source.]

<b> Part 1 - International Media Data Aggregation (CSV Processing) </b>

Code Overview:
- This code uses the Pandas library to read multiple CSV files from international media sources and merge them into a single dataset. The program concatenates the individual CSV files into one DataFrame, resets the index to maintain consistency, removes any redundant index columns if present, and saves the cleaned data into a new CSV file for subsequent analysis.
  
```python
import pandas
input_dataframe = pandas.concat(map(pandas.read_csv, [
    'kcra3.csv',
    'abc10.csv',
    'sac_bee.csv',
    'fox_news.csv',
    'abc_news.csv',
    'usa_today.csv',
    'nbc-news (1).csv'
]))
input_dataframe = input_dataframe.reset_index(drop=True)

if "Index" in input_dataframe.columns:
    del input_dataframe["Index"]

input_dataframe.to_csv('us_data.csv', index=True, index_label="Index")
print("done")
```

Skills Developed –

Data Integration:
- Learned how to combine multiple datasets from different international news sources into a single structured DataFrame for unified analysis.

Index Management:
- Practiced resetting and managing DataFrame indices to ensure the merged dataset remained organized and free of inconsistencies.

Data Cleaning:
- Identified and removed unnecessary index columns to prevent duplication and maintain data integrity.

Data Export and Reusability:
- Developed the ability to export processed data into a new CSV file, enabling efficient reuse in later text analytics and NLP tasks.

<b> Part 2 - International Media Data Aggregation (CSV Processing) </b>

Code Overview:
- This code uses the Pandas library to read multiple CSV files from international media sources and merge them into a single dataset. The program concatenates the individual CSV files into one DataFrame, resets the index to maintain consistency, removes any redundant index columns if present, and saves the cleaned data into a new CSV file for subsequent analysis.

```python
import pandas
input_dataframe = pandas.concat(map(pandas.read_csv, [
    'cbc.csv',
    'irish_examiner.csv',
    'press_gazette.csv'
]))
input_dataframe = input_dataframe.reset_index(drop=True)

if "Index" in input_dataframe.columns:
    del input_dataframe["Index"]

input_dataframe.to_csv('int_data.csv', index=True, index_label="Index")
print("done")
```

Skills Developed –

Data Integration:
- Learned how to combine multiple datasets from different international news sources into a single structured DataFrame for unified analysis.

Index Management:
- Practiced resetting and managing DataFrame indices to ensure the merged dataset remained organized and free of inconsistencies.

Data Cleaning:
- Identified and removed unnecessary index columns to prevent duplication and maintain data integrity.

Data Export and Reusability:
- Developed the ability to export processed data into a new CSV file, enabling efficient reuse in later text analytics and NLP tasks.

## 📈 Question 2 – Unigram Histograms (U.S. vs International Media)
Question Overview
- Using Python, I processed article text to calculate and visualize the top 10 unigrams for both U.S.-based and international news sources. Histograms were created to compare word frequency distributions between the two groups.

Skills Developed –
Text Preprocessing: 
- Cleaned and tokenized article text for analysis.

Data Visualization: 
- Created histograms to represent unigram frequency distributions.

Comparative Analysis:
- Interpreted similarities and differences between U.S. and international media coverage .

<b> Part 1 - U.S. Media Text Preprocessing & Tokenization (NLP Preparation) </b>

Code Overview:
-This code prepares U.S.-based news article text for natural language processing (NLP) analysis. Using Pandas, NLTK, and regular expressions, the program loads article text from a CSV file, cleans and standardizes the text, removes unnecessary characters and URLs, eliminates common stopwords, and tokenizes the text into individual words for further unigram, bigram, and trigram analysis.

```python
import pandas
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ssl

inputdata={}
inputdata = pandas.read_csv('us_data.csv', header=[0], index_col=0).to_dict()

textdictionary = inputdata.get('text')
textlist =  list(textdictionary.values())

textinstring = ''
for eachletter in  textlist:
    textinstring += ' '+ str(eachletter)

lowercasetext=textinstring.lower()

lowercasetext= re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', lowercasetext)

lowercasetext = lowercasetext.replace(".", "")
lowercasetext = lowercasetext.replace("#", "")
lowercasetext = lowercasetext.replace(",", "")
lowercasetext = lowercasetext.replace("\\", "")
lowercasetext = lowercasetext.replace("(", "")
lowercasetext = lowercasetext.replace(")", "")
lowercasetext = lowercasetext.replace("+", "")
lowercasetext = lowercasetext.replace("!", "")
lowercasetext = lowercasetext.replace("&", "")

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
nltk.download('punkt_tab')

text_tokens = word_tokenize(lowercasetext)
text_tokens_without_stopwords = [word for word in text_tokens if not word in stopwords.words()]

print(text_tokens_without_stopwords)
```

<b> Part 2 - International Media Text Preprocessing & Tokenization (NLP Preparation) </b>

Code Overview:
- This code prepares international news article text for natural language processing (NLP) analysis. Using Pandas, NLTK, and regular expressions, the program loads article text from an international media CSV file, cleans and normalizes the text, removes URLs and irrelevant characters, filters out common stopwords, and tokenizes the text into individual words. The processed tokens are then ready for unigram, bigram, and trigram frequency analysis.

## 🔗 Question 3 – Bigram & Trigram Analysis
Question Overview
- This section expanded the analysis to include bigrams and trigrams, visualized through histograms for both U.S. and international sources. Additionally, stemming and lemmatization techniques were compared to evaluate their impact on text patterns.

Skills Developed –
N-gram Modeling: 
- Generated bigrams and trigrams to capture contextual word relationships.

Advanced NLP Techniques: 
- Applied stemming and lemmatization to compare linguistic normalization methods.

Critical Interpretation: 
- Analyzed how phrase-level patterns differed across regions and preprocessing techniques .

## 🌍 Question 4 – Non-English NLP with NLTK (GenAI-Permitted)
Code Overview

For this GenAI-permitted question, I installed an NLTK language package for a non-English language and analyzed a foreign-language article scraped using non-coding methods. Histograms for unigrams, bigrams, and trigrams were generated and compared to English-language results from earlier questions.

Skills Developed –

Multilingual NLP: Installed and used a non-English NLTK language package.

Cross-Language Comparison: Identified similarities and differences between English and non-English text patterns.

Engagement Analysis: Explored potential relationships between word usage patterns and user engagement metrics such as likes or comments .

📧 Question 5 – Email Spam Detection Analysis
Code Overview

In this question, I analyzed 20 spam emails by extracting and processing both the subject lines and body text. Top unigrams, bigrams, and trigrams were calculated to identify common linguistic patterns associated with spam content.

Skills Developed –

Text Classification Foundations: Identified linguistic features commonly associated with spam emails.

Feature Extraction: Compared word and phrase patterns between email titles and body text.

Practical Application: Evaluated how n-gram frequency supports spam detection logic .
