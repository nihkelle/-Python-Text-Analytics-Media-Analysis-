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

<ins> Skills Developed – </ins>

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

<ins> Skills Developed – </ins>

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

<ins> Skills Developed – </ins>

Data Extraction and Transformation:
- Learned how to extract specific text fields from a CSV file and convert them into a format suitable for NLP analysis.

Text Cleaning and Normalization:
- Applied lowercase conversion, URL removal, and character filtering to standardize raw text and reduce noise in the dataset.

Tokenization:
- Used NLTK’s word tokenizer to break text into individual tokens, a foundational step for text analytics and n-gram modeling.

Stopword Removal:
- Implemented stopword filtering to eliminate common, low-value words, improving the quality and relevance of frequency-based text analysis.

NLP Environment Configuration:
- Configured SSL settings and downloaded required NLTK resources to ensure compatibility and smooth execution across different systems.

<b> Part 2 - International Media Text Preprocessing & Tokenization (NLP Preparation) </b>

Code Overview:
- This code prepares international news article text for natural language processing (NLP) analysis. Using Pandas, NLTK, and regular expressions, the program loads article text from an international media CSV file, cleans and normalizes the text, removes URLs and irrelevant characters, filters out common stopwords, and tokenizes the text into individual words. The processed tokens are then ready for unigram, bigram, and trigram frequency analysis.

```python
import pandas
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ssl

inputdata={}
inputdata = pandas.read_csv('int_data.csv', header=[0], index_col=0).to_dict()

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

<ins> Skills Developed – </ins>

International Text Processing:
- Learned how to process and analyze text data from non-U.S. media sources while maintaining a consistent NLP workflow.

Text Cleaning and Normalization:
- Applied standardized cleaning techniques to reduce noise and ensure fair comparison between U.S. and international text datasets.

Tokenization and Stopword Filtering:
- Used NLTK tools to tokenize text and remove common words, improving the quality of frequency-based text analysis.

Comparative NLP Readiness:
- Prepared international media text in the same format as U.S. media data, enabling accurate cross-regional comparisons in unigram, bigram, and trigram analysis.

NLP Resource Management:
- Configured SSL settings and downloaded required NLTK datasets to ensure smooth execution across different environments.

<b> Part 3 - U.S. Media N-Gram Frequency Analysis (Top Unigrams, Bigrams, Trigrams) </b>

Code Overview:
- This code performs n-gram analysis on U.S.-based news article text stored in us_data.csv. After loading the dataset with Pandas, the program cleans and normalizes the text by lowercasing, removing URLs and unwanted characters, and filtering out stopwords using NLTK. The cleaned text is then tokenized, and unigrams, bigrams, and trigrams are generated using nltk.util.ngrams. Finally, the code uses Counter to calculate frequency counts and prints the top 10 most common unigrams, bigrams, and trigrams.

```python
import pandas
import re
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
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

unigrams = ngrams(text_tokens_without_stopwords, 1)
bigrams = ngrams(text_tokens_without_stopwords,2)
trigrams = ngrams(text_tokens_without_stopwords,3)

mostcommonunigrams = Counter(unigrams)
print(mostcommonunigrams.most_common(10))

mostcommonbigrams = Counter(bigrams)
print(mostcommonbigrams.most_common(10))

mostcommontrigrams = Counter(trigrams)
print(mostcommontrigrams.most_common(10))
```

<ins> Skills Developed – </ins>

NLP Preprocessing Pipeline:
- Reinforced a complete workflow for text analysis, including cleaning, normalization, tokenization, and stopword removal—steps that improve the accuracy of n-gram results.

N-Gram Generation (Context Building):
- Learned how unigrams capture individual word frequency, while bigrams and trigrams capture short phrases that provide more context and meaning from the text.

Frequency Counting with Counter:
- Used Python’s collections.Counter to efficiently count word and phrase occurrences and extract the most common patterns in the dataset.

Comparative Text Analytics Readiness:
- Prepared outputs that can be directly used for histogram visualizations and comparisons against international news sources in later questions.

<b> Part 4 - International Media N-Gram Frequency Analysis (Top Unigrams, Bigrams, Trigrams) </b>

Code Overview:
- This code performs n-gram frequency analysis on international news articles stored in int_data.csv. The program cleans and normalizes the text by converting it to lowercase, removing URLs and irrelevant characters, and filtering out stopwords. After tokenization, unigrams, bigrams, and trigrams are generated using NLTK. The Counter class is then used to calculate and print the top 10 most frequent unigrams, bigrams, and trigrams, enabling comparison with U.S. media coverage.

```python
import pandas
import re
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import ssl

# created a dictionary here titled inputdata
inputdata={}
#I am assigning the content of the csv file to my dictionary
#header is my row in the csv file that is why header is 0 below
inputdata = pandas.read_csv('us_data.csv', header=[0], index_col=0).to_dict()

# I created a new dictionary here for the description column in my csv file
textdictionary = inputdata.get('text')
#print(type(textdictionary))

# I am converting the dictionary to a list so I can analyze the data
textlist =  list(textdictionary.values())

#convert list to string
# need the data in string format for analysis purposes
textinstring = ''
for eachletter in  textlist:
    textinstring += ' '+ str(eachletter)

#Make the string lower case
lowercasetext=textinstring.lower()
#print(lowercasetext)

#remove the url from text to prevent a future error
lowercasetext= re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', lowercasetext)

#remove anything that does not make sense to you from the string
lowercasetext = lowercasetext.replace(".", "")
lowercasetext = lowercasetext.replace("#", "")
lowercasetext = lowercasetext.replace(",", "")
lowercasetext = lowercasetext.replace("\\", "")
lowercasetext = lowercasetext.replace("(", "")
lowercasetext = lowercasetext.replace(")", "")
lowercasetext = lowercasetext.replace("+", "")
lowercasetext = lowercasetext.replace("!", "")
lowercasetext = lowercasetext.replace("&", "")

#remove the stop or common words from the string
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

#print(text_tokens_without_stopwords)

unigrams = ngrams(text_tokens_without_stopwords, 1)

bigrams = ngrams(text_tokens_without_stopwords,2)
trigrams = ngrams(text_tokens_without_stopwords,3)

mostcommonunigrams = Counter(unigrams)
#print(type(mostcommonunigrams))
#This will print top 10 unigrams
print(mostcommonunigrams.most_common(10))

#This will print top 10 bigrams
mostcommonbigrams = Counter(bigrams)
print(mostcommonbigrams.most_common(10))

#This will print top 10 trigrams
mostcommontrigrams = Counter(trigrams)
print(mostcommontrigrams.most_common(10))
```

<ins> Skills Developed – </ins>

Cross-Regional Text Analytics:
- Applied the same n-gram analysis workflow to international media data, ensuring consistency and comparability with U.S. news analysis.

Contextual Phrase Extraction:
- Used bigrams and trigrams to capture short phrases that reveal narrative framing and topic emphasis in international media.

Frequency-Based Pattern Recognition:
- Identified dominant words and phrases by calculating occurrence counts, supporting deeper media comparison insights.

Reproducible NLP Workflow:
- Built a repeatable pipeline that can be applied to multiple datasets with minimal modification.

## 🔗 Question 3 – Bigram & Trigram Analysis
Question Overview
- This section expanded the analysis to include bigrams and trigrams, visualized through histograms for both U.S. and international sources. Additionally, stemming and lemmatization techniques were compared to evaluate their impact on text patterns.

<b> Part 1 - 

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
