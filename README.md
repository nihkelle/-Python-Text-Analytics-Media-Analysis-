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

<b> Skills Developed </b>

Media Literacy & Source Evaluation:
- Developed the ability to identify, compare, and select credible U.S. and international news sources while maintaining topic consistency across multiple media outlets.

Non-Coding Data Collection Techniques:
- Gained experience using non-coding scraping tools to collect structured article data, including titles, authors, publication dates, and full text, in compliance with assignment guidelines.

Data Organization & Structuring:
- Learned how to organize raw scraped data into well-structured CSV files, enabling efficient downstream analysis using Python and NLP libraries.

Dataset Integration:
- Strengthened skills in combining multiple datasets from different media sources into unified U.S. and international datasets suitable for comparative analysis.

Data Cleaning & Quality Control:
- Practiced identifying and removing redundant or unnecessary data (such as duplicate index columns) to ensure consistency, accuracy, and reusability.

Analytical Readiness:
- Prepared clean, structured datasets that serve as the foundation for text preprocessing, n-gram analysis, and visualization tasks in later questions.


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

<b> Part 5 - U.S. Media Unigram Visualization & N-Gram Analysis </b>

Code Overview:
- This code extends U.S. media text analysis by combining n-gram frequency extraction with data visualization. After cleaning, tokenizing, and removing stopwords from the text in us_data.csv, the program calculates the top unigrams, bigrams, and trigrams. The top 10 unigrams are then visualized using a bar chart created with Matplotlib, allowing for a clear visual representation of word frequency in U.S. media articles.

```python
import pandas
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import ssl


#I am creating a dictionary here titled inputdata
inputdata={}
#I am assigning the content of the csv file to my dictionary
#header is my row in the csv file that is why header is 0 below
inputdata = pandas.read_csv('us_data.csv', header=[0], index_col=0).to_dict()

#We can use type to check the data type of a variable
#print(type(inputdata))

#I am using the column headers from the csv file to find the data I am interested to analyze

# I created a new dictionary here for the description column in my csv file
textdictionary = inputdata.get('text')
#print(type(textdictionary))

# I am converting the dictionary to a list so I can analyze the data
textlist =  list(textdictionary.values())
#print(type(titlelist))

#convert list to string
#I need the data in string format for analysis purposes
textinstring = ''
for eachletter in  textlist:
    textinstring += ' '+ str(eachletter)
lowercasetext=textinstring.lower()
#print(lowercasetext)

#remove the url from text to prevent a future error
lowercasetext= re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', lowercasetext)
#print(lowercasedescriptions)

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
#print(stopwords.words('english'))

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


#This will print top 3 bigrams
mostcommonbigrams = Counter(bigrams)
print(mostcommonbigrams.most_common(10))

#This will print top 3 trigrams
mostcommontrigrams = Counter(trigrams)
print(mostcommontrigrams.most_common(10))

top3unigrams = mostcommonunigrams.most_common(10)
top3unigrams_keys = []
top3unigrams_values = []

for i in range(len(top3unigrams)):
    #print(top3unigrams[i][0][0])
    top3unigrams_keys.append(top3unigrams[i][0][0])
    top3unigrams_values.append(top3unigrams[i][1])

#print(top3unigrams_keys)
#print(top3unigrams_values)

import matplotlib.pyplot as plt

plt.bar(top3unigrams_keys, top3unigrams_values)
plt.title("Top 10 Unigrams")
plt.xlabel("Unigrams")
plt.ylabel("Frequency")
plt.show()
```

<ins> Skills Developed – </ins>

Data Visualization:
- Learned how to transform frequency counts into bar charts, making textual patterns easier to interpret and communicate.

Integration of NLP and Visualization:
- Combined NLTK-based text analysis with Matplotlib to create visual insights from unstructured data.

Feature Selection for Analysis:
- Extracted unigram keys and values from frequency results to prepare data for plotting.

Comparative Media Analysis:
- Created outputs that directly support visual and statistical comparison between U.S. and international media coverage.

<b> Part 6 - International Media Unigram Visualization & N-Gram Analysis </b>

Code Overview:
- This code performs n-gram frequency analysis and visualization on international news article text stored in int_data.csv. The program first cleans and preprocesses the text by converting it to lowercase, removing URLs and irrelevant characters, and filtering out common stopwords. After tokenization, unigrams, bigrams, and trigrams are generated using NLTK and counted with the Counter class. The top 10 unigrams are then visualized using a bar chart created with Matplotlib, providing a clear graphical representation of word frequency in international media coverage.

```python
import pandas
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import ssl


#I am creating a dictionary here titled inputdata
inputdata={}
#I am assigning the content of the csv file to my dictionary
#header is my row in the csv file that is why header is 0 below
inputdata = pandas.read_csv('int_data.csv', header=[0], index_col=0).to_dict()

#We can use type to check the data type of a variable
#print(type(inputdata))

#I am using the column headers from the csv file to find the data I am interested to analyze

# I created a new dictionary here for the description column in my csv file
textdictionary = inputdata.get('text')
#print(type(textdictionary))

# I am converting the dictionary to a list so I can analyze the data
textlist =  list(textdictionary.values())
#print(type(titlelist))

#convert list to string
#I need the data in string format for analysis purposes
textinstring = ''
for eachletter in  textlist:
    textinstring += ' '+ str(eachletter)

lowercasetext=textinstring.lower()
#print(lowercasetext)

#remove the url from text to prevent a future error
lowercasetext= re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', lowercasetext)
#print(lowercasedescriptions)

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
else: ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
nltk.download('punkt_tab')
#print(stopwords.words('english'))

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

#Draw a histogram of top 10 unigrams
top3unigrams = mostcommonunigrams.most_common(10)
top3unigrams_keys = []
top3unigrams_values = []

for i in range(len(top3unigrams)):
    #print(top3unigrams[i][0][0])
    top3unigrams_keys.append(top3unigrams[i][0][0])
    top3unigrams_values.append(top3unigrams[i][1])

#print(top3unigrams_keys)
#print(top3unigrams_values)

import matplotlib.pyplot as plt

plt.bar(top3unigrams_keys, top3unigrams_values)
plt.title("Top 10 Unigrams")
plt.xlabel("Unigrams")
plt.ylabel("Frequency")
plt.show()
```

<ins> Skills Developed – </ins>

End-to-End NLP Analysis:
- Implemented a complete natural language processing workflow, from raw text cleaning to frequency analysis and visualization.

N-Gram Modeling:
- Generated unigrams, bigrams, and trigrams to capture both individual word usage and contextual phrase patterns in international news content.

Data Visualization with Matplotlib:
- Learned how to convert textual frequency data into bar charts, making abstract word patterns easier to interpret and communicate.

Comparative Media Readiness:
- Produced visual outputs that can be directly compared with U.S. media histograms to identify similarities and differences in coverage and framing.

Analytical Communication:
- Created interpretable visual evidence that supports narrative explanations in the homework video presentation.

## 🔗 Question 3 – Bigram & Trigram Analysis
Question Overview
- This section expanded the analysis to include bigrams and trigrams, visualized through histograms for both U.S. and international sources. Additionally, stemming and lemmatization techniques were compared to evaluate their impact on text patterns.

<b> Part 1 - International Media Text Cleaning & Token Extraction </b>

Code Overview:
- This code focuses on cleaning and tokenizing international news article text stored in int_data.csv. Using Pandas, NLTK, and regular expressions, the program extracts article text, converts it into a single string, standardizes the text by lowercasing and removing URLs and irrelevant characters, and eliminates common stopwords. The final output is a list of cleaned word tokens that can be used for further NLP tasks such as unigram, bigram, and trigram analysis.

```python
# example 3 for int data
import pandas
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ssl

# created a dictionary here titled inputdata
inputdata={}
# assigned the content of the csv file to my dictionary
#header is my row in the csv file that is why header is 0 below
inputdata = pandas.read_csv('int_data.csv', header=[0], index_col=0).to_dict()

# using the column headers from the csv file to find the data I am interested to analyze

# created a new dictionary here for the description column in my csv file
textdictionary = inputdata.get('text')
#print(type(textdictionary))

# converting the dictionary to a list so I can analyze the data
textlist =  list(textdictionary.values())
#print(type(titlelist))

#convert list to string
# need the data in string format for analysis purposes
textinstring = ''
for eachletter in  textlist:
    textinstring += ' '+ str(eachletter)

# make the string lower case
lowercasetext=textinstring.lower()

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

print(text_tokens_without_stopwords)
```

<ins> Skills Developed – </ins>

Text Extraction from Structured Data:
Learned how to retrieve and isolate relevant text fields from a CSV file for focused analysis.

Text Cleaning and Standardization:
Applied multiple preprocessing steps—including case normalization, URL removal, and character filtering—to reduce noise in international text data.

Tokenization:
Used NLTK’s word tokenizer to split cleaned text into individual words, forming the foundation for frequency and n-gram analysis.

Stopword Removal:
Filtered out commonly used words to improve the relevance and interpretability of text analysis results.

Preparation for Advanced NLP:
Produced clean, structured tokens suitable for downstream analytical tasks such as frequency counting, visualization, and comparative media analysis.

## 🌍 Question 4 – Non-English NLP with NLTK (GenAI-Permitted)
Code Overview

For this GenAI-permitted question, I installed an NLTK language package for a non-English language and analyzed a foreign-language article scraped using non-coding methods. Histograms for unigrams, bigrams, and trigrams were generated and compared to English-language results from earlier questions.

Skills Developed –

Multilingual NLP: Installed and used a non-English NLTK language package.

Cross-Language Comparison: Identified similarities and differences between English and non-English text patterns.

Engagement Analysis: Explored potential relationships between word usage patterns and user engagement metrics such as likes or comments .

## 📧 Question 5 – Email Spam Detection Analysis
Code Overview

In this question, I analyzed 20 spam emails by extracting and processing both the subject lines and body text. Top unigrams, bigrams, and trigrams were calculated to identify common linguistic patterns associated with spam content.

Skills Developed –

Text Classification Foundations: Identified linguistic features commonly associated with spam emails.

Feature Extraction: Compared word and phrase patterns between email titles and body text.

Practical Application: Evaluated how n-gram frequency supports spam detection logic .
