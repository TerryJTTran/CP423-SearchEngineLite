"""
This project aims to build a basic search engine including different fundamental components we
talked about them for building up Indexing and Query Processing pipelines. The search engine starts from
command line using “python searchengine.py”. Then, the script shows following options, and the user
selects an option for doing related task.

1- Collect new documents.
2- Index documents.
3- Search for a query.
4- Train ML classifier.
5- Predict a link.
6- Your story!
7- Exit

"""


#Imports
import sys
import os
import re
import pandas as pd
import numpy as np
import nltk
import string
import json
import joblib
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import hashlib
import datetime
import requests
import warnings
from joblib import dump
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

#pip install bs4
from bs4 import BeautifulSoup

#Constants
TOPICS = ['Technology', 'Health', 'Entertainment']
STORYFILENAME = "story.txt"
INVERTEDINDEXFILENAME = "invertedindex.txt"
MAPFILENAME = "mapping.txt"

#Global Variables
optionSelected = 1
optionValid = False

# Clear screen wait 
def afterPrint():
    
    # Wait for key press
    waste = input("\nPress the ENTER key to return to options menu...")  
    
    # Clear screen
    os.system('clear')
    return

# Check that a valid option was entered
def UserOption():
    global optionSelected, optionValid

    # If option is non numeric or less than 1 or greater than 7 
    if optionSelected.isnumeric() == False or int(optionSelected) < 1 or int(optionSelected) > 7:
        print("Error in option entered, ensure that the format follows:\npython searchengine.py \'1-7\'\n\nOptions:\n1- Collect new documents.\n2- Index documents.\n3- Search for a query.\n4- Train ML classifier.\n5- Predict a link.\n6- Your story!\n7- Exit")
        print("\nPress any key to return to options menu...")
        msvcrt.getch()

    # Else valid input is entered
    else:
        optionSelected = int(optionSelected)
        optionValid = True

# Collect documents from source links file
def collect_documents():
    print("Collecting Documents...\nIf being run for the first time expect collection time of 12 minutes.")
    global TOPICS

    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.mkdir("data")
    
    # Create subdirectories for each topic if they don't exist
    for topic in TOPICS:
        if not os.path.exists("data/" + topic):
            os.mkdir("data/" + topic)

    # Create mapping txt file
    mapFile = open(MAPFILENAME, "w", encoding = "UTF-8")
    mapFile.close()

    max_depth = 1
    
    # Read from sources.txt
    with open('sources.txt', 'r', encoding = "UTF-8") as f:
        for line in f:
            
            # Split line into topic and link
            topic, link = line.strip().split(',')
            
            # Hash URL
            url_hash = hashlib.md5(link.encode('utf-8')).hexdigest()
            
            # Check if page has already been crawled
            if os.path.exists(f'data/{topic}/{url_hash}.txt'):
                continue
           
            # Crawl page
            try:
                crawl_link(link, topic, max_depth, url_hash)

            except requests.exceptions.RequestException:
                pass

    warnings.filterwarnings("default")

    # Populate hashid to docID dictonary
    docIDsDict = {}
    with open("crawl.log", "r", encoding="utf-8") as f:
        lines = f.readlines()
        count = 1
        for line in lines:
            lineAttributes = line.split(", ")
            docIDsDict[str(lineAttributes[2] + ".txt")] = "H" + str(count)
            count += 1

    # Dump dictonary into map file
    with open(MAPFILENAME, "w", encoding="utf-8") as f:
        json.dump(docIDsDict, f)
        
    print("\nCollection complete.")

# Crawl through the link, collect contents, and save contents to hashed url file
def crawl_link(link, topic, max_depth, url_hash):
    warnings.filterwarnings("ignore")
    response = requests.get(link)
    response.raise_for_status()

    translator = str.maketrans('', '', string.punctuation)

    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.translate(translator) 
    
    
    # Remove stopwords from content
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    
    # Save page content in topic related subfolder
    with open(f'data/{topic}/{url_hash}.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    
    # Write to crawl.log file
    with open('crawl.log', 'a', encoding='utf-8') as f:
        f.write(f'{topic}, {link}, {url_hash}, {datetime.datetime.now()}\n')
    
    # Crawl links on page up to a maximum depth
    if max_depth > 0:
        for links in soup.find_all('a'):
            href = links.get('href')
            
            linkParts = link.split("/")
            linkStart = "https://"+ linkParts[2]

            # Check if link exist and if it has intial link in it
            if href is not None and href.startswith(linkStart):
                # Hash URL
                link_hash = hashlib.md5(href.encode('utf-8')).hexdigest()
                
                # Check if link has already been crawled
                if not os.path.exists(f'data/{topic}/{link_hash}.txt'):
                    
                    # Crawl link
                    try:
                        response = requests.get(href)
                        response.raise_for_status()
                        
                        soup = BeautifulSoup(response.content, 'html.parser')
                        text = soup.get_text()
                        text = re.sub(r'\s+', ' ', text).strip()
                        text = text.translate(translator) 
                        
                        
                        if text is not None:

                            # Remove stopwords from content
                            text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
                            
                            # Save page content in topic related subfolder
                            with open(f'data/{topic}/{link_hash}.txt', 'w', encoding='utf-8') as f:
                                f.write(text)
                            
                            # Write to crawl.log file
                            with open('crawl.log', 'a') as f:
                                f.write(f'{topic}, {href}, {link_hash}, {datetime.datetime.now()}\n')
                            
                            # Recursively crawl links up to max_depth
                            if max_depth > 1:
                                crawl_link(href, topic, max_depth-1, link_hash)

                    except requests.exceptions.RequestException:
                        pass
                    except:
                        pass

# Calculate soundex
def soundex(word):
    # remove all non-alphabetic characters and convert to uppercase
    word = re.sub(r'[^A-Za-z]+', '', word.upper())
    
    # handle empty string or strings with only one character
    if not word:
        return word
    
    # remove select characters
    removeCode = str.maketrans('', '', 'AEIOUYHW')
    wordCode = word[1:].translate(removeCode)

    # map the first character to itself and the rest to their corresponding digits
    soundex_code = word[0]
    digit_map = str.maketrans('BFPVCGJKQSXZDTLMNR', '111122222222334556')
    soundex_code += word[1:].translate(digit_map)
    
    # remove consecutive duplicates and all zeros except the first one
    soundex_code = re.sub(r'(\d)\1+', r'\1', soundex_code)
    soundex_code = re.sub(r'0', '', soundex_code)
    
    # pad the code with zeros or truncate it to length 4
    soundex_code = soundex_code + '000'
    return soundex_code[:4]

# Create inverted index using downloaded page and save it as invertedindex.txt
def index_documents():
    print("Indexing Documents...\nExpect indexing time of 5 minutes.")
    
    # initialize the inverted index dictionary
    inverted_index = {}

    # Initalize docID dictonary
    with open(MAPFILENAME, "r", encoding="utf-8") as f:
        docIDDict = json.load(f)

    # loop through each file in the "data" folder
    for topic in TOPICS:
        for filename in os.listdir("data/" + str(topic)):
            with open(os.path.join("data/" + str(topic), filename), "r", encoding = "UTF-8") as f:
                text = f.read()
                
                # Turn text to lower case
                text = text.lower()
                
                # Tokenize text
                tokenizedText = word_tokenize(text)
                
                # Remove stopwords
                stop_words = set(stopwords.words('english'))
                tokenizedText = [word for word in tokenizedText if word not in stop_words]
                
                # Stem text
                stemmer = PorterStemmer()
                tokenizedText = [stemmer.stem(word) for word in tokenizedText]
                

                # loop through each term in the text
                for term in set(tokenizedText):
                    
                    # add the term to the inverted index dictionary
                    if term not in inverted_index:
                        inverted_index[term] = []
                    
                    # update the corresponding document identifier and frequency
                    inverted_index[term].append((docIDDict[filename], tokenizedText.count(term)))

    # write the inverted index to a file
    with open("invertedindex.txt", "w", encoding = "UTF-8") as f:
        f.write("| Term | Soundex | Appearances (DocID, Frequency) |\n")
        f.write("|------|---------|--------------------------------|\n")
        for term in sorted(inverted_index.keys()):
            # Check if word contains a number surrounded by letters at all (considered as not real string)
            pattern = re.compile(r'\b\w*[a-zA-Z]+\d+\w*[a-zA-Z]+\w*\b')
            matchingWord = pattern.findall(term)
            if (len(matchingWord) == 0):
                # we can use Soundex algorithm to calculate the Soundex code for each term
                soundexCode = soundex(term)
            
                # we can use a mapping file to store the mapping between document identifier and DocID
                appearances = "".join("({}, {})|".format(docid, freq) for (docid, freq) in inverted_index[term])
                f.write("|{}|{}|{}\n".format(term, soundexCode, appearances))

# Search for the 3 most related documents based on the query
def search_query():
    
    query = input("Enter query: ")


    # If option is non numeric or less than 1 or greater than 7 
    if query.replace(' ', '').isalnum() == True:
        words = {}
        sound = {}
        
        # Set up dictonary with terms as keys and docID and frequency as values
        with open("invertedindex.txt", "r", encoding = "UTF-8") as f:
            lines = f.readlines()[2:]
            for line in lines:
                line = line.strip()
                line = line.split('|')
                words[line[1]] = line[3:-1]

                # Add soundex as key and the term as value
                if line[2] not in sound:
                    sound[line[2]] = [line[1]]

                # Soundex already added, add new term with same soundex
                else:
                    sound[line[2]].append(line[1])
        
        

        # Turn text to lower case
        query = [query.lower()]
        
        
        frequencyTotal = {}
        
        print("Searching for a query...\nExpect searching time of 1 minutes.")
        
        # finds every document where word occurs at least once
        for word in query:
            
            # If word is not in index add soundex
            if word not in words:
                soundexCode = soundex(word)
                
                # Check if soundex already exists
                if soundexCode in sound:
                    # Retrieve terms with soundex
                    soundexTerms = sound[soundexCode]

                    maxTerm = ""
                    maxDocs = 0
                            

                    # Search for number of documents with term
                    for term in soundexTerms:
                        if term in words:
                            
                            # Retrieve documents and total frequency
                            termInfo = words[term]
                            
                            if len(termInfo) > maxDocs:
                                maxDocs = len(termInfo)
                                maxTerm = term
                    
                    # Add best word replace to frequencytotal
                    if maxTerm in words:
                        # Retrieve documents and total frequency
                        termInfo = words[maxTerm]
                        
                        # Add frequency to respective docID in dictonary
                        for values in termInfo:
                            docID, Freq = values.split(", ")
                            docID = docID[1:]
                            Freq = Freq[:-1]
                            if docID in frequencyTotal:
                                frequencyTotal[docID] += int(Freq)
                            
                            else:
                                frequencyTotal[docID] = int(Freq)

            # Word is in index
            elif word in words:
                
                # Retrieve documents and total frequency
                termInfo = words[word]

                # Add frequency to respective docID in dictonary
                for values in termInfo:
                    docID, Freq = values.split(", ")
                    docID = docID[1:]
                    Freq = Freq[:-1]
                    if docID in frequencyTotal:
                        frequencyTotal[docID] += int(Freq)
                    
                    else:
                        frequencyTotal[docID] = int(Freq)
        
        
        # Load mapping
        with open("mapping.txt", "r", encoding = "UTF-8") as f:
            mapDict = json.load(f)

        docHashFiles = []

        # Retrieve corresponding hashed file names for docIDs
        for hashFiles in frequencyTotal.keys():
            
            # Retrieve keys with corresponding docID value
            keyFile = [k for k, v in mapDict.items() if v == hashFiles]
            docHashFiles.append(keyFile[0])
    
        
    
        similarities = {}

        # Retrieve contents of selected files, vectorize and compare
        for file in docHashFiles:
            # Check each topic
            for topic in TOPICS:

                fileExistance = os.path.isfile(f'data/{topic}/{file}')
                
                # Check if file exist in folder
                if (fileExistance == True):
                    
                    # Open file, read contents, tokenize, remove stopwords, vectorize, and compare
                    with open(f'data/{topic}/{file}', "r", encoding="utf-8") as f:
                        text = f.read()

                        # X holds the text contents, Y holds the topic the text is associated with
                        X = []
                        Y = []
                        X.append(text)
                        Y.append(topic)

                        # Vectorize and remove stop words from the docs and query 
                        vectorizer = TfidfVectorizer(stop_words='english')
                        docVector = vectorizer.fit_transform(X).toarray()
                        queryVector = vectorizer.transform(query).toarray()

                        # Calculate cosine similarity
                        cos_sim = cosine_similarity(docVector, queryVector)
                        similarities[file] = cos_sim


        # Check similarities for top 3 scores
        topKeys = sorted(similarities, key=similarities.get, reverse=True)[:3]    


        print("Top 3 most related documents:\n")

        # Grab urls for hash in crawl.log
        for key in topKeys:
            hashFile = key[:-4]

            with open("crawl.log", "r", encoding="utf-8") as f:
                lines = f.readlines()
                
                # Search for correct line that contains the actual url of the document
                for line in lines:
                    if hashFile in line:
                        crawlInfo = line.split(", ")
                        print(crawlInfo[1] + " : " + str(similarities[key]))

    # If invalid query is sent
    else:
        print("Error in input for query.\nReturning to main menu...\n")

# Use SVC model to train using the document and topic data, then display various performance metrics
def train_classifier():
    print("Training classifier...")
    # X holds the text contents, Y holds the topic the text is associated with
    X = []
    Y = []
    
    # loop through each file in the "data" folder
    for topic in TOPICS:
        for filename in os.listdir("data/" + str(topic)):
            with open(os.path.join("data/" + str(topic), filename), "r", encoding = "UTF-8") as f:
                text = f.read()
            X.append(text)
            Y.append(topic)



    # Vectorize and remove stop words from the docs
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train and fit the classifier
    clf = SVC(probability=True) # 1
    #clf = MultinomialNB() # 2
    #clf = KNeighborsClassifier() # 5
    #clf = RandomForestClassifier() # 3
    #clf = DecisionTreeClassifier() # 4
    clf.fit(X_train, Y_train)

    # Save the classifier as classifier.model and vectorizer
    joblib.dump(clf, "classifier.model")
    joblib.dump(vectorizer, "vectorizer.joblib")

    # Print the training results
    Y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred, average="macro")
    precision = precision_score(Y_test, Y_pred, average="macro")
    f1 = f1_score(Y_test, Y_pred, average="macro")
    matrix = confusion_matrix(Y_test, Y_pred)
    print("Performance of SV Classification:\nAccuracy: {:.3f}\nRecall: {:.3f}\nPrecision: {:.3f}\nF1-score: {:.3f}\nConfusion Matrix:\n{}".format(accuracy, recall, precision, f1, matrix))

# Use saved model and classifier to predict topic of user inputted link
def predict_link():
    predictLink = input("Enter the link you would like to predict it's topic (Options - Technology, Health, Entertainment): ")
    try:
        response = requests.get(predictLink)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Load the saved classifier
        with open("classifier.model", 'rb') as file:
            clf = joblib.load(file)
        
        # Load the saved vectorizer
        with open("vectorizer.joblib", 'rb') as file:
            vectorizer = joblib.load(file)

        # Apply vectorizer used in model to inputted text
        input_vector = vectorizer.transform([text])

        # Predict the label for the input text
        label_Pred = clf.predict(input_vector)
        confidence_Level = clf.predict_proba(input_vector)
        typeIndex = 0
        if (label_Pred == "Technology"):
            typeIndex = 2
        elif (label_Pred == "Health"):
            typeIndex = 1

        # Print predicted label
        print(f"\n<{label_Pred[0]}, {confidence_Level[0][typeIndex]:.3f}>")

    except:
        print("\nError in link entered...")

# Read story file and print to screen
def user_story():
    # Clear screen
    os.system('clear')
    # Save page content in topic related subfolder
    with open(STORYFILENAME, 'r', encoding='utf-8') as f:
        for line in f:
            print(line, end="")


# Run program main
def run():
    global optionSelected, optionValid
    while optionSelected != 7:

        # Clear screen
        os.system('clear')
        
        # Ask user for option
        optionSelected = input("Options:\n1- Collect new documents.\n2- Index documents.\n3- Search for a query.\n4- Train ML classifier.\n5- Predict a link.\n6- Your story!\n7- Exit\n\nEnter your choice: ")

        # Check option selected
        UserOption()

        # If option enter valid
        if optionValid == True:

            # Collect new documents selected
            if optionSelected == 1:
                collect_documents()
                afterPrint()

            # Index documents selected
            elif optionSelected == 2:
                index_documents()
                afterPrint()

            # Search for a query selected
            elif optionSelected == 3:
                search_query()
                afterPrint()

            # Train ML classifer selected
            elif optionSelected == 4:
                train_classifier()
                afterPrint()

            # Predict a link selected
            elif optionSelected == 5:
                predict_link()
                afterPrint()

            # Story option selected
            elif optionSelected == 6:
                user_story()
                afterPrint()

            # Exit option selected
            elif optionSelected == 7:
                print("Exiting...")
                



if __name__ == '__main__':
    run()