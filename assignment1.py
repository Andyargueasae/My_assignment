# This is the file you will need to edit in order to complete assignment 1
# You may create additional functions, but all code must be contained within this file


# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items
import numpy as np
from numpy import arange
import pandas as pd
from matplotlib import pyplot as plt
import json
import csv
import os
import re
import seaborn as sns
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from numpy.linalg import norm

# You should use these two variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'
nltk.download('punkt')
nltk.download('stopwords')

def task1():
    #Complete task 1 here
    with open(datafilepath) as fjson:
        data=json.load(fjson)
    list_tc = data['teams_codes']
    list_tc = sorted(list_tc)
    #return the list of teamcodes from the collection of clubs.
    return list_tc
    
def task2():
    #Complete task 2 here
    '''
    Output the csv file that contains the goals scored and lost by each club.
    '''
    with open(datafilepath) as fjson:
        data=json.load(fjson)
    list_tc = data['teams_codes']
    list_tc = sorted(list_tc)
    clubs = data['clubs']
    goalscore = []
    goalconc = []
    
    #iterate over the team information and store their goals.
    for i in list_tc:
        for j in clubs:
            if j['club_code'] == i:
                goalscore.append(j["goals_scored"])
                goalconc.append(j["goals_conceded"])
   
    #Now write them in csv files.
    t2_out = open("task2.csv", 'w', newline = '')
    output = csv.writer(t2_out)
    headings = ["team_code","goals_scored_by_team", 
                "goals_scored_against_team"]
    output.writerow(headings)
    for i in range(0, len(list_tc)):
        row = [list_tc[i], goalscore[i], goalconc[i]]
        output.writerow(row)
    t2_out.close()
    #Done.
    return 
      
def task3():
    #Complete task 3 here
    '''
    Find the soccer score matches from a collection of articles.
    '''
    #iterate the data folder, find the maximum score for each file. 
    dirs = os.listdir(articlespath)
    file_score = []
    dirs = sorted(dirs)
    for i in dirs:
        goals = find_goal(i)
        file_score.append(goals)
    
    #Write them in the csv file.
    task3_out = open("task3.csv", "w", newline ='')
    t3 = csv.writer(task3_out)
    headings = ['filename', 'total goals']
    t3.writerow(headings)
    for i in range(0, len(file_score)):
        rows = [dirs[i], file_score[i]]
        t3.writerow(rows)
    task3_out.close()
    return

def find_goal(filename):
    '''
    Using the regular expresion, find the maximum sum of goals
    in each file.
    '''
    file = open(articlespath + '/' + filename, 'r')
    strings = file.read()
    match = r'\D[0-9]{1,2}\-[0-9]{1,2}'
    scores = re.findall(match, strings)
    if len(scores) == 0:
        return 0
    sum_sc = []
    for i in scores:
        i = i.split('-')
        count = 0
        for j in i:
            count += int(j)
        sum_sc.append(count)
        count = 0
    file.close()
    return max(sum_sc)

def task4():
    #Complete task 4 here
    #store the goals sum for each file.
    dirs = os.listdir(articlespath)
    file_score = []
    for i in dirs:
        goals = find_goal(i)
        file_score.append(goals)
    
    #plot the boxplot and set the labels, titles for axis.
    plt.subplots(figsize = (9,12))
    plt.boxplot(file_score)
    plt.ylabel("Total goals")
    plt.xlabel("All texts")
    plt.title("Total goals distribution from the texts")
    plt.savefig("task4.png", dpi=300)
    plt.show()
    plt.close()
    return

def task5():
    '''
    returns a graph summarizing the total mentions of club by media.
    '''
    #Complete task 5 here
    #Store the names for each club first.
    with open(datafilepath) as fjson:
        data=json.load(fjson)
    list_tn = data["participating_clubs"]
    list_tn = sorted(list_tn)
    clubdic = {}
    for i in list_tn:
        clubdic[i] = 0
    #count the number of mentions for each club.
    dirs = os.listdir(articlespath)
    for i in clubdic.keys():
        for j in dirs:
            count_mention(j, i, clubdic)
    
    #Now write them in the csv file.
    task5_out = open("task5.csv", "w", newline = '')
    t5 = csv.writer(task5_out)
    headings = ["club_name", "number_of_mentions"]
    t5.writerow(headings)
    for i in list_tn:
        rows = [i, clubdic[i]]
        t5.writerow(rows)
    task5_out.close()
    
    #now produce the bar chart of the mentions.
    team_mention = clubdic.values()
    plt.subplots(figsize = (25,16))
    plt.bar(arange(len(team_mention)), team_mention)
    plt.xticks(arange(len(list_tn)), list_tn, rotation = 45, fontsize = 13)
    plt.ylabel("The number of mentions", fontsize = 25)
    plt.xlabel("Club Names", fontsize = 15)
    plt.title("The number of mentions of each team by media", 
              fontdict = {'weight':'normal','size':25})
    plt.savefig("task5.png", dpi = 300)
    plt.show()
    plt.close()
    return

def count_mention(filename, pattern, Dict):
    #individually count the mention of one club in articles.
    file = open(articlespath+'/'+filename, 'r')
    strings = file.read()
    findings = re.findall(pattern, strings)
    if pattern in findings:
        Dict[pattern] += 1
    return

def task6():
    #Complete task 6 here
    #Read in files first.
    '''
    Produces a heatmap that analyzes the co-existance 
    between clubs in articles.
    '''
    dirs = os.listdir(articlespath)
    file = open("task5.csv", 'r')
    read = csv.reader(file)
    head = next(read)
    data = list(read)
    #store the names of clubs.
    name_list = []
    for i in data:
        name_list.append(i[0])
    sims = {}
    dirs = os.listdir(articlespath)
    
    #pairing those clubs and calculate the similarity scores of the pair.
    for i in range(len(data)):
        sims[data[i][0]] = []
        for j in range(len(data)):
            interact = co_check(data[i][0], data[j][0], dirs)
            similarity = calculate_sim_score(data[i], data[j], interact)
            sims[data[i][0]].append(similarity)
        
    #Make the dataframe for producing the heatmap.
    sim_data = pd.DataFrame(sims, index = name_list)
    #set the size and write in information.
    plt.subplots(figsize = (18,15))
    sns.heatmap(sim_data, cmap = 'rocket_r', xticklabels=True)
    plt.xticks(fontsize = 13, rotation = 60)
    plt.yticks(fontsize = 14)
    plt.savefig("task6.png", dpi = 500)
    plt.show()
    plt.close()
    return

def co_check(pattern1, pattern2, filenames):
    #count the number of articles both mentioning two clubs.
    co_exist = 0
    for i in filenames:
        element = open(articlespath+'/'+i, 'r')
        strings = element.read()
        p1_scores = re.findall(pattern1, strings)
        p2_scores = re.findall(pattern2, strings)
        if (pattern1 in p1_scores) and (pattern2 in p2_scores):
            co_exist += 1
        element.close()
            
    return co_exist

def calculate_sim_score(list1, list2, shared):
    #calculate similarity scores for a pair.
    score = 0
    denom = int(list1[-1])+int(list2[-1])
    if denom == 0:
        return 0
    else:    
        score = ((2*shared)/(denom))
    return score

def task7():
    '''
    Produces a scatterplot that visualizes the relationship
    between mentions and scores of club.
    '''
    #Complete task 7 here
    #extract the goals and mentions from preserved files.
    data_goal = open("task2.csv", 'r')
    data_mention = open("task5.csv", 'r')
    goal = csv.reader(data_goal)
    mention = csv.reader(data_mention)
    head_g = next(goal)
    goals = list(goal)
    head_m = next(mention)
    mentions = list(mention)
    data_goal.close()
    data_mention.close()
    
    #store the information in list and plot.
    goal_list = [int(i[1]) for i in goals]
    mention_list = [int(j[-1]) for j in mentions]
    plt.scatter(goal_list, mention_list,color = "red", alpha = 0.6)
    plt.xlim(0,14)
    plt.ylim(0, 100)
    #show the title, x/y axis and grids.
    plt.title("Relationship of frequency of mention and goals scored by club")
    plt.ylabel("number of articles mentioning club")
    plt.xlabel("number of goals scored by club")
    plt.grid(True)
    plt.savefig("task7.png", dpi = 500)
    plt.show()
    plt.close()
    return
    
def task8(filename):
    #Complete task 8 here
    '''
    The function returns a list of words in a text file, with 
    non-alphabetic items removed and stopwords cleared.
    '''
    file = open(filename, 'r')
    strings = file.read()
    #ensure only white spaces in between.
    for i in strings:
        if i.isdigit():
            strings = strings.replace(i, ' ')
    strings = re.sub(r"[^a-zA-Z]", ' ', strings)
    limit = 127
    #If there are any strings which are misread as other characters, delete it.
    for i in strings:
        if ord(i) > limit:
            strings = re.sub(i, ' ', strings)
    strings = strings.lower()
    strings = ' '.join(strings.split())
    #divide the strings into tokens, words bag created.
    words = word_tokenize(strings)
    words = pre_processing(words)
    #Remove stopwords.
    stopWords = set(stopwords.words('english'))
    stopWords.add('cannot')
    remaining = [i for i in words if not i in stopWords]
    #remove single-character strings.
    newlst = [i for i in remaining if len(i) > 1]
    file.close()
    #done.
    return newlst

def pre_processing(wordlist):
    #this is the function to remove any items that are non-alphabetic.
    newlist = []
    for i in wordlist:
        if i.isalpha():
            newlist.append(i)
    return newlist

def task9():
    #Complete task 9 here
    '''
    The function returns a rank of top 10 article pairs by their similarity
    scores.
    '''
    dirs = os.listdir(articlespath)
    dirs = sorted(dirs)
    wordbags = {}
    all_words = []
    #Iterate over the folders and create a sum wordbags.
    for i in dirs:
        filename = articlespath+'/'+i
        elemental = task8(filename)
        results = count_dict(elemental)
        all_words+=elemental
        wordbags[i] = results
    
    #based on the wordbags, create the list of raw counts for each file.
    all_words = set(all_words)
    keys = list(wordbags.keys())
    vals = list(wordbags.values())
    raw_count = construct_frequency(all_words, vals)
    #then, produce the tfidf table.
    docs = calculate_tfidf(raw_count)
    rows = []
    
    #compare two files and do the calculation.
    for i in range(0, len(keys)):
        for j in range(i+1, len(keys)):
            sim_score = cos_sim(docs[i], docs[j])
            rows.append([keys[i], keys[j], sim_score])
    #sort the rows in descending order.
    rows = sorted(rows, key = lambda x:x[-1], reverse = True)
    #write the files.
    file = open("task9.csv", 'w', newline = '')
    task9_out = csv.writer(file)
    heading = ['article1', 'article2', 'similarity']
    task9_out.writerow(heading)
    top_ten = 10
    for i in range(0, top_ten):
        task9_out.writerow(rows[i])
    file.close()
    #done.
    return

def construct_frequency(word_bank, values):
    '''Make a list of raw counts for each words in wordbag, 
    stored for each file.'''
    tfreq = []
    for i in values:
        file_count = []
        file_words = list(i.keys())
        for ws in word_bank:
            if ws not in file_words:
                file_count.append(0)
            else:
                file_count.append(i[ws])
        tfreq.append(file_count)
        
    return tfreq
    
def calculate_tfidf(raw_tf):
    #return an array of tfidf values.
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(raw_tf)
    docs_arr = tfidf.toarray()
    return docs_arr

def count_dict(word_bag):
    #count the presence of words in each file.
    wordDict = {}
    for word in word_bag:
        if word not in wordDict:
            wordDict[word] = 1
        else:
            wordDict[word] += 1
    return wordDict

def cos_sim(v1, v2):
    #return the cosine similarity for a pair.
    result = np.dot(v1, v2)/(norm(v1)*norm(v2))
    return format(result, '.16f')
