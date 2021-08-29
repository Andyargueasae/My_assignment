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
import math
import seaborn as sns
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfTransformer
from numpy.linalg import norm

# You should use these two variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'

#The task1 is finished!!
def task1():
    #Complete task 1 here
    with open(datafilepath) as fjson:
        data=json.load(fjson)
    list_tc = data['teams_codes']
    list_tc = sorted(list_tc)
    #Done.
    return list_tc
    
def task2():
    #Complete task 2 here
    with open(datafilepath) as fjson:
        data=json.load(fjson)
    list_tc = data['teams_codes']
    list_tc = sorted(list_tc)
    clubs = data['clubs']
    #print(clubs)
    goalscore = []
    goalconc = []
    
    for i in list_tc:
        for j in clubs:
            if j['club_code'] == i:
                goalscore.append(j["goals_scored"])
                goalconc.append(j["goals_conceded"])
   
    #print(goalscore)
    #print(goalconc)
        
    t2_out = open("task2.csv", 'w', newline = '')
    output = csv.writer(t2_out)
    headings = ["team_code","goals_scored_by_team", "goals_scored_against_team"]
    output.writerow(headings)
    for i in range(0, len(list_tc)):
        row = [list_tc[i], goalscore[i], goalconc[i]]
        output.writerow(row)
    t2_out.close()
    #Done.
    return 
      
def task3():
    #Complete task 3 here
    dirs = os.listdir(articlespath)
    #Now you have obtained the filenames dirs in the directory.
    file_score = []
    #print(find_goal(dirs[211]))
    for i in dirs:
        goals = find_goal(i)
        file_score.append(goals)
    
    #print(file_score)
    #os.chdir("E:/2021 SM 2 2021.7--11/assignments/EODP/individual/My_assignment")
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

def task4():       #objective complete!
    #Complete task 4 here
    dirs = os.listdir(articlespath)
    print(dirs)
    file_score = []
    #print(find_goal(dirs[211]))
    for i in dirs:
        goals = find_goal(i)
        file_score.append(goals)
    
    #print(stat_check(file_score))   
    #os.chdir("E:/2021 SM 2 2021.7--11/assignments/EODP/individual/My_assignment")
    plt.boxplot(file_score)
    plt.ylabel("Total goals")
    plt.xlabel("text1_to_text265")
    plt.title("Total goals distribution from the texts")
    plt.savefig("task4.png", dpi=300)
    plt.show()
    return

def stat_check(lscore):
    count = 0
    lscore = sorted(lscore)
    for i in range(len(lscore)):
        if lscore[i]==0:
            count+=1
    print(count)
    k1 = (len(lscore)-1)*0.25+1
    print(lscore[int(k1)])
    med = (len(lscore)-1)*0.5+1
    print(lscore[int(med)])
    k3 = (len(lscore)-1)*0.75+1
    print(lscore[int(k3)])
    #IQR = lscore[int(k3)]-lscore[int(k1)]
    outlier = 0
    for i in lscore:
        if i > 3+1.5*3:
            outlier += 1
    return outlier

def task5():
    #Complete task 5 here, completed.
    with open(datafilepath) as fjson:
        data=json.load(fjson)
    list_tn = data["participating_clubs"]
    list_tn = sorted(list_tn)
    clubdic = {}
    for i in list_tn:
        clubdic[i] = 0
    #print(clubdic)
    
    dirs = os.listdir(articlespath)
    for i in clubdic.keys():
        for j in dirs:
            count_mention(j, i, clubdic)
    
    #print(clubdic)    
    #os.chdir("E:/2021 SM 2 2021.7--11/assignments/EODP/individual/My_assignment")
    task5_out = open("task5.csv", "w", newline = '')
    t5 = csv.writer(task5_out)
    headings = ["club_name", "number_of_mentions"]
    t5.writerow(headings)
    for i in list_tn:
        rows = [i, clubdic[i]]
        t5.writerow(rows)
    task5_out.close()
    
    team_mention = clubdic.values()
    plt.bar(arange(len(team_mention)), team_mention)
    plt.xticks(arange(len(list_tn)), list_tn, rotation = 90)
    plt.ylabel("The number of mentions")
    plt.title("The number of mentions of each team by media")
    plt.rcParams["figure.figsize"] = (3, 3)
    plt.savefig("task5.png", dpi = 300)
    plt.show()
    return

def count_mention(filename, pattern, Dict):
    file = open(articlespath+'/'+filename, 'r')
    strings = file.read()
    findings = re.findall(pattern, strings)
    if pattern in findings:
        Dict[pattern] += 1
    return

def task6():
    #Complete task 6 here, will be done on 8.25.
    #os.chdir("E:/2021 SM 2 2021.7--11/assignments/EODP/individual/My_assignment/data/football")
    #filepath = r'E:/2021 SM 2 2021.7--11/assignments/EODP/individual/My_assignment/data/football'
    dirs = os.listdir(articlespath)
    task5()
    file = open("task5.csv", 'r')
    read = csv.reader(file)
    head = next(read)
    data = list(read)
    
    name_list = []
    for i in data:
        name_list.append(i[0])
    #print(name_list)
    sims = {}
    dirs = os.listdir(articlespath)
    
    for i in range(len(data)):
        sims[data[i][0]] = []
        for j in range(len(data)):
            interact = co_check(data[i][0], data[j][0], dirs)
            #print(interact)
            similarity = calculate_sim_score(data[i], data[j], interact)
            sims[data[i][0]].append(similarity)

    sim_data = pd.DataFrame(sims, index = name_list)
    sns.heatmap(sim_data, cmap = 'viridis', xticklabels=True)
    plt.rcParams["figure.figsize"] = (3, 3)
    plt.savefig("task6.png", dpi = 500)
    plt.show()
    return

def co_check(pattern1, pattern2, filenames):
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
    score = 0
    denom = int(list1[-1])+int(list2[-1])
    if denom == 0:
        return 0
    else:    
        score = ((2*shared)/(denom))
    return score

def task7():
    #Complete task 7 here
    task2()
    task5()
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
    goal_list = [int(i[1]) for i in goals]
    mention_list = [int(j[-1]) for j in mentions]
    #print(goal_list)
    #print(mention_list)
    plt.scatter(mention_list, goal_list,color = "red", alpha = 0.6)
    plt.xlim(0,100)
    plt.ylim(0, 14)
    plt.title("Relationship of frequency of mention and goals scored by club")
    plt.xlabel("number of articles mentioning club")
    plt.ylabel("number of goals scored by club")
    plt.grid(True)
    plt.savefig("task7.png", dpi = 500)
    plt.show()
    return
    
def task8(filename):
    #Complete task 8 here, 截止8月31日，需完成至此。
    file = open(filename, 'r')
    strings = file.read()
    #ensure only white spaces in between.
    for i in strings:
        if i.isdigit():
            strings = strings.replace(i, ' ')
    #strings = remove_while_retain(strings)
    strings = re.sub(r"[^a-zA-Z]", ' ', strings)
    limit = 127
    for i in strings:
        if ord(i) > limit:
            strings = re.sub(i, ' ', strings)
    strings = strings.lower()
    strings = ' '.join(strings.split())
    #divide the strings into tokens, words bag created.
    words = word_tokenize(strings)
    #slice out those which are punctuations and numbers.
    words = pre_processing(words)
    #Remove stopwords.
    stopWords = set(stopwords.words('english'))
    stopWords.add('cannot')
    remaining = [i for i in words if not i in stopWords]
    #remove single-character strings.
    newlst = [i for i in remaining if len(i) > 1]
    file.close()
    return newlst

def pre_processing(wordlist):
    newlist = []
    for i in wordlist:
        if i.isalpha():
            newlist.append(i)
    return newlist

'''
def remove_while_retain(strings):
    puncs = string.punctuation
    punc = list(puncs)
    punc.append('\n')
    for i in punc:
        strings = strings.replace(i, ' ')
    return strings
'''    

def task9():
    #Complete task 9 here，于9月3日之前完成。
    dirs = os.listdir(articlespath)
    wordbags = {}
    all_words = []
    #Iterate over the folders and create wordbags for each text.
    for i in dirs:
        filename = articlespath+'/'+i
        elemental = task8(filename)
        results = count_dict(elemental)
        all_words+=elemental
        wordbags[i] = results
    
    all_words = set(all_words)
    keys = list(wordbags.keys())
    vals = list(wordbags.values())
    #print(vals)
    raw_count = construct_frequency(all_words, vals)
    docs = calculate_tfidf(raw_count)
    rows = []
    
    for i in range(0, len(keys)):
        for j in range(i+1, len(keys)):
            sim_score = cos_sim(docs[i], docs[j])
            rows.append([keys[i], keys[j], sim_score])
    rows = sorted(rows, key = lambda x:x[-1], reverse = True)
    file = open("task9.csv", 'w', newline = '')
    task9_out = csv.writer(file)
    heading = ['article1', 'article2', 'similarity']
    task9_out.writerow(heading)
    top_ten = 10
    for i in range(0, top_ten):
        task9_out.writerow(rows[i])
    file.close()
    
    return

def construct_frequency(word_bank, values):
    tfreq = []
    
    #iterate over all dicts for each file.
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
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(raw_tf)
    docs_arr = tfidf.toarray()
    return docs_arr

def count_dict(word_bag):
    #采集空字典。
    wordDict = {}
    for word in word_bag:
        #有词加一，无词添一。
        if word not in wordDict:
            wordDict[word] = 1
        else:
            wordDict[word] += 1
    return wordDict

def cos_sim(v1, v2):
    result = np.dot(v1, v2)/(norm(v1)*norm(v2))
    return format(result, '.16f')




def answer_check(filename):
    out = task8(filename)
    answer = ['man',
     'utd',
     'stroll',
     'cup',
     'win',
     'wayne',
     'rooney',
     'made',
     'winning',
     'return',
     'everton',
     'manchester',
     'united',
     'cruised',
     'fa',
     'cup',
     'quarter',
     'finals',
     'rooney',
     'received',
     'hostile',
     'reception',
     'goals',
     'half',
     'quinton',
     'fortune',
     'cristiano',
     'ronaldo',
     'silenced',
     'jeers',
     'goodison',
     'park',
     'fortune',
     'headed',
     'home',
     'minutes',
     'ronaldo',
     'scored',
     'nigel',
     'martyn',
     'parried',
     'paul',
     'scholes',
     'free',
     'kick',
     'marcus',
     'bent',
     'missed',
     'everton',
     'best',
     'chance',
     'roy',
     'carroll',
     'later',
     'struck',
     'missile',
     'saved',
     'feet',
     'rooney',
     'return',
     'always',
     'going',
     'potential',
     'flashpoint',
     'involved',
     'angry',
     'exchange',
     'spectator',
     'even',
     'kick',
     'rooney',
     'every',
     'touch',
     'met',
     'deafening',
     'chorus',
     'jeers',
     'crowd',
     'idolised',
     'year',
     'old',
     'everton',
     'started',
     'brightly',
     'fortune',
     'needed',
     'alert',
     'scramble',
     'away',
     'header',
     'bent',
     'near',
     'goal',
     'line',
     'cue',
     'united',
     'take',
     'complete',
     'control',
     'supreme',
     'passing',
     'display',
     'goodison',
     'park',
     'pitch',
     'cutting',
     'fortune',
     'gave',
     'united',
     'lead',
     'minutes',
     'rising',
     'meet',
     'ronaldo',
     'cross',
     'eight',
     'yards',
     'portuguese',
     'youngster',
     'allowed',
     'much',
     'time',
     'space',
     'hapless',
     'gary',
     'naysmith',
     'united',
     'dominated',
     'without',
     'creating',
     'many',
     'clear',
     'cut',
     'chances',
     'almost',
     'paid',
     'price',
     'making',
     'domination',
     'two',
     'minutes',
     'half',
     'time',
     'mikel',
     'arteta',
     'played',
     'superb',
     'ball',
     'area',
     'bent',
     'played',
     'onside',
     'gabriel',
     'heintze',
     'hesitated',
     'carroll',
     'plunged',
     'fee',
     'save',
     'united',
     'almost',
     'doubled',
     'lead',
     'minutes',
     'ronaldo',
     'low',
     'drive',
     'yards',
     'took',
     'deflection',
     'tony',
     'hibbert',
     'martyn',
     'dived',
     'save',
     'brilliantly',
     'martyn',
     'came',
     'everton',
     'rescue',
     'three',
     'minutes',
     'later',
     'rooney',
     'big',
     'moment',
     'almost',
     'arrived',
     'raced',
     'clean',
     'veteran',
     'keeper',
     'outstanding',
     'form',
     'nothing',
     'martyn',
     'could',
     'united',
     'doubled',
     'lead',
     'minutes',
     'doubled',
     'advantage',
     'scholes',
     'free',
     'kick',
     'took',
     'deflection',
     'martyn',
     'could',
     'parry',
     'ball',
     'ronaldo',
     'reacted',
     'first',
     'score',
     'easily',
     'everton',
     'problems',
     'worsened',
     'james',
     'mcfadden',
     'limped',
     'injury',
     'may',
     'trouble',
     'ahead',
     'everton',
     'goalkeeper',
     'carroll',
     'required',
     'treatment',
     'struck',
     'head',
     'missile',
     'thrown',
     'behind',
     'goal',
     'rooney',
     'desperate',
     'search',
     'goal',
     'return',
     'everton',
     'halted',
     'martyn',
     'injury',
     'time',
     'outpaced',
     'stubbs',
     'martyn',
     'denied',
     'england',
     'striker',
     'manchester',
     'united',
     'coach',
     'sir',
     'alex',
     'ferguson',
     'fantastic',
     'performance',
     'us',
     'fairness',
     'think',
     'everton',
     'missed',
     'couple',
     'players',
     'got',
     'young',
     'players',
     'boy',
     'ronaldo',
     'fantastic',
     'player',
     'persistent',
     'never',
     'gives',
     'know',
     'many',
     'fouls',
     'gets',
     'wants',
     'ball',
     'truly',
     'fabulous',
     'player',
     'everton',
     'martyn',
     'hibbert',
     'yobo',
     'stubbs',
     'naysmith',
     'osman',
     'carsley',
     'arteta',
     'kilbane',
     'mcfadden',
     'bent',
     'subs',
     'wright',
     'pistone',
     'weir',
     'plessis',
     'vaughan',
     'manchester',
     'united',
     'carroll',
     'gary',
     'neville',
     'brown',
     'ferdinand',
     'heinze',
     'ronaldo',
     'phil',
     'neville',
     'keane',
     'scholes',
     'fortune',
     'rooney',
     'subs',
     'howard',
     'giggs',
     'smith',
     'miller',
     'spector',
     'referee',
     'styles',
     'hampshire']
    print(len(answer))
    diffs = [i for i in answer if not i in out]
    print(len(diffs))
    return 