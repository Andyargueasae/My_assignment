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
    #os.chdir("E:/2021 SM 2 2021.7--11/assignments/EODP/individual/My_assignment/data/football")
    #filepath = r'E:/2021 SM 2 2021.7--11/assignments/EODP/individual/My_assignment/data/football'
    dirs = os.listdir(articlespath)
    #Now you have obtained the filenames dirs in the directory.
    file_score = []
    #print(find_goal(dirs[211]))
    for i in dirs:
        goals = find_goal(i)
        file_score.append(goals)
    
    #print(file_score)
    os.chdir("E:/2021 SM 2 2021.7--11/assignments/EODP/individual/My_assignment")
    task3_out = open("task3.csv", "w", newline = '')
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
        
    return max(sum_sc)

def task4():       #objective complete!
    #Complete task 4 here
    #os.chdir("E:/2021 SM 2 2021.7--11/assignments/EODP/individual/My_assignment/data/football")
    #filepath = r'E:/2021 SM 2 2021.7--11/assignments/EODP/individual/My_assignment/data/football'
    dirs = os.listdir(articlespath)
    print(dirs)
    file_score = []
    #print(find_goal(dirs[211]))
    for i in dirs:
        goals = find_goal(i)
        file_score.append(goals)
    
    #print(stat_check(file_score))   
    os.chdir("E:/2021 SM 2 2021.7--11/assignments/EODP/individual/My_assignment")
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
    os.chdir("E:/2021 SM 2 2021.7--11/assignments/EODP/individual/My_assignment")
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
    os.chdir("E:/2021 SM 2 2021.7--11/assignments/EODP/individual/My_assignment")
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
    print(len(data))
    file.close()
    
    sims = []
    
    return

def sim_score(club1, club2, filetrace):
    sim_scores = 0
    name1 = club1[0]
    name2 = club2[0]
    ment1 = club1[-1]
    ment2 = club2[-1]
    
    return sim_scores

def double_check(pattern1, pattern2, filename):
    co_exist = 1
    return co_exist

def task7():
    #Complete task 7 here
    return
    
def task8(filename):
    #Complete task 8 here, 截止8月31日，需完成至此。
    return
    
def task9():
    #Complete task 9 here，于9月3日之前完成。
    return