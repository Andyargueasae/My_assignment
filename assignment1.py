# This is the file you will need to edit in order to complete assignment 1
# You may create additional functions, but all code must be contained within this file


# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import csv

# You should use these two variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'

#The task1 is finished!!
def task1():
    #Complete task 1 here
    with open('data/data.json') as fjson:
        data=json.load(fjson)
    list_tc = data['teams_codes']
    list_tc = sorted(list_tc)
    return list_tc
    
def task2():
    #Complete task 2 here
    with open('data/data.json') as fjson:
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
        
    t2_out = open("task2.csv", 'w')
    output = csv.writer(t2_out)
    headings = ["team_code","goals_scored_by_team", "goals_scored_against_team"]
    output.writerow(headings)
    for i in range(0, len(list_tc)):
        row = [list_tc[i], goalscore[i], goalconc[i]]
        output.writerow(row)
    t2_out.close()
    return 
      
def task3():
    #Complete task 3 here
    return

def task4():
    #Complete task 4 here
    return
    
def task5():
    #Complete task 5 here
    return
    
def task6():
    #Complete task 6 here
    return
    
def task7():
    #Complete task 7 here
    return
    
def task8(filename):
    #Complete task 8 here
    return
    
def task9():
    #Complete task 9 here
    return