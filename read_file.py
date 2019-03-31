
import pickle
import pandas as pd

def read_file():
    min_year = 3000
    max_year = 0
    data_set = {}

    #count the citations of each paper
    #exclude the papers with no citations
    with open("outputacm.txt", "r") as file:
        line = file.readline()
        for line in file:
            if line.startswith("#index"):
                tmp_index = int(line[6:].rstrip())
            if line.startswith("#t"):
                tmp_year = int(line[2:].rstrip())
            # if paper has no references, throw it
            if line.startswith("#%"):
                tmp_ref = int(line[2:].rstrip())
                if tmp_ref in data_set:
                    if tmp_year in data_set[tmp_ref]:
                        data_set[tmp_ref][tmp_year] += 1
                    else:
                        data_set[tmp_ref].update({tmp_year:1})
                else:
                    data_set.update({tmp_ref: {tmp_year: 1}})


    #create dataframe and save it in a csv file
    data_frame = pd.DataFrame(data_set)

    data_frame.to_csv("data_set.csv", sep='\t', encoding='utf-8')

read_file()
