
# coding:utf8

import os, sys
import csv
import time
#from tqdm import tqdm
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import gzip
import pickle
import json
import bisect

buffering_size = 1 * 1024 * 1024



def get_filepaths(file_path):
    # paths of target 
    paths = []
    for root, dirs, files in os.walk(file_path): 
        for file in files:  
            if os.path.splitext(file)[1] == '.csv':  
                paths.append(os.path.join(root, file)) 
    return paths


def main(args):
    
    cutoff_rel = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]

    file_cutoff = {}
    writer_cutoff = {}
    for cut in cutoff_rel:
        file_cutoff[cut] = open(args.path_dict_qdr[:-4] + '_rel_' + str(cut) + '.csv', 'w')
        writer_cutoff[cut] = csv.writer(file_cutoff[cut], delimiter='\01')
        writer_cutoff[cut].writerow(['query', 'doc', 'rel'])

    stamp = time.time()
    with open(args.path_dict_qdr, "r", buffering=buffering_size) as file:
        reader = csv.reader(file, delimiter='\01')
        for rindex, row in enumerate(reader):
            if rindex == 0:
                print('\t'.join(row))
            else:
                query = row[0]
                doc = row[1]
                rel = float(row[2])
                
                for cut in cutoff_rel:
                    if rel >= cut:
                        writer_cutoff[cut].writerow([query, doc, rel])
                

            if rindex % 1000000 == 0:
                print('rindex', rindex, time.time() - stamp,)
    print('rindex', rindex, time.time() - stamp,)

    for cut in cutoff_rel:
        file_cutoff[cut].close()



    print('done')
    


if __name__ == "__main__":
    # parameter
    parser = ArgumentParser()
    path_group = '/mnt/wfs/mmchongqingwfssz/meilang/vertical_search/'
    parser.add_argument('--path_all_qdrank', type=str, default=path_group + "res_query_doc_rank_data/")
    parser.add_argument('--path_longzhu_doc', type=str, default=path_group + "longzhu_doc/ft_local/")
    parser.add_argument('--path_click_longzhu_query', type=str, default=path_group + "path_click_longzhu_query.csv")
    parser.add_argument('--path_general_longzhu_qdrank', type=str, default=path_group + "general_longzhu_query_doc_rank_data.csv")

    parser.add_argument('--path_dict_qdr', type=str, default=path_group + "path_dict_qdr.csv")
    parser.add_argument('--path_dict_pp', type=str, default=path_group + "path_dict_pp.csv")

    parser.add_argument('--epoch', type=int, default=40)
    
    args = parser.parse_args()

    print(args)
    # function to sample data
    main(args)



