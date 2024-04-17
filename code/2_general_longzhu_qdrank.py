# coding:utf8

import os
import csv
#csv.field_size_limit(100 * 1024 * 1024)
import time
#from tqdm import tqdm
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import gzip
import pickle
import json
#from transformers import BertTokenizerFast

import io
print(io.DEFAULT_BUFFER_SIZE)
buffering_size = 1 * 1024 * 1024


def compress_dict(dictionary, filename):
    with gzip.open(filename, 'wb') as file:
        pickle.dump(dictionary, file)


def decompress_dict(filename):
    with gzip.open(filename, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary

def time_stat(start_time, end_time):
    return ((end_time - start_time).seconds)

def get_filepaths(file_path):
    # paths of target 
    paths = []
    for root, dirs, files in os.walk(file_path): 
        for file in files:  
            if os.path.splitext(file)[1] == '.csv':  
                paths.append(os.path.join(root, file)) 
    return paths



def process_docid_list(idddd):
    if len(idddd.strip()) > 0:
        splits = idddd.split('-')
        if len(splits) == 2:
            return splits[1]
    
    return idddd


def main(args):
    print('begin')
    
    stamp = time.time()
    count_ = 0
    click_longzhu_query = {}
    with open(args.path_click_longzhu_query, "r", buffering=buffering_size) as file:
        reader = csv.reader(file, delimiter='\01')
        for row in reader:
            query_, = row
            click_longzhu_query.setdefault(query_, 0)
            count_ += 1
    print('count_', count_, time.time() - stamp, 'click_longzhu_query', len(click_longzhu_query))


    # path_all_qdrank
    path_all_qdrank = get_filepaths(args.path_all_qdrank)#.sort()
    print('path_all_qdrank', len(path_all_qdrank))

    stamp = time.time()
    count_ = 0
    count_general = 0
    with open(args.path_general_longzhu_qdrank, 'w', buffering=buffering_size) as file:
        writer = csv.writer(file, delimiter='\01')
        writer.writerow(["query", "docid_list", "docid_list_pos", "click_times", "recall_times"])

        for pf in path_all_qdrank:
            print('reading', pf)
            with open(pf, "r", buffering=buffering_size) as file:
                reader = csv.reader(file, delimiter='\01')
                for rindex, row in enumerate(reader):
                    if rindex == 0:
                        print(row)
                    else:
                        query, docid_list, docid_list_pos, click_times, recall_times = row
                        docid_list = process_docid_list(docid_list)

                        if query in click_longzhu_query and \
                            len(query.strip()) > 0 and \
                            len(docid_list.strip()) > 0:

                            writer.writerow([query, docid_list, docid_list_pos, click_times, recall_times])
                            count_general += 1

                        count_ += 1
                        if count_ % 1000000 == 0:
                            print('count_', count_, time.time() - stamp, count_general)
    print('count_', count_, time.time() - stamp, count_general)


    print('done')


if __name__ == "__main__":
    # parameter
    parser = ArgumentParser()
    path_group = '/mnt/wfs/mmchongqingwfssz/meilang/vertical_search/'
    parser.add_argument('--path_all_qdrank', type=str, default=path_group + "res_query_doc_rank_data/")
    parser.add_argument('--path_longzhu_doc', type=str, default=path_group + "longzhu_doc/ft_local/")
    parser.add_argument('--path_click_longzhu_query', type=str, default=path_group + "path_click_longzhu_query.csv")
    parser.add_argument('--path_general_longzhu_qdrank', type=str, default=path_group + "general_longzhu_query_doc_rank_data.csv")
    args = parser.parse_args()

    print(args)
    # function to sample data
    main(args)
