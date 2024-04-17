# coding:utf8

import os, sys
import csv
import time
from tqdm import tqdm
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import gzip
import pickle
import json

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
    '''
    dict_longzhu_docid = {}
    
    # path_longzhu_doc
    path_longzhu_doc = get_filepaths(args.path_longzhu_doc)
    print('path_longzhu_doc', len(path_longzhu_doc))
    
    stamp = time.time()
    dict_longzhu_docid = {}
    count_ = 0
    for pf in path_longzhu_doc:
        print('reading', pf)
        with open(pf, "r", buffering=buffering_size) as file:
            reader = csv.reader(file, delimiter='\01')
            for row in reader:
                content_, = row
                docid_ = str(json.loads(content_)['docid'])
                dict_longzhu_docid.setdefault(docid_, content_)
                count_ += 1

                if count_ % 2000000 == 0:
                    print('count_', count_, time.time() - stamp, 'dict_longzhu_docid', len(dict_longzhu_docid), sys.getsizeof(dict_longzhu_docid))
    print('count_', count_, time.time() - stamp, 'dict_longzhu_docid', len(dict_longzhu_docid))

    del dict_longzhu_docid
    '''

    # path_click_longzhu_query
    stamp = time.time()
    count_ = 0
    click_longzhu_query = {}
    with open(args.path_click_longzhu_query, "r", buffering=buffering_size) as file:
        reader = csv.reader(file, delimiter='\01')
        for row in reader:
            query_, = row
            click_longzhu_query.setdefault(query_, 0)
            count_ += 1
    print('count_', count_, time.time() - stamp, 'click_longzhu_query', len(click_longzhu_query), sys.getsizeof(click_longzhu_query))


    ######### ----------------------------------------------------------------------------------
   
    # pbm training
    dict_qdr = {}
    dict_pp = {}
    ### initialize
    stamp = time.time()
    print('initialize')
    with open(args.path_general_longzhu_qdrank, "r", buffering=buffering_size) as file:
        reader = csv.reader(file, delimiter='\01')
        for rindex, row in enumerate(reader):
            if rindex == 0:
                    print(row)
            else:
                #query, docid_list, docid_list_pos, click_times, recall_times = row
                query = row[0]
                docid_list = row[1]
                docid_list_pos = int(row[2])
                click_times = float(row[3])
                recall_times = float(row[4])
                
                assert query in click_longzhu_query

                dict_qdr.setdefault(
                    (query, docid_list), 
                    # rel, zi, mu
                    [0.5, 1.0, 2.0],
                )
                dict_pp.setdefault(
                    docid_list_pos, 
                    # prob, zi, mu
                    [0.5, 1.0, 2.0],
                )
                if (rindex + 1) % 10000000 == 0:
                    print('rindex', rindex, time.time() - stamp, 'dict_qdr', len(dict_qdr), 'dict_pp', len(dict_pp),)
    print(dict_pp)
    print('rindex', rindex, time.time() - stamp, 'dict_qdr', len(dict_qdr), 'dict_pp', len(dict_pp),)
    
    # training
    stamp = time.time()
    for epoch in range(args.epoch):
        print(epoch, 're-counting')
        for key in dict_qdr:
            # zi, mu
            dict_qdr[key][1] = 1.0
            dict_qdr[key][2] = 2.0
        for key in dict_pp:
            # zi, mu
            dict_pp[key][1] = 1.0
            dict_pp[key][2] = 2.0

        print(epoch, 'training')
        with open(args.path_general_longzhu_qdrank, "r", buffering=buffering_size) as file:
            reader = csv.reader(file, delimiter='\01')
            for rindex, row in enumerate(reader):
                if rindex == 0:
                        print(row)
                else:
                    #query, docid_list, docid_list_pos, click_times, recall_times = row
                    query = row[0]
                    docid_list = row[1]
                    docid_list_pos = int(row[2])
                    click_times = float(row[3])
                    recall_times = float(row[4])

                    old_rel = dict_qdr[(query, docid_list)][0]
                    old_prob = dict_pp[docid_list_pos][0]

                    new_rel_zi = (
                        recall_times * old_rel * (1 - old_prob) + click_times * (1 - old_rel)
                    ) / (1 - old_rel * old_prob)
                    
                    new_prob_zi = (
                        recall_times * old_prob * (1 - old_rel) + click_times * (1 - old_prob)
                    ) / (1 - old_rel * old_prob)

                    # zi, mu
                    dict_qdr[(query, docid_list)][1] += new_rel_zi
                    dict_qdr[(query, docid_list)][2] += recall_times
                    # zi, mu
                    dict_pp[docid_list_pos][1] += new_prob_zi
                    dict_pp[docid_list_pos][2] += recall_times

        ### update
        print(epoch, 'updating')
        for key in dict_qdr:
            dict_qdr[key][0] = dict_qdr[key][1] / dict_qdr[key][2]
        for key in dict_pp:
            dict_pp[key][0] = dict_pp[key][1] / dict_pp[key][2]
        print(dict_pp)
        print(epoch, 'epoch done', time.time() - stamp, len(dict_qdr), len(dict_pp))


        if (epoch + 1) % 10 == 0:
            # save
            stamp = time.time()
            with open(args.path_dict_qdr, 'w') as file:
                writer = csv.writer(file, delimiter='\01')
                writer.writerow(['query', 'doc', 'rel'])
                for qd in dict_qdr:
                    query, doc = qd
                    rel = dict_qdr[qd][0]
                    writer.writerow([query, doc, rel])
            print('save qdr', time.time() - stamp)
            with open(args.path_dict_pp, 'w') as file:
                writer = csv.writer(file, delimiter='\01')
                writer.writerow(['position', 'prob'])
                for pos in dict_pp:
                    prob = dict_pp[pos][0]
                    writer.writerow([pos, prob])
            print('save pp', time.time() - stamp)
    
    print('done')
    
    
    '''
    # path_general_longzhu_qdrank
    stamp = time.time()
    count_ = 0
    general_longzhu_qdrank = []
    with open(args.path_general_longzhu_qdrank, "r", buffering=buffering_size) as file:
        reader = csv.reader(file, delimiter='\01')
        for rindex, row in enumerate(reader):
            if rindex == 0:
                    print(row)
            else:
                #query, docid_list, docid_list_pos, click_times, recall_times = row
                query = row[0]
                docid_list = row[1]

                docid_list_pos = int(row[2])
                click_times = float(row[3])
                recall_times = float(row[4])
                
                assert query in click_longzhu_query

                general_longzhu_qdrank.append((query, docid_list, docid_list_pos, click_times, recall_times))
                count_ += 1

                if count_ % 10000000 == 0:
                    print('count_', count_, time.time() - stamp, 'general_longzhu_qdrank', len(general_longzhu_qdrank))

    general_longzhu_qdrank = tuple(general_longzhu_qdrank)
    print('count_', count_, time.time() - stamp, 'general_longzhu_qdrank', len(general_longzhu_qdrank))

    # pbm training
    dict_qdr = {}
    dict_pp = {}
    ### initialize
    stamp = time.time()
    print('initialize')
    for rindex, row in enumerate(general_longzhu_qdrank):
        query, docid_list, docid_list_pos, click_times, recall_times = row
        dict_qdr.setdefault(
            (query, docid_list), 
            # rel, zi, mu
            [0.5, 1.0, 2.0],
        )
        dict_pp.setdefault(
            docid_list_pos, 
            # prob, zi, mu
            [0.5, 1.0, 2.0],
        )
        if (rindex + 1) % 10000000 == 0:
            print('rindex', rindex, time.time() - stamp, 'dict_qdr', len(dict_qdr), 'dict_pp', len(dict_pp),)
    print(dict_pp)
    print('rindex', rindex, time.time() - stamp, 'dict_qdr', len(dict_qdr), 'dict_pp', len(dict_pp),)
    
    # training
    stamp = time.time()
    for epoch in range(args.epoch):
        print(epoch, 're-counting')
        for key in dict_qdr:
            # zi, mu
            dict_qdr[key][1] = 1.0
            dict_qdr[key][2] = 2.0
        for key in dict_pp:
            # zi, mu
            dict_pp[key][1] = 1.0
            dict_pp[key][2] = 2.0

        print(epoch, 'training')
        for rindex, row in enumerate(general_longzhu_qdrank):
            query, docid_list, docid_list_pos, click_times, recall_times = row

            old_rel = dict_qdr[(query, docid_list)][0]
            old_prob = dict_pp[docid_list_pos][0]

            new_rel_zi = (
                recall_times * old_rel * (1 - old_prob) + click_times * (1 - old_rel)
            ) / (1 - old_rel * old_prob)
            
            new_prob_zi = (
                recall_times * old_prob * (1 - old_rel) + click_times * (1 - old_prob)
            ) / (1 - old_rel * old_prob)

            # zi, mu
            dict_qdr[(query, docid_list)][1] += new_rel_zi
            dict_qdr[(query, docid_list)][2] += recall_times
            # zi, mu
            dict_pp[docid_list_pos][1] += new_prob_zi
            dict_pp[docid_list_pos][2] += recall_times

        ### update
        print(epoch, 'updating')
        for key in dict_qdr:
            dict_qdr[key][0] = dict_qdr[key][1] / dict_qdr[key][2]
        for key in dict_pp:
            dict_pp[key][0] = dict_pp[key][1] / dict_pp[key][2]
        print(dict_pp)
        print(epoch, 'epoch done', time.time() - stamp, len(dict_qdr), len(dict_pp))



        if (epoch + 1) % 10 == 0:
            # save
            stamp = time.time()
            with open(args.path_dict_qdr, 'w') as file:
                writer = csv.writer(file, delimiter='\01')
                writer.writerow(['query', 'doc', 'rel'])
                for qd in dict_qdr:
                    query, doc = qd
                    rel = dict_qdr[qd][0]
                    writer.writerow([query, doc, rel])
            print('save qdr', time.time() - stamp)
            with open(args.path_dict_pp, 'w') as file:
                writer = csv.writer(file, delimiter='\01')
                writer.writerow(['position', 'prob'])
                for pos in dict_pp:
                    prob = dict_pp[pos][0]
                    writer.writerow([pos, prob])
            print('save pp', time.time() - stamp)

    print('done')
    '''
    '''
    # path_general_longzhu_qdrank
    stamp = time.time()
    general_longzhu_qdrank = []
    with open(args.path_general_longzhu_qdrank, "r", buffering=buffering_size) as file:
        reader = csv.reader(file, delimiter='\01')
        for rindex, row in enumerate(reader):
            if rindex == 0:
                    print(row)
            else:
                # query, docid_list, docid_list_pos, click_times, recall_times
                query, docid_list, docid_list_pos, click_times, recall_times = row
                assert row[0] in click_longzhu_query
                
                docid_list_pos = int(docid_list_pos)
                click_times = float(click_times)
                recall_times = float(recall_times)

                general_longzhu_qdrank.append(
                    (query, docid_list, docid_list_pos, click_times, recall_times,)
                )

                if rindex % 10000000 == 0:
                    print('rindex', rindex, time.time() - stamp, 'general_longzhu_qdrank', len(general_longzhu_qdrank), sys.getsizeof(general_longzhu_qdrank))
    general_longzhu_qdrank = tuple(general_longzhu_qdrank)

    print('rindex', rindex, time.time() - stamp, 'general_longzhu_qdrank', len(general_longzhu_qdrank), sys.getsizeof(general_longzhu_qdrank))

    # training
    dict_qdr = {}
    dict_pp = {}
    ### initialize
    stamp = time.time()
    print('initialize')
    for rindex, row in enumerate(general_longzhu_qdrank):
        query, docid_list, docid_list_pos, click_times, recall_times = row
        dict_qdr.setdefault(
            (query, docid_list), 
            # rel, zi, mu
            [0.5, 1.0, 2.0],
        )
        dict_pp.setdefault(
            docid_list_pos, 
            # prob, zi, mu
            [0.5, 1.0, 2.0],
        )
        if (rindex + 1) % 10000000 == 0:
            print('rindex', rindex, time.time() - stamp, 'dict_qdr', len(dict_qdr), sys.getsizeof(dict_qdr), 'dict_pp', len(dict_pp), sys.getsizeof(dict_pp))
    print(dict_pp)
    print('rindex', rindex, time.time() - stamp, 'dict_qdr', len(dict_qdr), sys.getsizeof(dict_qdr), 'dict_pp', len(dict_pp), sys.getsizeof(dict_pp))


    stamp = time.time()
    for epoch in range(args.epoch):
        print(epoch, 're-counting')
        for key in dict_qdr:
            # zi, mu
            dict_qdr[key][1] = 1.0
            dict_qdr[key][2] = 2.0
        for key in dict_pp:
            # zi, mu
            dict_pp[key][1] = 1.0
            dict_pp[key][2] = 2.0

        print(epoch, 'training')
        for rindex, row in enumerate(general_longzhu_qdrank):
            query, docid_list, docid_list_pos, click_times, recall_times = row

            old_rel = dict_qdr[(query, docid_list)][0]
            old_prob = dict_pp[docid_list_pos][0]

            new_rel_zi = (
                recall_times * old_rel * (1 - old_prob) + click_times * (1 - old_rel)
            ) / (1 - old_rel * old_prob)
            # zi, mu
            dict_qdr[(query, docid_list)][1] += new_rel_zi
            dict_qdr[(query, docid_list)][2] += recall_times
            
            new_prob_zi = (
                recall_times * old_prob * (1 - old_rel) + click_times * (1 - old_prob)
            ) / (1 - old_rel * old_prob)
            # zi, mu
            dict_pp[docid_list_pos][1] += new_prob_zi
            dict_pp[docid_list_pos][2] += recall_times

            if (rindex + 1) % 10000000 == 0:
                print('rindex training', rindex + 1, time.time() - stamp, )

        ### update
        print(epoch, 'updating')
        for key in dict_qdr:
            dict_qdr[key][0] = dict_qdr[key][1] / dict_qdr[key][2]
        for key in dict_pp:
            dict_pp[key][0] = dict_pp[key][1] / dict_pp[key][2]
        print(dict_pp)
        print(epoch, 'epoch done', time.time() - stamp, len(dict_qdr), len(dict_pp))


    
    # save
    stamp = time.time()
    with open(args.path_dict_qdr, 'w') as file:
        writer = csv.writer(file, delimiter='\01')
        writer.writerow(['query', 'doc', 'rel'])
        for qd in dict_qdr:
            query, doc = qd
            rel = dict_qdr[qd][0]
            writer.writerow([query, doc, rel])
    print('save qdr', time.time() - stamp)
    with open(args.path_dict_pp, 'w') as file:
        writer = csv.writer(file, delimiter='\01')
        writer.writerow(['position', 'prob'])
        for pos in dict_pp:
            prob = dict_pp[pos][0]
            writer.writerow([pos, prob])
    print('save pp', time.time() - stamp)
    
    print('done')
    '''


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



