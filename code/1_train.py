from ast import arg
import sys, os
#os.environ['CUDA_VISIBLE_DEVICES']='0'#,1,2,3,4,5,6,7'
import torch
import torch.nn as nn
import csv, random, json
from argparse import ArgumentParser
import numpy as np
import pickle, faiss, time
from transformers import AdamW, BertTokenizerFast
from contextlib import suppress
import copy

import model



def save_checkpoint(checkpoint_path, model, epoch, ):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    checkpoint = {}
    checkpoint['model_state_dict'] = model.state_dict()
    torch.save(checkpoint, checkpoint_path + '/model-' + str(epoch) + '.ck')


def get_filepaths(file_path):
    # paths of target 
    paths = []
    for root, dirs, files in os.walk(file_path): 
        for file in files:  
            if os.path.splitext(file)[1] == '.csv':  
                paths.append(os.path.join(root, file)) 
    return paths


class OurDataset(torch.utils.data.Dataset):
    def __init__(self, userlog_data):
        self.userlog_data = userlog_data
        
    def __len__(self):
        return len(self.userlog_data)

    def __getitem__(self, index):
        return self.userlog_data[index]


def has_chinese(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def filter_structure(one_dict_):
    if isinstance(one_dict_, dict):
        keys = list(one_dict_.keys())
        for key in keys:
            if isinstance(one_dict_[key], dict):
                filter_structure(one_dict_[key])
            else:
                if isinstance(one_dict_[key], str):
                    if not has_chinese(one_dict_[key]):
                        one_dict_.pop(key)
                else:
                    one_dict_.pop(key)


def split_text_and_struct(one_filter_dict_, this_hierarchy=0):
    text_ = []
    if isinstance(one_filter_dict_, dict):
        keys = list(one_filter_dict_.keys())
        for key in keys:
            if isinstance(one_filter_dict_[key], str):
                if has_chinese(one_filter_dict_[key]):
                    text_.append(
                        {
                            'node': one_filter_dict_[key],
                            'hierarchy' : this_hierarchy + 1,
                            'type' : 1,
                        }
                    )
                    #one_filter_dict_[key] = '[MASK]' + one_filter_dict_[key]
            if isinstance(one_filter_dict_[key], dict):
                text_ += split_text_and_struct(one_filter_dict_[key], this_hierarchy + 1)[0]

    return text_, one_filter_dict_



def parse_struct(struct_, tokeriz, this_hierarchy=0):
    edge_ = {
        'start' : [],
        'end' : [],
    }

    if isinstance(struct_, dict):
        for key in list(struct_.keys()):
            parent_toks = tokeriz(key)['input_ids']
            # self attention
            for pos_index_1, tt_1 in enumerate(parent_toks):
                for pos_index_2, tt_2 in enumerate(parent_toks):
                    edge_["start"].append(
                        {
                            'node': tt_1,
                            'position' : pos_index_1,
                            'hierarchy' : this_hierarchy,
                            'type' : 0,
                        }
                    )
                    edge_["end"].append(
                        {
                            'node': tt_2,
                            'position' : pos_index_2,
                            'hierarchy' : this_hierarchy,
                            'type' : 0,
                        }
                    )

            if isinstance(struct_[key], str):
                # parent-child attention
                child_toks = tokeriz('[MASK]')['input_ids']
                mask_content = struct_[key]
                for pos_index_1, tt_1 in enumerate(parent_toks):
                    for pos_index_2, tt_2 in enumerate(child_toks):
                        edge_["start"].append(
                            {
                                'node': tt_1,
                                'position' : pos_index_1,
                                'hierarchy' : this_hierarchy,
                                'type' : 0,
                            }
                        )
                        edge_["end"].append(
                            {
                                'node': tt_2,
                                'position' : pos_index_2,
                                'hierarchy' : this_hierarchy + 1,
                                'type' : 1,
                                'mask_content' : mask_content,
                            }
                        )
                for pos_index_1, tt_1 in enumerate(child_toks):
                    for pos_index_2, tt_2 in enumerate(parent_toks):
                        edge_["start"].append(
                            {
                                'node': tt_1,
                                'position' : pos_index_1,
                                'hierarchy' : this_hierarchy + 1,
                                'type' : 1,
                                'mask_content' : mask_content,
                            }
                        )
                        edge_["end"].append(
                            {
                                'node': tt_2,
                                'position' : pos_index_2,
                                'hierarchy' : this_hierarchy,
                                'type' : 0,
                            }
                        )
                
            if isinstance(struct_[key], dict):
                for child_key in list(struct_[key].keys()):
                    # parent-child attention
                    child_toks = tokeriz(child_key)['input_ids']
                    for pos_index_1, tt_1 in enumerate(parent_toks):
                        for pos_index_2, tt_2 in enumerate(child_toks):
                            edge_["start"].append(
                                {
                                    'node': tt_1,
                                    'position' : pos_index_1,
                                    'hierarchy' : this_hierarchy,
                                    'type' : 0,
                                }
                            )
                            edge_["end"].append(
                                {
                                    'node': tt_2,
                                    'position' : pos_index_2,
                                    'hierarchy' : this_hierarchy + 1,
                                    'type' : 0,
                                }
                            )
                    for pos_index_1, tt_1 in enumerate(child_toks):
                        for pos_index_2, tt_2 in enumerate(parent_toks):
                            edge_["start"].append(
                                {
                                    'node': tt_1,
                                    'position' : pos_index_1,
                                    'hierarchy' : this_hierarchy + 1,
                                    'type' : 0,
                                }
                            )
                            edge_["end"].append(
                                {
                                    'node': tt_2,
                                    'position' : pos_index_2,
                                    'hierarchy' : this_hierarchy,
                                    'type' : 0,
                                }
                            )

                edge_["start"] += parse_struct(struct_[key], tokeriz, this_hierarchy + 1)['start']
                edge_["end"] += parse_struct(struct_[key], tokeriz, this_hierarchy + 1)['end']

    return edge_







def main(args):
    # initalize
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    torch.distributed.barrier()


    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.path_pretrain_model)


    # path_longzhu_doc
    dict_longzhu_doc = {}
    
    stamp = time.time()
    count_ = 0
    for pf in get_filepaths(args.path_longzhu_doc):
        if args.local_rank == 0:
            print('reading longzhu doc:', pf)
        with open(pf, "r", buffering=1*1024*1024) as file:
            reader = csv.reader(file, delimiter='\01')
            for row in reader:
                value_, = row

                value_ = json.loads(value_)
                
                docid_ = str(value_['docid'])

                btype = str(value_['content']['business_type'])

                text_ = {'doc': {}}
                if 'content' in value_:
                    text_['content'] = value_['content']
                if 'dragon_doc_business_info' in value_:
                    text_['dragon_doc_business_info'] = value_['dragon_doc_business_info']

                filter_structure(text_)


                text_, structure_ = split_text_and_struct(text_)

                assert isinstance(text_, list)
                
                edge_ = parse_struct(structure_, tokenizer, 0)
                mask_index = []
                node_ = {}
                for i in range(len(edge_['start'])):
                    node_.setdefault(repr(edge_['start'][i]), len(node_))
                    node_.setdefault(repr(edge_['end'][i]), len(node_))

                    edge_['start'][i]['node_index'] = i
                    edge_['end'][i]['node_index'] = i

                    if edge_['end'][i]['node'] == '[MASK]':
                        mask_index.append(edge_['end'][i]['node_index'])
                
                
                dict_longzhu_doc[docid_] = (btype, text_, edge_, mask_index)
                count_ += 1

                if count_ % 1000000 == 0:
                    if args.local_rank == 0:
                        print('count_', count_, 'time', time.time() - stamp, 'dict_longzhu_doc', len(dict_longzhu_doc), )
                        print(text_)
                
    if args.local_rank == 0:
        print('count_', count_, 'time', time.time() - stamp, 'dict_longzhu_doc', len(dict_longzhu_doc), )


    # path_normal_doc
    dict_normal_doc = {}

    stamp = time.time()
    if args.local_rank == 0:
        print('reading normal doc:', )
    with open(args.path_normal_doc, 'r', buffering=1*1024*1024) as file:
        reader = csv.reader(file, delimiter='\01')
        for rindex, row in enumerate(reader):
            value, = row
            value = json.loads(value)
            docid_ = str(value['docid'])
            text_ = str(value['title'])

            dict_normal_doc[docid_] = ('text', text_, 
                {
                    'start' : [],
                    'end' : [],
            }, [])

            if (rindex + 1) % 1000000 == 0:
                if args.local_rank == 0:
                    print('rindex+1', rindex + 1, 'time', time.time() - stamp, 'dict_normal_doc', len(dict_normal_doc), )
    if args.local_rank == 0:
        print('rindex+1', rindex + 1, 'time', time.time() - stamp, 'dict_normal_doc', len(dict_normal_doc), )


    # merge 
    dict_doc = {}
    for key in list(dict_longzhu_doc.keys()) + list(dict_normal_doc.keys()):
        if key in dict_longzhu_doc and key in dict_normal_doc:
            if args.local_rank == 0:
                print('same key')
                print(key, )
                print(dict_longzhu_doc[key])
                print(dict_normal_doc[key])
        else:
            if key in dict_longzhu_doc:
                dict_doc[key] = dict_longzhu_doc[key]
            if key in dict_normal_doc:
                dict_doc[key] = dict_normal_doc[key]
    if args.local_rank == 0:
        print('dict_doc', len(dict_doc))
    

    # path longzhu log train
    train_longzhu_data = []
    if args.local_rank == 0:
        print('reading longzhu log:', )
    with open(args.path_longzhu_log_train, 'r', buffering=1*1024*1024) as file:
        reader = csv.reader(file, delimiter='\01')
        for rindex, row in enumerate(reader):
            if rindex == 0:
                if args.local_rank == 0:
                    print(row, )
            else:
                query, docid, rel = row
                train_longzhu_data.append([query, docid])
    if args.local_rank == 0:
        print('train_longzhu_data', len(train_longzhu_data))

    # path normal log train
    train_normal_data = []
    if args.local_rank == 0:
        print('reading normal log:', )
    with open(args.path_normal_log_train, 'r', buffering=1*1024*1024) as file:
        reader = csv.reader(file, delimiter='\01')
        for rindex, row in enumerate(reader):
            if rindex == 0:
                if args.local_rank == 0:
                    print(row, )
            else:
                query, docid, rel = row
                train_normal_data.append([query, docid])
    if args.local_rank == 0:
        print('train_normal_data', len(train_normal_data))


    train_data = train_longzhu_data + train_normal_data
    if args.local_rank == 0:
        print('train_data', len(train_data))

    
    train_data = OurDataset(train_data)


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, 
        sampler=train_sampler, 
        batch_size=args.batch_size,
        num_workers=4, 
        pin_memory=True,
        collate_fn=lambda x:x,
    )



    # ----- model load
    itmodel = model.structure.from_pretrained(args.path_pretrain_model, args)
    itmodel = itmodel.to(device)
    itmodel.train()

    itmodel = nn.parallel.DistributedDataParallel(
        itmodel, 
        device_ids=[args.local_rank],
        output_device=args.local_rank,
    )

    # optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, itmodel.parameters()), lr=args.learning_rate, eps=1e-8)
    scaler = torch.cuda.amp.GradScaler() if args.mix_amp == 1 else None
    autocast = torch.cuda.amp.autocast if args.mix_amp == 1 else suppress


    torch.distributed.barrier()


    # training
    stamp = time.time()
    for epoch in range(args.epoch):
        # shuffle
        train_sampler.set_epoch(epoch)

        # save
        if args.local_rank == 0:
            print('save checkpoint', args.local_rank)
            save_checkpoint(args.checkpoint_path, itmodel.module, epoch)
        
        # batch
        for idx_,  batch_ in enumerate(train_dataloader):
            assert args.local_rank == torch.distributed.get_rank()
            ### query
            batch_query_context = []
            ### pos doc
            batch_pos_doc_btypeid = []
            batch_pos_doc_text = []

            batch_pos_doc_edge4start_nodeindex = []
            batch_pos_doc_edge4end_nodeindex = []

            batch_pos_doc_edge_nodetok = []
            batch_pos_doc_edge_position = []
            batch_pos_doc_edge_hierarchy = []
            batch_pos_doc_edge_attrtype = []


            batch_mask_index = []


            
            this_batch_size = len(batch_)
            for kkkkk in range(this_batch_size):
                query_context = batch_[kkkkk][0]
                pos_doc_id = batch_[kkkkk][1]
                pos_doc_btype, pos_doc_text, pos_doc_edge, pos_doc_mask_index = dict_doc[pos_doc_id]

                batch_query_context.append(query_context)
                
                batch_pos_doc_btypeid.append(args.btype2id[pos_doc_btype])
                batch_pos_doc_text.append(pos_doc_text)

                assert len(pos_doc_edge['start']) == len(pos_doc_edge['end'])

                for ooo in range(len(pos_doc_edge['start'])):
                    batch_pos_doc_edge4start_nodeindex.append(pos_doc_edge['start'][ooo]['node_index'])
                    batch_pos_doc_edge4end_nodeindex.append(pos_doc_edge['end'][ooo]['node_index'])
                
                dict_batch_pos_doc_edge = set(batch_pos_doc_edge4start_nodeindex + batch_pos_doc_edge4end_nodeindex)
                for ppp in range(len(dict_batch_pos_doc_edge)):
                    batch_pos_doc_edge_nodetok.append(pos_doc_edge['start'][ooo]['node'] if ppp in dict_batch_pos_doc_edge else pos_doc_edge['end'][ooo]['node'])
                    batch_pos_doc_edge_position.append(pos_doc_edge['start'][ooo]['position'] if ppp in dict_batch_pos_doc_edge else pos_doc_edge['end'][ooo]['position'])
                    batch_pos_doc_edge_hierarchy.append(pos_doc_edge['start'][ooo]['hierarchy'] if ppp in dict_batch_pos_doc_edge else pos_doc_edge['end'][ooo]['hierarchy'])
                    batch_pos_doc_edge_attrtype.append(pos_doc_edge['start'][ooo]['type'] if ppp in dict_batch_pos_doc_edge else pos_doc_edge['end'][ooo]['type'])

                    batch_mask_index.append(pos_doc_mask_index)


            ### query
            batch_query_context = tokenizer(batch_query_context, padding=True, return_tensors='pt', truncation=True, max_length=args.trunc_length)
            ### pos doc 
            batch_pos_doc_text = tokenizer(batch_pos_doc_text, padding=True, return_tensors='pt', truncation=True, max_length=args.trunc_length)
            batch_pos_doc_edge_nodetok = tokenizer(batch_pos_doc_edge_nodetok, padding=True, return_tensors='pt', truncation=True, max_length=args.trunc_length)


            with autocast():
                # cls emb
                query_context_emb = itmodel.module.query_embed(
                    batch_query_context['input_ids'].to(device), 
                    batch_query_context['attention_mask'].to(device), 
                )
                pos_doc_context_emb = itmodel.module.doc_embed(
                    batch_pos_doc_btypeid.to(device), 

                    batch_pos_doc_text['input_ids'].to(device), 
                    batch_pos_doc_text['attention_mask'].to(device), 

                    batch_pos_doc_edge4start_nodeindex.to(device), 
                    batch_pos_doc_edge4end_nodeindex.to(device), 

                    batch_pos_doc_edge_nodetok['input_ids'].to(device), 
                    batch_pos_doc_edge_nodetok['attention_mask'].to(device), 
                    batch_pos_doc_edge_position.to(device), 
                    batch_pos_doc_edge_hierarchy.to(device), 
                    batch_pos_doc_edge_attrtype.to(device), 

                    batch_mask_index.to(device), 
                )
                # score
                scores_pos = itmodel.module.score_pair(query_context_emb, pos_doc_context_emb)  # (batch, )
                scores_inbatch_neg = itmodel.module.score_inbatch_neg(query_context_emb, pos_doc_context_emb) # (batch, batch - 1)


                # loss
                scores_pos = scores_pos.reshape(-1, 1)
                
                scores = torch.cat(
                    [
                        scores_pos, 
                        scores_inbatch_neg, 
                    ], 
                    dim=1
                )
                labels_scores = torch.zeros(scores.shape[0], dtype=torch.long).to(scores.device)
                
                # loss
                loss = itmodel.module.criterion(scores / args.temp, labels_scores) 
                
                pos_neg_diff = scores_pos.mean() - scores_inbatch_neg.mean()


            # backward
            if scaler is not None:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(itmodel.parameters(), 2.0)
                
                scaler.step(optimizer)  
                scaler.update()
            else:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(itmodel.parameters(), 2.0)
                optimizer.step()

            # print
            if (idx_ + 1) % 100 == 0 and args.local_rank == 0:
                print(epoch, args.local_rank, (idx_ + 1) * args.batch_size, 'time', time.time() - stamp)
                print('rank:{}, loss:{}, pos_neg_diff:{}'.format(args.local_rank, loss.item(), pos_neg_diff.item()))
    
    
    print('done')

if __name__ == "__main__":
    # parameter
    parser = ArgumentParser()
    
    
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--mix_amp', default=1, type=int, help='apex')


    # data
    path_public = '/mnt/wfs/mmchongqingwfssz/meilang/vertical_search/'
    parser.add_argument('--path_longzhu_doc', type=str, default=path_public + "longzhu_doc/ft_local/")
    parser.add_argument('--path_normal_doc', type=str, default=path_public + "path_dict_qdr_rel_0.75_normal_docid_value.csv")
    parser.add_argument('--path_longzhu_log_train', type=str, default=path_public + "path_dict_qdr_rel_0.75_longzhu_train.csv")
    parser.add_argument('--path_normal_log_train', type=str, default=path_public + "path_dict_qdr_rel_0.75_normal_final_sampled_train.csv")



    
    # model
    parser.add_argument('--path_pretrain_model', type=str, default=path_public + 'roberta/')

    # checkpoint
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/')

    
    # run
    # contrastive temp
    parser.add_argument('--temp', type=float, default=1)
    # learning rate
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    # batch size
    parser.add_argument('--batch_size', type=int, default=32)
    # epoch
    parser.add_argument('--epoch', type=int, default=10)
    # trunc
    parser.add_argument('--trunc_length', type=int, default=300)
    # prompt_num
    parser.add_argument('--prompt_num', type=int, default=32)


    dict_btype2id = {}
    for btype in ('text', '10', '5', '14', '16', '3', '8', '999', '4', '9', '15'):
        dict_btype2id.setdefault(btype, len(dict_btype2id))
    print('dict_btype2id', len(dict_btype2id))


    # prompt_num
    parser.add_argument('--btype_num', type=int, default=len(dict_btype2id))
    parser.add_argument('--btype2id', type=dict, default=dict_btype2id)

    
    args = parser.parse_args()
    print(args)

    
    # main
    main(args)
