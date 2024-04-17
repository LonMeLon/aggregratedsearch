import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers import (
    BertForPreTraining, 
    #PreTrainedModel,
    #BertModel, 
    #BertGenerationEncoder, 
    #BertGenerationDecoder, 
    #EncoderDecoderModel,
    #EncoderDecoderConfig,
)

from transformers.models.bert.modeling_bert import BertForPreTrainingOutput

import torch.nn.functional as F
from torch_scatter import scatter_max, segment_csr, scatter_max
from torch_scatter import scatter

from typing import Callable








class graph_transformer(nn.Module):
    def __init__(
        self, 
        d_model, 
        nhead, 
        num_encoder_layers,
        dim_feedforward, 
        dropout, 
        activation,
    ):
        super(graph_transformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.encoder_layers = nn.ModuleList(
            [
                GraphEncoderLayer(
                    d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                )
                for i in range(num_encoder_layers)
            ]
        )
        self.encoder_norm = torch.nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, edge_index: torch.Tensor):
        if src.shape[1] != self.d_model:
            raise RuntimeError("the feature number of src must be equal to d_model")
        # Encode
        memory = src
        for mod in self.encoder_layers:
            memory = mod(memory, edge_index)
        memory = self.encoder_norm(memory)

        return memory


class GraphEncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model, 
        nhead, 
        dim_feedforward, 
        dropout, 
        activation,
    ):
        super(GraphEncoderLayer, self).__init__()
        self.attn = MultiheadAttention(d_model, num_heads=nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.activation = activation
        self.norm1 = nn.BatchNorm1d(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(
        self, 
        src: torch.Tensor, 
        edge_index: torch.Tensor, 
    ):
        # src: shape [N, d_model], (unique) node embedding
        # edge_index: shape [2,E], node-node edge connect
        src2 = self.attn(
            src, src, src, 
            torch.stack([edge_index[1], edge_index[0]], dim=0)
        )
        # Residual 1
        src = src + self.dropout1(src2)  # Residual 1
        src = self.norm1(src)
        # mlp
        src2 = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(src)
                )
            )
        )
        # Residual 2
        src = src + self.dropout2(src2)  # Residual 2
        src = self.norm2(src)
        
        return src


class MultiheadAttention(nn.Module):
    # Sparse MultiheadAttention
    def __init__(
        self, 
        embed_dim, 
        num_heads, 
        dropout, 
        bias=True, 
        add_bias_kv=False,
    ):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=add_bias_kv)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=add_bias_kv)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
        if self.k_proj.bias is not None:
            nn.init.xavier_normal_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.xavier_normal_(self.v_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)


    def sparse_softmax(self, src, index, num_nodes, dim=0):
        # Perform sparse softmax
        # scatter_max, same node (several score) to same group, softmax (max-normalize)
        src_max = scatter(src, index, dim=dim, reduce="max", dim_size=num_nodes) # 
        src_max = src_max.index_select(dim, index)
        out = (src - src_max).exp() # all edge node, un-normalized vector
        ### add the score ratio, normalize, according to same index
        out_sum = scatter(out, index, dim=dim, reduce="sum", dim_size=num_nodes)
        out_sum = out_sum.index_select(dim, index)

        return out / (out_sum + 1e-16)

    def forward(self, query, key, value, edge_index):
        r"""
        :param query: Tensor, shape [tgt_len, embed_dim]
        :param key: Tensor of shape [src_len, kdim]
        :param value: Tensor of shape [src_len, vdim]
        :param edge_index: Tensor of shape [2, E], a sparse matrix that has shape len(query)*len(key),
        :return Tensor of shape [tgt_len, embed_dim]
        """

        # Dimension checks
        assert edge_index.shape[0] == 2
        assert key.shape[0] == value.shape[0]
        # Dictionary size
        src_len, tgt_len, idx_len = key.shape[0], query.shape[0], edge_index.shape[1]

        assert query.shape[1] == self.embed_dim
        assert key.shape[1] == self.embed_dim
        assert value.shape[1] == self.embed_dim

        scaling = float(self.head_dim) ** -0.5
        q: torch.Tensor = self.q_proj(query) * scaling
        k: torch.Tensor = self.k_proj(key)
        v: torch.Tensor = self.v_proj(value)
        assert self.embed_dim == q.shape[1] == k.shape[1] == v.shape[1]

        # Split into heads
        q = q.contiguous().view(tgt_len, self.num_heads, self.head_dim)
        k = k.contiguous().view(src_len, self.num_heads, self.head_dim)
        v = v.contiguous().view(src_len, self.num_heads, self.head_dim)

        # Get score
        assert edge_index[0].shape == edge_index[1].shape
        attn_output_weights = (
            torch.index_select(q, 0, edge_index[0]) * torch.index_select(k, 0, edge_index[1])
        ).sum(dim=-1)
        
        # finite refered node
        assert list(attn_output_weights.size()) == [idx_len, self.num_heads]

        attn_output_weights = self.sparse_softmax(src=attn_output_weights, index=edge_index[0], num_nodes=tgt_len)
        attn_output_weights = self.dropout(attn_output_weights)

        # Get values
        # [edge in-node, n-head, dim]
        attn_output = attn_output_weights.unsqueeze(2) * torch.index_select(v, 0, edge_index[1])
        """Aggregation"""
        attn_output = scatter(attn_output, edge_index[0], dim=0, reduce="sum", dim_size=tgt_len)

        # all node in graph
        assert list(attn_output.size()) == [tgt_len, self.num_heads, self.head_dim]

        attn_output = self.out_proj(attn_output.contiguous().view(tgt_len, self.embed_dim))
        # all node in graph
        assert list(attn_output.size()) == [tgt_len, self.embed_dim]

        return attn_output
    

class structure(BertForPreTraining):
    def __init__(self, config, data_args, ):

        super().__init__(config)

        print("config:\n", config)


        self.args = data_args
        self.criterion = nn.CrossEntropyLoss()

        self.layernorm = nn.LayerNorm(config.hidden_size)
        
        self.shared_prompt = nn.Parameter(
            torch.rand(
                self.args.prompt_num,
                self.config.hidden_size,
            )
        )

        self.btype_prompt = nn.Parameter(
            torch.rand(
                self.args.btype_num,
                self.args.prompt_num,
                self.config.hidden_size,
            )
        )

        self.hierarchy_ = nn.Parameter(
            torch.rand(
                10,
                self.config.hidden_size,
            )
        )

        self.attrtype_ = nn.Parameter(
            torch.rand(
                2,
                self.config.hidden_size,
            )
        )



        self.structure2prompt = graph_transformer(
            d_model=768,
            nhead=12,
            num_encoder_layers=2,
            dim_feedforward=768,
            dropout=0.1,
            activation=nn.LeakyReLU(),
        )

        # weight initialized
        self.init_weights()

    def score_pair(self, Q, D):
        return torch.mul(Q, D).sum(1)
    
    def score_inbatch_neg(self, Q, D):
        assert Q.shape == D.shape
        bsize = Q.shape[0]
        return (torch.matmul(Q, D.T)).flatten()[1: ].view(bsize - 1, bsize + 1)[: ,: -1].reshape(bsize, bsize - 1) 
    
    def score_all(self, Q, D):
        return torch.matmul(Q, D.T) 
    
    
    def query_emd(
        self, 
        input_ids_context, 
        attention_mask_context, 
    ):
        emb_ = self.layernorm(
            self.bert(
                inputs_ids=input_ids_context, 
                attention_mask=attention_mask_context, 
            )[1]
        )

        return emb_
    

    def doc_embed(self, 
        doc_btypeid, 

        doc_text_input_ids, 
        doc_text_attention_mask, 

        doc_edge4start_nodeindex, 
        doc_edge4end_nodeindex, 

        doc_edge_nodetok_input_ids, 
        doc_edge_nodetok_attention_mask, 
        doc_edge_position, 
        doc_edge_hierarchy, 
        doc_edge_attrtype, 

        mask_index
    ):
        # text feature
        bsize = doc_btypeid.shape[0]

        
        # embedding & attention
        text_emb_ = self.bert.embeddings.word_embeddings(doc_text_input_ids) 
        text_cls_emb = text_emb_[:, 0:1, :] # (batch, 1, emb)
        text_rest_emb = text_emb_[:, 1:, :] # (batch, len - 1, emb)
        text_cls_mask = doc_text_attention_mask[:, 0:1]
        text_rest_mask = doc_text_attention_mask[:, 1:]

        shared_prompt_emb = self.shared_prompt\
            .reshape(1, self.args.prompt_num, self.config.hidden_size)\
            .repeat(bsize, 1, 1)
        vertical_prompt_emb = self.btype_prompt(doc_btypeid)\
            .reshape(1, self.args.prompt_num, self.config.hidden_size)\
            .repeat(bsize, 1, 1)
        shared_prompt_mask = torch.ones(
            (bsize, self.args.prompt_num)
        )\
        .long()\
        .to(text_emb_.device)
        vertical_prompt_mask = torch.ones(
            (bsize, self.args.prompt_num)
        )\
        .long()\
        .to(text_emb_.device)


        # graph
        embs_doc_edge_nodetok_input_ids = self.bert.embeddings.word_embeddings(doc_edge_nodetok_input_ids) 
        embs_doc_edge_position = self.bert.embeddings.position_embeddings(doc_edge_position) 
        embs_doc_edge_hierarchy = self.hierarchy_(doc_edge_hierarchy)
        embs_doc_edge_attrtype = self.attrtype_(doc_edge_attrtype) 


        mask_graph_embs = embs_doc_edge_nodetok_input_ids + embs_doc_edge_position + embs_doc_edge_hierarchy + embs_doc_edge_attrtype
        mask_graph_embs = self.graph_transformer(mask_graph_embs, [doc_edge4start_nodeindex, doc_edge4end_nodeindex])
        mask_graph_embs = mask_graph_embs[mask_index]

        mask_graph_attention_mask = torch.ones(
            (mask_graph_embs.shape[0], mask_graph_embs.shape[1])
        )\
        .long()\
        .to(text_emb_.device)

        # merge 
        merge_emb = torch.cat(
            [text_cls_emb, shared_prompt_emb, vertical_prompt_mask, mask_graph_embs, text_rest_emb], 
            dim=1,
        )
        merge_mask = torch.cat(
            [text_cls_emb, shared_prompt_emb, vertical_prompt_mask, mask_graph_attention_mask, text_rest_emb], 
            dim=1,
        )

        emb_ = self.layernorm(
            self.bert(
                inputs_embeds=merge_emb, 
                attention_mask=merge_mask, 
            )[1]
        )

        return emb_
