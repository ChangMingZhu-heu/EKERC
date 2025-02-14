

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
from transformers import AlbertTokenizer
from transformers import RobertaTokenizer
import json
from prepare_glove import load_w2v, tokenize_glove
from senticnet.senticnet import SenticNet
import re
from gensim.parsing.preprocessing import remove_stopwords
from configs import inputconfig_func
Configs = inputconfig_func()

torch.random.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.manual_seed(1234)

def tokenize(data, tokenizer, MAX_L=20, model_type='roberta_large'):
    input_ids = {}
    masks = {}
    token_types = {}
    keys = data.keys()
    count=0

    for key in keys:
        dial = data[key]
        dial = re.sub(r'\x92', '', dial)
        dial = re.sub(r'\x91', '', dial)
        res = tokenizer(dial, padding='longest', return_tensors='pt')
        if res['input_ids'].size(1) > 512:
            pun = '!"#$%&\'()*+,-.:;=?@[\\]^_`{|}~'
            dial = re.sub(r'[{}]+'.format(pun), '', dial)
            res = tokenizer(dial, padding='longest', return_tensors='pt')
            count +=1
            print(res['input_ids'].size(1))
        input_ids[key] = res['input_ids']
        masks[key] = res['attention_mask']
        token_types[key] = []
        if model_type == 'albert':
            token_types[key] = res['token_type_ids']

    return input_ids, masks, token_types


def tokenize_daily(data, tokenizer, MAX_L=30, model_type='roberta_large'):
    input_ids = {}
    masks = {}
    token_types = {}
    keys = data.keys()
    count=0
    for key in keys:
        dial = data[key]

        res = tokenizer(dial, padding='longest', return_tensors='pt')
        if res['input_ids'].size(1) > 512:
            pun = '!"#$%&\'()*+,-.:;=?@[\\]^_`{|}~'
            dial = re.sub(r'[{}]+'.format(pun), '', dial)
            dial = remove_stopwords(dial)
            res = tokenizer(dial, padding='longest', return_tensors='pt')
            count +=1
        input_ids[key] = res['input_ids']
        masks[key] = res['attention_mask']
        token_types[key] = []
        if model_type == 'albert':
            token_types[key] = res['token_type_ids']

    print(count)

    return input_ids, masks, token_types


def tokenize_emory(data, tokenizer, MAX_L=20, model_type='roberta_large'):
    input_ids = {}
    masks = {}
    token_types = {}
    keys = data.keys()
    count=0

    for key in keys:
        dial = data[key]
        dial = re.sub(r'\x92', '', dial)
        dial = re.sub(r'\x91', '', dial)

        res = tokenizer(dial, padding='longest', return_tensors='pt')
        if res['input_ids'].size(1) > 512:
            pun = '!"#$%&\'()*+,-.:;=?@[\\]^_`{|}~'
            dial = re.sub(r'[{}]+'.format(pun), '', dial)
            res = tokenizer(dial, padding='longest', return_tensors='pt')
            count +=1
            print(res['input_ids'].size(1))
        input_ids[key] = res['input_ids']
        masks[key] = res['attention_mask']
        token_types[key] = []
        if model_type == 'albert':
            token_types[key] = res['token_type_ids']

    return input_ids, masks, token_types
def concate_sen(sentences, speakers, tokenizer):
    concatenated_sentences = {}
    keys = sentences.keys()
    concatenated = ''
    max_sp = 0
    for key in keys:
        dialog = sentences[key]
        speaker = speakers[key]
        for idx, (sp, dial) in enumerate(zip(speaker, dialog)):
            sp_index = sp.index(1)
            tokenizer.additional_special_tokens[sp_index]
            if idx == 0:
                concatenated += tokenizer.additional_special_tokens[sp_index] + dial
            else:
                concatenated += tokenizer.sep_token + tokenizer.additional_special_tokens[sp_index] + dial
        concatenated_sentences[key] = concatenated
        concatenated = ''

    return concatenated_sentences
def occupationprocess(occupations, speakers,tokenizer):
    concatenated_sentences = {}
    keys = occupations.keys()
    concatenated = ''
    max_sp = 0
    for key in keys:
        occupation = occupations[key]
        speaker = speakers[key]
        for idx, (sp, occupa) in enumerate(zip(speaker, occupation)):
            sp_index = sp.index(1)
            if idx == 0:
                concatenated += tokenizer.additional_special_tokens[sp_index] + occupa
            else:
                concatenated += tokenizer.sep_token + tokenizer.additional_special_tokens[sp_index] + occupa
        concatenated_sentences[key] = concatenated
        concatenated = ''
    return concatenated_sentences
def concate_userinfo(genders,occupations,characters, speakers, tokenizer):
    concatenated_sentences = {}
    keys = genders.keys()

    concatenated = ''

    max_sp = 0

    for key in keys:
        gender = genders[key]
        occupation = occupations[key]
        character = characters[key]
        speaker = speakers[key]

        for idx, (gen,occupa,chara,sp) in enumerate(zip(gender,occupation,character,speaker)):
            sp_index = sp.index(1)
            tokenizer.additional_special_tokens[sp_index]
            if idx == 0:
                concatenated += tokenizer.additional_special_tokens[sp_index]+str(gen)+occupa+chara

            else:
                concatenated += tokenizer.sep_token + tokenizer.additional_special_tokens[sp_index]+str(gen)+occupa+chara
        concatenated_sentences[key] = concatenated

        concatenated = ''

    return concatenated_sentences
def concate_senwithgender(sentences, genders,tokenizer):
    concatenated_sentences = {}
    keys = sentences.keys()
    concatenated = ''

    max_sp = 0

    for key in keys:
        dialog = sentences[key]
        gender = genders[key]

        for idx, (gen, dial) in enumerate(zip(gender, dialog)):
            gen_index=2
            if gen==0:
                gen_index = 0
            elif gen==1:
                gen_index=1
            tokenizer.additional_special_tokens[gen_index]
            if idx == 0:
                concatenated += tokenizer.additional_special_tokens[gen_index]+ dial

            else:
                concatenated += tokenizer.sep_token + tokenizer.additional_special_tokens[gen_index]+dial
        concatenated_sentences[key] = concatenated

        concatenated = ''

    return concatenated_sentences

def concate_st(styles, speakers, tokenizer):
    concatenated_sentences = {}

    keys = styles.keys()

    concatenated = ''

    max_sp = 0

    for key in keys:
        style = styles[key]
        speaker = speakers[key]

        for idx, (sty,sp) in enumerate(zip(style,speaker)):
            sp_index = sp.index(1)
            if idx == 0:
                concatenated += tokenizer.additional_special_tokens[sp_index]+sty

            else:
                concatenated += tokenizer.sep_token + tokenizer.additional_special_tokens[sp_index]+sty
        concatenated_sentences[key] = concatenated

        concatenated = ''

    return concatenated_sentences

def concate_stword(topics,topicwords,styles, speakers, tokenizer):
    concatenated_sentences = {}

    keys = topics.keys()

    concatenated = ''

    max_sp = 0

    for key in keys:
        topic = topics[key]
        topicword = topicwords[key]
        style = styles[key]
        speaker = speakers[key]

        for idx, (top,tow,sty,sp) in enumerate(zip(topic,topicword,style,speaker)):
            sp_index = sp.index(1)
            if idx == 0:
                concatenated += tokenizer.additional_special_tokens[sp_index]+str(top)+tokenizer.sep_token+str(tow)+tokenizer.sep_token+sty

            else:
                concatenated += tokenizer.sep_token + tokenizer.additional_special_tokens[sp_index]+str(top)+tokenizer.sep_token+str(tow)+tokenizer.sep_token+sty
        concatenated_sentences[key] = concatenated

        concatenated = ''

    return concatenated_sentences

def concate_sen_daily(sentences, speakers, tokenizer):
    concatenated_sentences = {}
    keys = sentences.keys()
    concatenated = ''

    for key in keys:
        dialog = sentences[key]
        speaker = speakers[key]
        for idx, (sp, dial) in enumerate(zip(speaker, dialog)):
            if sp == '0':
                sper = tokenizer.additional_special_tokens[0]
            elif sp == '1':
                sper = tokenizer.additional_special_tokens[1]
            else:
                print('speaker error!')
            if idx == 0:
                concatenated += sper + dial
            else:
                concatenated += tokenizer.sep_token + sper + dial
        concatenated_sentences[key] = concatenated
        concatenated = ''

    return concatenated_sentences


def concate_sen_emory(sentences, speakers, tokenizer):
    concatenated_sentences = {}
    keys = sentences.keys()
    concatenated = ''
    max_sp = 0
    for key in keys:
        dialog = sentences[key]
        speaker = speakers[key]
        for idx, (sp, dial) in enumerate(zip(speaker, dialog)):
            sp_index = sp.index(1)
            tokenizer.additional_special_tokens[sp_index]
            if idx == 0:
                concatenated += tokenizer.additional_special_tokens[sp_index] + dial
            else:
                concatenated += tokenizer.sep_token + tokenizer.additional_special_tokens[sp_index] + dial
        concatenated_sentences[key] = concatenated
        concatenated = ''

    return concatenated_sentences


def prepare_graph(structure):
    src = {}
    dst = {}
    edge_type = {}

    keys = structure.keys()


    for key in keys:
        src[key] = []
        dst[key] = []
        edge_type[key] = []

        dial = structure[key]

        for utt in dial:
            src[key].append(utt['x'])
            dst[key].append(utt['y'])
            edge_type[key].append(utt['type_num'])

    return src, dst, edge_type


def gen_cpt_vocab(sn, dst_num_per_rel=Configs.dst_num_per_rel):
    cpt_vocab = ['<pad>']

    rel_list = ['isa', 'causes', 'hascontext']

    dicts = []

    for rel in rel_list:
        with open('./data/dialog_concept/{}_weight_dict_all.json'.format(rel), 'r', encoding='utf-8') as f:
            rel_dict_origin = json.load(f)

        f.close()

        rel_dict = {}

        for key in rel_dict_origin.keys():
            dst = rel_dict_origin[key]

            weights = [item[1] for item in dst]

            weights_scaled = min_max_scale(weights)

            dst_ = []

            for idx, item in enumerate(dst):

                try:
                    dst_.append([item[0], weights_scaled[idx], abs(float(sn.polarity_value(item[0])))])

                except KeyError:
                    dst_.append([item[0], weights_scaled[idx], 0.])
            dst_.sort(key =lambda i: i[1]+i[2], reverse=True)

            l_idx = min(len(dst_), dst_num_per_rel)

            rel_dict[key] = dst_[0:l_idx]


        keys = rel_dict.keys()
        dicts.append(rel_dict)

        for key in keys:
            if key not in cpt_vocab:
                cpt_vocab.append(key)
            for v in rel_dict[key]:
                if v[0] not in cpt_vocab:
                    cpt_vocab.append(v[0])
    return cpt_vocab, dicts



def tok_cpt_vocab(cpt_vocab, max_length=5):
    word_idx_rev, word_idx = load_w2v(100, './glove/', cpt_vocab)

    cpt_ids = []

    for line in cpt_vocab:

        cpt_ids.append(torch.LongTensor(tokenize_glove(word_idx, line)[:max_length]))


    cpt_ids = pad_sequence(cpt_ids, batch_first=True)

    sel_mask = cpt_ids.ne(0)

    pad_cpt_ids = {'input_ids': cpt_ids, 'sel_mask': sel_mask}


    return pad_cpt_ids
#bycmz
def tok_cpt_vocab_(tokenizer, cpt_vocab, cuda=False):
    cpt_ids = tokenizer(cpt_vocab, max_length=5, padding='max_length', truncation=True)

    pad_cpt_ids = {}
    pad_cpt_ids['sel_mask'] = []
    keys = cpt_ids.keys()
    for key in keys:
        items = []
        for item in cpt_ids[key]:
            if key == 'attention_mask':
                sel_mask = item[:]
                sel_mask[0] = 0
                sel_mask[-1] = 0
                items.append(torch.LongTensor(sel_mask))
            else:
                items.append(torch.LongTensor(item))

        pads = pad_sequence(items, batch_first=True)
        pad_cpt_ids[key] = pads.cuda() if cuda else pads

    sel_masks = pad_sequence(pad_cpt_ids['sel_mask'], batch_first=True)
    pad_cpt_ids['sel_mask'] = sel_masks.cuda() if cuda else sel_masks

    return pad_cpt_ids

# dst_num = num_rel * dst_num_per_rel
def gen_cpt_graph(text, cpt_vocab, isa_dict, causes_dict, hscnt_dict, sn, src_num=Configs.src_num, dst_num=Configs.dst_num_per_rel*3):
    graph = {}

    keys = text.keys()


    for key in keys:  # key: dialogue index
        dial = text[key]

        srcs = []
        dsts = []
        weights = []
        sentics = []
        rel_types = []
        masks = []
        for idx, utt in enumerate(dial):

            tokens = utt.strip().split()

            src = []
            traversed_src = []
            src_pos = 0
            dst_ = []
            weight_ = []
            sentic_ = []
            rel_type_ = []
            mask_ = []

            for token in tokens:

                dst = [0 for _ in range(dst_num)]
                weight = [0. for _ in range(dst_num)]
                sentic = [0. for _ in range(dst_num)]
                rel_type = [0 for _ in range(dst_num)]
                mask = [0 for _ in range(dst_num)]
                # retrieve cpt dst and rel for each token
                if token in isa_dict:
                    isa_res = isa_dict[token]
                else:
                    isa_res = []
                if token in causes_dict:
                    causes_res = causes_dict[token]
                else:
                    causes_res = []
                if token in hscnt_dict:
                    hscnt_res = hscnt_dict[token]
                else:
                    hscnt_res = []
                res = [isa_res, causes_res, hscnt_res]

                if token not in traversed_src and (token in isa_dict or token in causes_dict or token in hscnt_dict):
                    try:
                        src.append([cpt_vocab.index(token), float(sn.polarity_value(token)), src_pos])
                    except KeyError:
                        src.append([cpt_vocab.index(token), 0., src_pos])
                    traversed_src.append(token)
                    src_pos += 1

                dst_count = 0
                for idx_r, res_ in enumerate(res):

                    for e in res_:
                        # dst.append(cpt_vocab.index(e[0]))
                        # weight.append(e[1])
                        # rel_type.append(idx_r)
                        dst[dst_count]= cpt_vocab.index(e[0])
                        weight[dst_count] = e[1]
                        rel_type[dst_count] = idx_r
                        mask[dst_count] = 1

                        # retrieve polarity_value from senticnet for each dst_node
                        e_ = [tok.lower() for tok in e[0].split(' ')]
                        score = []
                        for tok in e_:
                            try:
                                score.append(float(sn.polarity_value(tok)))
                            except KeyError:
                                pass
                        # sentic.append(sum(score) / len(score) if len(score) > 0 else 0.)
                        sentic[dst_count] = (sum(score) / len(score) if len(score) > 0 else 0.)
                        dst_count += 1

                weight_scaled = min_max_scale(weight)
                if sum(dst)>0:
                    dst_.append(dst)
                    weight_.append(weight_scaled)
                    sentic_.append(sentic)
                    rel_type_.append(rel_type)
                    mask_.append(mask)

            src.sort(key=lambda i: i[1], reverse=True)

            l_idx = min(len(src), src_num)

            srcs.append(torch.LongTensor([item[0] for item in src[:l_idx]]))

            dsts.append([dst_[item[2]] for item in src[:l_idx]])

            weights.append([weight_[item[2]] for item in src[:l_idx]])

            sentics.append([sentic_[item[2]] for item in src[:l_idx]])

            rel_types.append([rel_type_[item[2]] for item in src[:l_idx]])

            masks.append([mask_[item[2]] for item in src[:l_idx]])

        # padding info
        srcs = pad_sequence(srcs, batch_first=True, padding_value=0)
        bz, max_n_srcs = srcs.size()

        for d, w, s, r, m in zip(dsts, weights, sentics, rel_types, masks):
            num_item = len(d)
            if num_item < max_n_srcs:
                for i in range(max_n_srcs-num_item):
                    d.append([0 for _ in range(dst_num)])
                    w.append([0. for _ in range(dst_num)])
                    s.append([0. for _ in range(dst_num)])
                    r.append([0 for _ in range(dst_num)])
                    m.append([0 for _ in range(dst_num)])
        src_masks = torch.sum(torch.LongTensor(masks), dim=-1)
        for m,src_m in zip(masks, src_masks):
            if len(m)>0:
                for i in range(src_m.size(0)):
                    m[i][0] = 1

        dial_graph = [torch.LongTensor(srcs), torch.LongTensor(dsts),
                      torch.FloatTensor(weights), torch.FloatTensor(sentics),
                      src_masks, torch.LongTensor(masks), torch.LongTensor(rel_types)]
        graph[key] = dial_graph

    return graph


def get_chunk(cpt_graph_i, cpt_ids, model_type='roberta_large', chunk_size=10, dst_num=Configs.dst_num_per_rel*3):

    srcs, dsts, weights, sentics, src_masks, masks, rels = cpt_graph_i
    if masks.sum() == 0:
        bz, seq_len = srcs.size()
        utt_idx = torch.arange(bz).to(srcs.device)
        return [[srcs, srcs, srcs, dsts, dsts, dsts, weights, sentics, src_masks, masks, rels, utt_idx]]
    bz, seq_len = srcs.size()
    utt_idx = torch.arange(bz).to(srcs.device)
    num_chunck = bz//chunk_size+1 if bz%chunk_size>0 else bz//chunk_size
    srcs = srcs.contiguous().view(bz*seq_len, -1)
    srcs_input_ids = cpt_ids['input_ids'][srcs].contiguous().view(bz, seq_len, -1)

    # srcs_attention_mask = cpt_ids['attention_mask'][srcs].contiguous().view(bz, seq_len, -1)
    srcs_sel_mask = cpt_ids['sel_mask'][srcs].contiguous().view(bz, seq_len, -1)
    dsts = dsts.contiguous().view(bz*seq_len*dst_num,-1)
    dsts_input_ids = cpt_ids['input_ids'][dsts].contiguous().view(bz, seq_len, dst_num, -1)

    # dsts_attention_mask = cpt_ids['attention_mask'][dsts].contiguous().view(bz, seq_len, dst_num, -1)
    dsts_sel_mask = cpt_ids['sel_mask'][dsts].contiguous().view(bz, seq_len, dst_num, -1)
    srcs_input_ids_chunked = srcs_input_ids.chunk(num_chunck)

    # srcs_attention_mask_chunked = srcs_attention_mask.chunk(num_chunck)
    srcs_sel_mask_chunked = srcs_sel_mask.chunk(num_chunck)
    dsts_input_ids_chunked = dsts_input_ids.chunk(num_chunck)

    # dsts_attention_mask_chunked = dsts_attention_mask.chunk(num_chunck)
    dsts_sel_mask_chunked = dsts_sel_mask.chunk(num_chunck)
    utt_idx_chunked = utt_idx.chunk(num_chunck)
    weights_chunked = weights.chunk(num_chunck)
    sentics_chunked = sentics.chunk(num_chunck)
    src_masks_chunked = src_masks.chunk(num_chunck)
    masks_chunked = masks.chunk(num_chunck)
    rels_chunked = rels.chunk(num_chunck)
    if model_type == 'albert':
        #yuanxian
        srcs_token_type_ids = cpt_ids['token_type_ids'][srcs]       #出现bug keyerror
        dsts_token_type_ids = cpt_ids['token_type_ids'][dsts]
        srcs_token_type_ids_chunked = srcs_token_type_ids.chunk(num_chunck)
        dsts_token_type_ids_chunked = dsts_token_type_ids.chunk(num_chunck)
        # srcs_token_type_ids_chunked = [[] for _ in range(num_chunck)]
        # dsts_token_type_ids_chunked = [[] for _ in range(num_chunck)]
    elif model_type in ['roberta', 'roberta_large']:
        srcs_token_type_ids_chunked = [[] for _ in range(num_chunck)]
        dsts_token_type_ids_chunked = [[] for _ in range(num_chunck)]

    dial = []
    for i in range(num_chunck):
        chunk_content = [srcs_input_ids_chunked[i], srcs_token_type_ids_chunked[i],
                         srcs_sel_mask_chunked[i], dsts_input_ids_chunked[i], dsts_token_type_ids_chunked[i],
                         dsts_sel_mask_chunked[i], weights_chunked[i],
                         sentics_chunked[i],
                         src_masks_chunked[i], masks_chunked[i], rels_chunked[i], utt_idx_chunked[i]]
        dial.append(chunk_content)

    return dial


# deprecated
def cpt_graph(text, cpt_vocab, rel_dict_ids, sn, MAX_L=20):
    graph = {}
    keys = text.keys()
    for key in keys:
        dial = text[key]
        srcs = []
        dsts = []
        weights = []
        sentics = []

        dst_ = []
        weight_ = []

        for idx, utt in enumerate(dial):
            words = utt.strip().split()
            src = []
            dst_ = []
            weight_ = []
            sentic_ = []


            for word in words:#[:MAX_L]:
                dst = []
                weight = []
                sentic = []
                if word in rel_dict_ids:
                    res = rel_dict_ids[word]
                    # src.append([cpt_vocab.index(word)])
                    try:
                        src.append([cpt_vocab.index(word), float(sn.polarity_value(word))])
                    except KeyError:
                        src.append([cpt_vocab.index(word), 0.])
                    for e in res:
                        # src.append(cpt_vocab.index(word))
                        dst.append(cpt_vocab.index(e[0]))
                        weight.append(e[1])

                        # retrieve polarity_value from senticnet for each dst_node
                        e_ = [tok.lower() for tok in e[0].split(' ')]
                        score = []
                        for tok in e_:
                            try:
                                score.append(float(sn.polarity_value(tok)))
                            except KeyError:
                                pass
                        sentic.append(sum(score) / len(score) if len(score) > 0 else 0.)

                    weight_scaled = min_max_scale(weight)
                    # sentic_scaled = min_max_scale(sentic)
                    dst_.append(dst)
                    weight_.append(weight_scaled)
                    # sentic_.append(sentic_scaled)
                    sentic_.append(sentic)
            src.sort(key = lambda i: i[1], reverse=True)
            l_idx = min(len(src), Configs.src_num)
            srcs.append([[item[0]] for item in src[:l_idx]])

            dsts.append(dst_)
            weights.append(weight_)
            sentics.append(sentic_)

        dial_graph = [srcs, dsts, weights, sentics]
        graph[key] = dial_graph
    return graph


# deprecated
def merge(isa_graph, causes_graph, hcontext_graph):
    keys = isa_graph.keys()
    agg_graph = {}
    # isa_src, causes_src, hcontext_src = isa_graph[], causes_graph, hcontext_graph
    for key in keys:
        # dial_src = [isa_graph[key][0], causes_graph[key][0], hcontext_graph[key][0]]
        # dial_dst = [isa_graph[key][1], causes_graph[key][1], hcontext_graph[key][1]]
        # agg_dial_src = []
        # agg_dial_dst = []
        utt_srcs = []
        utt_dsts = []
        utt_srcs_origin = []
        for idx, utt in enumerate(zip(isa_graph[key][0], causes_graph[key][0], hcontext_graph[key][0])):

            # if len(utt[0]) + len(utt[1]) + len(utt[2]) == 0:
            #     continue
            utt_srcs.append([])
            utt_dsts.append([])
            utt_srcs_origin.append([])
            for idx_i, nodes in enumerate(utt):
                for idx_j, nd in enumerate(nodes):
                    # if len(nd) == 0:
                    #     continue
                    if nd not in utt_srcs_origin[idx]:
                        utt_srcs[idx].append([idx_i, idx_j])
                        utt_srcs_origin[idx].append(nd)
                        # for idx_k in range(len(dial_dst[idx][idx_i][idx_j])):
                        utt_dsts[idx].append(
                            [[idx_i, idx_j]])  # idx_i: relation_type_id, idx_j: node positinon in each nodes
                    else:
                        nd_pos = utt_srcs_origin[idx].index(nd)
                        utt_dsts[idx][nd_pos].append([idx_i, idx_j])
            utt_srcs[idx] = [torch.LongTensor(item) for item in utt_srcs[idx]]
            utt_dsts[idx] = [torch.LongTensor(item) for item in utt_dsts[idx]]
        agg_graph[key] = [utt_srcs, utt_dsts]
    return agg_graph


def min_max_scale(v, new_min=0, new_max=1.0):
    if len(v) == 0:
        return v
    v_min, v_max = min(v), max(v)
    if v_min == v_max:
        v_p = [new_max for e in v]
    else:
        v_p = [(e - v_min) / (v_max - v_min) * (new_max - new_min) + new_min for e in v]

    return v_p


class MELDDataset(Dataset):

    def __init__(self, path, n_classes=7, MAX_L=20, train=True, cuda=True, model_type='roberta_large'):

        if model_type == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        elif model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif model_type == 'roberta_large':
            pathc = './roberta_large'
            self.tokenizer = RobertaTokenizer.from_pretrained(pathc)
        sn = SenticNet()

        if n_classes == 3:
            self.videoIDs_, self.videoSpeakers_, _, self.videoText_, \
            self.videoAudio_, self.videoSentence_, self.trainVid, \
            self.testVid, self.videoLabels_, self.structure_, self.action_ = pickle.load(open(path, 'rb'))
            self.Speakergender_=pickle.load(open('./data/meld/gender.pickle', 'rb'))
            self.Speakeroccupution_=pickle.load(open('./data/meld/occupation.pickle', 'rb'))
            self.Speakercharacter_ = pickle.load(open('./data/meld/character.pickle', 'rb'))
            self.Timestructure_ = pickle.load(open('./data/meld/timestructure.pkl', 'rb'))
            self.style_ = pickle.load(
                open('./data/meld/style.pickle', 'rb'))
            self.topic_ = pickle.load(
                open('./data/meld/topicclass7.pickle', 'rb'))
            # topic的类别是超参数
        elif n_classes == 7:
            self.videoIDs_, self.videoSpeakers_, self.videoLabels_, self.videoText_, \
            self.videoAudio_, self.videoSentence_, self.trainVid, \
            self.testVid, _, self.structure_, self.action_ = pickle.load(open(path, 'rb'))
            #在这里加上加载说话者属性信息bycmz
            self.Speakergender_=pickle.load(open('./data/meld/gender.pickle', 'rb'))
            self.Speakeroccupution_=pickle.load(open('./data/meld/occupation.pickle', 'rb'))
            self.Speakercharacter_ = pickle.load(open('./data/meld/character.pickle', 'rb'))
            self.Timestructure_ = pickle.load(open('./data/meld/timestructure.pkl', 'rb'))
            #男1女0未知0.5
            # 在这里加上说话者风格和topic类别和主题词
            self.style_ = pickle.load(
                open('./data/meld/style.pickle', 'rb'))
            self.topic_ = pickle.load(
                open('./data/meld/topicclass7.pickle', 'rb'))
            # topic的类别是超参数
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        self.cpt_vocab, [isa_dict, causes_dict, hascnt_dict] = gen_cpt_vocab(sn)
        self.cpt_ids = tok_cpt_vocab(self.cpt_vocab)
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoSentence, self.structure, self.Timestructure,self.action ,self.Speakergender,self.Speakeroccupution,self.Speakercharacter,self.topic,self.style= [self.partition(data) for data in
                                                                            [self.videoIDs_, self.videoSpeakers_,
                                                                             self.videoLabels_, self.videoText_,
                                                                             self.videoAudio_, self.videoSentence_,
                                                                             self.structure_,self.Timestructure_,self.action_,self.Speakergender_,self.Speakeroccupution_,self.Speakercharacter_,self.topic_,self.style_]]


        special_tokens_dict = {'additional_special_tokens': ['</s0>', '</s1>', '</s2>', '</s3>', '</s4>', '</s5>', '</s6>', '</s7>', '</s8>']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        userinfo = concate_userinfo(self.Speakergender,self.Speakeroccupution,self.Speakercharacter,self.videoSpeakers,self.tokenizer)
        self.occu_ids, self.occu_masks, self.occu_token_types = tokenize(userinfo, self.tokenizer, MAX_L=MAX_L,
                                                               model_type=model_type)


        styletopicinfo = concate_st(self.style,self.videoSpeakers,self.tokenizer)

        self.styletopic_ids, self.styletopic_masks, self.styletopic_token_types = tokenize(styletopicinfo, self.tokenizer, MAX_L=MAX_L,
                                                               model_type=model_type)
        concated_sen = concate_sen(self.videoSentence, self.videoSpeakers, self.tokenizer)
        self.sent_ids, self.masks, self.token_types = tokenize(concated_sen, self.tokenizer, MAX_L=MAX_L,
                                                               model_type=model_type)

        self.node_src, self.node_dst, self.edge_type = prepare_graph(self.structure)


        self.timenode_src, self.timenode_dst, self.timeedge_type = prepare_graph(self.Timestructure)

        self.cpt_graph = gen_cpt_graph(self.videoSentence, self.cpt_vocab, isa_dict, causes_dict, hascnt_dict,sn)



        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return self.sent_ids[vid], \
               self.masks[vid], \
               self.token_types[vid], \
               self.occu_ids[vid], \
               self.occu_masks[vid], \
               self.occu_token_types[vid], \
               self.styletopic_ids[vid], \
               self.styletopic_masks[vid], \
               self.styletopic_token_types[vid], \
               self.cpt_graph[vid],\
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor(self.videoSpeakers[vid]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               torch.LongTensor(self.node_src[vid]), \
               torch.LongTensor(self.node_dst[vid]), \
               torch.LongTensor(self.edge_type[vid]), \
               torch.LongTensor(self.timenode_src[vid]), \
               torch.LongTensor(self.timenode_dst[vid]), \
               torch.LongTensor(self.timeedge_type[vid]), \
               vid
    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        # return [dat[i] for i in dat]
        result = []

        for i in dat:

            if i < 9:
                result.append(dat[i][0])
            elif i < 10:
                result.append(dat[i][0])
            elif i < 14:
                result.append(pad_sequence([dat[i][0]], True))
            elif i < 20:
                #graph
                result.append(dat[i][0])
            else:
                result.append(dat[i].tolist())
        return result
    def partition(self, data):

        return {key: data[key] for key in self.keys}



class EmoryNLPDataset(Dataset):

    def __init__(self, split, path, n_classes=7, MAX_L=20, cuda=True, model_type='albert'):

        '''
                label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
                '''

        if model_type == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        elif model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif model_type == 'roberta_large':
            pathc = './roberta_large'
            self.tokenizer = RobertaTokenizer.from_pretrained(pathc)
        sn = SenticNet()
        with open(path, 'rb') as f:
            self.speakers_, self.emotion_labels_, self.sentences_, self.trainId, self.testId, self.validId, self.structure_ = pickle.load(f)
        f.close()
        with open('./data/EMORY/emorynlpgender.pickle', 'rb') as f:
            self.Speakergender_=pickle.load(f)
        f.close()
        with open('./data/EMORY/emorynlpoccupation.pickle', 'rb') as f:
            self.Speakeroccupution_=pickle.load(f)
        f.close()
        with open('./data/EMORY/emorynlpcharacter.pickle', 'rb') as f:
            self.Speakercharacter_=pickle.load(f)
        f.close()
        with open('./data/EMORY/emorystyle.pickle', 'rb') as f:
            self.style_=pickle.load(f)
        f.close()
        with open('./data/EMORY/emorytimestructure.pickle', 'rb') as f:
            self.Timestructure_ = pickle.load(f)
        f.close()

        sentiment_labels = {}
        for item in self.emotion_labels_:
            array = []
            # 0 negative, 1 neutral, 2 positive
            for e in self.emotion_labels_[item]:
                if e in [1, 4, 6]:
                    array.append(0)
                elif e == 3:
                    array.append(1)
                elif e in [0, 2, 5]:
                    array.append(2)
            sentiment_labels[item] = array

        if n_classes == 7:
            self.labels_ = self.emotion_labels_
        elif n_classes == 3:
            self.labels_ = sentiment_labels


        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.cpt_vocab, [isa_dict, causes_dict, hascnt_dict] = gen_cpt_vocab(sn)
        # deprecated
        # self.cpt_ids = tok_cpt_vocab(self.tokenizer, self.cpt_vocab, cuda=cuda)
        self.cpt_ids = tok_cpt_vocab(self.cpt_vocab)
        # self.speakers, self.labels, self.structure, self.sentence,self.Speakergender,self.Speakeroccupution,self.Speakercharacter ,self.Timestructure\
        #     = [self.partition(data) for data in [self.speakers_, self.labels_, self.structure_, self.sentences_,self.Speakergender_,self.Speakeroccupution_,self.Speakercharacter_,self.Timestructure_]]

        self.speakers, self.labels, self.structure, self.sentence,self.Speakergender,self.Speakeroccupution,self.Speakercharacter ,self.style,self.Timestructure\
            = [self.partition(data) for data in [self.speakers_, self.labels_, self.structure_, self.sentences_,self.Speakergender_,self.Speakeroccupution_,self.Speakercharacter_,self.style_,self.Timestructure_]]

        special_tokens_dict = {
            'additional_special_tokens': ['</s0>', '</s1>', '</s2>', '</s3>', '</s4>', '</s5>', '</s6>', '</s7>','</s8>']

        }
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        userinfo = concate_userinfo(self.Speakergender,self.Speakeroccupution,self.Speakercharacter,self.speakers,self.tokenizer)
        self.occu_ids, self.occu_masks, self.occu_token_types = tokenize(userinfo, self.tokenizer, MAX_L=MAX_L,
                                                               model_type=model_type)

        concated_sen = concate_sen_emory(self.sentence, self.speakers, self.tokenizer)

        self.sent_ids, self.masks, self.token_types = tokenize_emory(concated_sen, self.tokenizer, MAX_L=MAX_L,
                                                               model_type=model_type)
        styletopicinfo = concate_st(self.style,self.speakers,self.tokenizer)
        self.styletopic_ids, self.styletopic_masks, self.styletopic_token_types = tokenize(styletopicinfo, self.tokenizer, MAX_L=MAX_L,
                                                               model_type=model_type)

        self.node_src, self.node_dst, self.edge_type = prepare_graph(self.structure)

        self.timenode_src, self.timenode_dst, self.timeedge_type = prepare_graph(self.Timestructure)

        self.cpt_graph = gen_cpt_graph(self.sentence, self.cpt_vocab, isa_dict, causes_dict, hascnt_dict, sn)


        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]
        return self.sent_ids[conv], \
               self.masks[conv], \
               self.token_types[conv], \
               self.occu_ids[conv], \
               self.occu_masks[conv], \
               self.occu_token_types[conv], \
               self.styletopic_ids[conv], \
               self.styletopic_masks[conv], \
               self.styletopic_token_types[conv], \
               self.cpt_graph[conv], \
               torch.FloatTensor(self.speakers_[conv]), \
               torch.FloatTensor([1] * len(self.labels[conv])), \
               torch.LongTensor(self.labels[conv]), \
               torch.LongTensor(self.node_src[conv]), \
               torch.LongTensor(self.node_dst[conv]), \
               torch.LongTensor(self.edge_type[conv]), \
               torch.LongTensor(self.timenode_src[conv]), \
               torch.LongTensor(self.timenode_dst[conv]), \
               torch.LongTensor(self.timeedge_type[conv]), \
               conv

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        result = []
        for i in dat:

            if i < 9:
                result.append(dat[i][0])
            elif i < 10:
                result.append(dat[i][0])

            elif i < 13:
                result.append(pad_sequence([dat[i][0]], True))
            elif i < 19:
                result.append(dat[i][0])
            else:
                result.append(dat[i].tolist())
        return result

    def partition(self, data):

        return {key: data[key] for key in self.keys}



# deprecated
def tokenize_concept(tokenizer, rel='isa'):
    isa_dict_ids = {}
    src2ids = {}
    with open('./data/dialog_concept/{}_weight_dict_all.json'.format(rel), 'r',encoding='utf-8') as f:
        isa_dict = json.load(f)

    keys = isa_dict.keys()
    for key in keys:
        tokenized = tokenizer(key)['input_ids']  # [1:-1]
        isa_dict_ids[key] = []
        src2ids[key] = tokenized
        for v, w in isa_dict[key]:
            # print(v)
            # break
            isa_dict_ids[key].append((tokenizer(v)['input_ids'][1:-1], w))
    return isa_dict_ids, src2ids


# deprecated
def locate_concept(text, sent_ids, isa_dict_ids, src2ids):
    graph = {}
    keys = sent_ids.keys()
    for key in keys:
        dial = sent_ids[key]
        dial_text = text[key]
        srcs = []
        dsts = []
        weights = []

        for idx, (utt, utt_text) in enumerate(zip(dial, dial_text)):
            # utt_ = utt[:].tolist()
            words = utt_text.strip().split()
            src = []
            dst = []
            weight = []

            for word in words:
                if word in isa_dict_ids:
                    res = isa_dict_ids[word]

                    for e in res:
                        src_ids = src2ids[word]
                        pos = [utt.index(e) for e in src_ids]
                        src.append(pos)
                        # pos = [utt.index(e) for e in e[0]]
                        dst.append(e[0])
                        weight.append(e[1])
            srcs.append(src)
            dsts.append(dst)
            weights.append(weight)

        dial_graph = [srcs, dsts, weights]
        graph[key] = dial_graph
    return graph
