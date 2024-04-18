
import numpy as np
import sys
import json
from transformers import BertTokenizer
import spacy
tags_to_extract =['perp_individual_id', 'perp_organization_id', 'phys_tgt_id', 'hum_tgt_name', 'incident_instrument_id']
tag2category = {'perp_individual_id': "PerpInd", 'perp_organization_id':"PerpOrg", 'phys_tgt_id':"Target", 'hum_tgt_name':"Victim", 'incident_instrument_id':"Weapon"}
## input as sentence level labels
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='../utils/bert') # Load bert pre-trained models tokenizer (vocabulary)



def recover_label(pred_variable, gold_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    dic={
        0:"O",1:"B",2:"I"
    }

    p=[       [ [[    dic[word]          for word in sentent]for sentent in b ]for b in q]            for q in pred_variable]
    g = [[[[dic[int(word)] for word in sentent] for sentent in b] for b in q] for q in gold_variable]


    r_batch_p=[]
    r_batch_g = []
    batch=[]
    for i in range(len(p[0])):
        for q in p:
            batch.append(q[i])
        r_batch_p.append(batch)
        batch = []

    for i in range(len(g[0])):
        for q in g:
            batch.append(q[i])
        r_batch_g.append(batch)
        batch = []


    return r_batch_p, r_batch_g

def remove_duplicate(items):
    items_no_dup = list()
    for item in items:
        if item not in items_no_dup:
            items_no_dup.append(item)
    return items_no_dup

def remove_duplicate_head_noun(items):
    items_no_dup = list()
    items_no_dup_hn = list()
    for item in items:
        if item["hn"] not in items_no_dup_hn or not item["hn"]:
            items_no_dup.append(item)
            items_no_dup_hn.append(item["hn"])
    return items_no_dup
def get_macro_avg(sequences, pred_results, doc_ids, to_print=False):
    # read pred spans
    doc_pred_spans =  dict()

    for seq, pred_result, doc_id in zip(sequences, pred_results, doc_ids):
        if doc_id not in doc_pred_spans:
            doc_pred_spans[doc_id] = dict()
            for tag in tags_to_extract:
                doc_pred_spans[doc_id][tag] = list()
        for index_q,qs in enumerate(pred_result):
            words=[]
            one_enti=[]
            for tag_sent,pred_sentence in zip(seq,qs):
                for index,word in enumerate(pred_sentence):
                    if word=="B":
                        if one_enti:
                            words.append("".join(one_enti).replace("##",''))
                            one_enti=[]
                        else:one_enti+=tag_sent[index]
                    if word=="I":
                        if '##'in tag_sent[index]:
                            one_enti += tag_sent[index]
                        else:one_enti+=' '+tag_sent[index]
                    if word=="O":
                        if one_enti:
                            words.append("".join(one_enti).replace("##",''))
                            one_enti=[]
                if one_enti:
                    words.append("".join(one_enti).replace("##",''))
                    one_enti=[]
            doc_pred_spans[doc_id][tags_to_extract[index_q]].extend(words)








    # read gold events
    with open("../../Muc34/test.json") as f_gold:
        doc_gold_events = json.loads(f_gold.read())

    # get_eval_results
    p, r, f = get_eval_results(doc_pred_spans, doc_gold_events,to_print=to_print)
    return p, r, f



def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string



def get_eval_results(doc_pred_spans, doc_gold_events, to_print=False):
    ## get pred spans (tokenized and headnoun)
    doc_pred_spans_head_noun = dict()
    for doc_id in doc_pred_spans:
        doc_pred_spans_head_noun[doc_id] = dict()
        for tag in tags_to_extract:
            doc_pred_spans_head_noun[doc_id][tag] = list()
            for idx, span in enumerate(doc_pred_spans[doc_id][tag]):
                span_tokenized = tokenizer.tokenize(span)  # to normalize (remove other diff between pred and gold)
                doc_pred_spans[doc_id][tag][idx] = span_tokenized

                head_noun = list()
                noun_chunks = list(nlp(" ".join(span_tokenized).replace(" ##", '')).noun_chunks)
                for noun_chunk in noun_chunks:
                    head_noun.append(noun_chunk.root.text)

                doc_pred_spans_head_noun[doc_id][tag].append({"span": span_tokenized, "hn": head_noun})

            doc_pred_spans[doc_id][tag] = remove_duplicate(doc_pred_spans[doc_id][tag])
            doc_pred_spans_head_noun[doc_id][tag] = remove_duplicate_head_noun(doc_pred_spans_head_noun[doc_id][tag])

    ## get gold event (tokenized and headnoun)
    doc_gold_events_head_noun = dict()
    for doc_id in doc_gold_events:
        doc_gold_events_head_noun[doc_id] = dict()
        for tag in tags_to_extract:
            doc_gold_events_head_noun[doc_id][tag] = list()
            for idx, event in enumerate(doc_gold_events[doc_id]["roles"][tag]):
                event_tokenized = []
                event_head_noun = []

                for span in event:
                    span_tokenized = tokenizer.tokenize(span)
                    event_tokenized.append(span_tokenized)

                    head_noun = list()
                    noun_chunks = list(nlp(" ".join(span_tokenized).replace(" ##", '')).noun_chunks)
                    for noun_chunk in noun_chunks:
                        head_noun.append(noun_chunk.root.text)
                    event_head_noun.append({"span": span_tokenized, "hn": head_noun})

                doc_gold_events[doc_id]["roles"][tag][idx] = event_tokenized
                doc_gold_events_head_noun[doc_id][tag].append(event_head_noun)

    prec_marco, recall_marco = 0, 0
    if to_print:
        print("Exact Match\n", "precision, recall, F-1")
    final_print = []
    for tag in tags_to_extract:
        gold_event_num, right_event_num, pred_span_num, right_span_num = 0, 0, 0, 0
        for doc_id in doc_gold_events:
            gold_events = doc_gold_events[doc_id]["roles"][tag]
            if doc_id not in doc_pred_spans: continue
            pred_spans = doc_pred_spans[doc_id][tag]

            # for recall
            for event in gold_events:
                gold_event_num += 1
                if match_exact(pred_spans, event):
                    right_event_num += 1
            # for prec
            for span in pred_spans:
                pred_span_num += 1
                all_events = list()
                for event in gold_events:
                    all_events += event
                if match_exact([span], all_events):
                    right_span_num += 1

        recall, prec = -1, -1
        if gold_event_num: recall = (right_event_num + 0.0) / gold_event_num
        if pred_span_num: prec = (right_span_num + 0.0) / pred_span_num

        if prec <= 0 or recall <= 0:
            f_measure = -1
        else:
            prec *= 100
            recall *= 100
            f_measure = 2 * prec * recall / (prec + recall)

        prec_marco += prec
        recall_marco += recall

        if to_print:
            print("%s\n%.4f %.4f %.4f" % (tag2category[tag], prec, recall, f_measure))
        final_print += [prec, recall, f_measure]

    prec_marco = prec_marco / 5
    recall_marco = recall_marco / 5
    f_measure_marco = 2 * prec_marco * recall_marco / (prec_marco + recall_marco)
    final_print += [prec_marco, recall_marco, f_measure_marco]
    # print("\n\nmarco avg")
    if to_print:
        print("Macro:")
        print("%.4f %.4f %.4f" % (prec_marco, recall_marco, f_measure_marco))
        print("\nfinal_print")
        for num in final_print: print(num, end=" ")

    ##
    ## report head noun
    ##
    doc_pred_spans, doc_gold_events = doc_pred_spans_head_noun, doc_gold_events_head_noun

    prec_marco, recall_marco = 0, 0
    if to_print:
        print("\n\nHead Noun Match\n", "precision, recall, F-1")
    final_print = []
    for tag in tags_to_extract:
        gold_event_num, right_event_num, pred_span_num, right_span_num = 0, 0, 0, 0
        for doc_id in doc_gold_events:
            gold_events = doc_gold_events[doc_id][tag]
            if doc_id not in doc_pred_spans: continue
            pred_spans = doc_pred_spans[doc_id][tag]

            # for recall
            for event in gold_events:
                gold_event_num += 1
                if match_noun(pred_spans, event):
                    right_event_num += 1
            # for prec
            for span in pred_spans:
                pred_span_num += 1
                all_events = list()
                for event in gold_events:
                    all_events += event
                if match_noun([span], all_events):
                    right_span_num += 1

        recall, prec = -1, -1
        if gold_event_num: recall = (right_event_num + 0.0) / gold_event_num
        if pred_span_num: prec = (right_span_num + 0.0) / pred_span_num

        if prec <= 0 or recall <= 0:
            f_measure = -1
        else:
            prec *= 100
            recall *= 100
            f_measure = 2 * prec * recall / (prec + recall)

        prec_marco += prec
        recall_marco += recall

        if to_print:
            print("%s\n%.4f %.4f %.4f" % (tag2category[tag], prec, recall, f_measure))
        final_print += [prec, recall, f_measure]

    prec_marco = prec_marco / 5
    recall_marco = recall_marco / 5
    f_measure_marco = 2 * prec_marco * recall_marco / (prec_marco + recall_marco)
    final_print += [prec_marco, recall_marco, f_measure_marco]
    if to_print:
        print("Macro:")
        print("%.4f %.4f %.4f" % (prec_marco, recall_marco, f_measure_marco))
        print("\nfinal_print")
        for num in final_print: print(num, end=" ")

    return prec_marco, recall_marco, f_measure_marco


def match_exact(preds, golds):
    for pred_span in preds:
        for gold_span in golds:
            if pred_span == gold_span:
                return True
    return False

def match_noun(preds, golds):
    for pred in preds:
        for gold in golds:
            # must have this line (no head noun in 'fmln')
            if pred['span'] == gold['span']:
                return True
            if pred['hn'] and gold['hn']:
                for n1 in pred['hn']:
                    for n2 in gold['hn']:
                        if n1 == n2:
                            return True
    return False



def measure_event_table_filling(
    pred_record_mat_list, gold_record_mat_list,
    event_type_roles_list,
    pred_event_types, gold_event_types,
    pred_spans_token_tuple_list, gold_spans_token_tuple_list,
    avg_type='micro',
    dict_return=False
):

    event_mcml_prf1 = get_mcml_prf1(pred_event_types, gold_event_types, event_type_roles_list)
    ent_prf1 = get_ent_prf1(pred_spans_token_tuple_list, gold_spans_token_tuple_list)

    event_role_num_list = [len(roles) for _, roles in event_type_roles_list]
    # to store total statistics of TP, FP, FN
    total_event_role_stats = [
        [
            [0]*3 for _ in range(role_num)
        ] for event_idx, role_num in enumerate(event_role_num_list)
    ]

    assert len(pred_record_mat_list) == len(gold_record_mat_list)
    for pred_record_mat, gold_record_mat in zip(pred_record_mat_list, gold_record_mat_list):
        event_role_tpfpfn_stats = agg_ins_event_role_tpfpfn_stats(
            pred_record_mat, gold_record_mat, event_role_num_list
        )
        for event_idx, role_num in enumerate(event_role_num_list):
            for role_idx in range(role_num):
                for sid in range(3):
                    total_event_role_stats[event_idx][role_idx][sid] += \
                        event_role_tpfpfn_stats[event_idx][role_idx][sid]

    per_role_metric = []
    per_event_metric = []

    num_events = len(event_role_num_list)
    g_tpfpfn_stat = [0] * 3
    g_prf1_stat = [0] * 3
    event_role_eval_dicts = []
    for event_idx, role_num in enumerate(event_role_num_list):
        event_tpfpfn = [0] * 3  # tp, fp, fn
        event_prf1_stat = [0] * 3
        per_role_metric.append([])
        role_eval_dicts = []
        for role_idx in range(role_num):
            role_tpfpfn_stat = total_event_role_stats[event_idx][role_idx][:3]
            role_prf1_stat = get_prec_recall_f1(*role_tpfpfn_stat)
            per_role_metric[event_idx].append(role_prf1_stat)
            for mid in range(3):
                event_tpfpfn[mid] += role_tpfpfn_stat[mid]
                event_prf1_stat[mid] += role_prf1_stat[mid]

            role_eval_dict = {
                'RoleType': event_type_roles_list[event_idx][1][role_idx],
                'Precision': role_prf1_stat[0],
                'Recall': role_prf1_stat[1],
                'F1': role_prf1_stat[2],
                'TP': role_tpfpfn_stat[0],
                'FP': role_tpfpfn_stat[1],
                'FN': role_tpfpfn_stat[2]
            }
            role_eval_dicts.append(role_eval_dict)

        for mid in range(3):
            event_prf1_stat[mid] /= role_num
            g_tpfpfn_stat[mid] += event_tpfpfn[mid]
            g_prf1_stat[mid] += event_prf1_stat[mid]

        micro_event_prf1 = get_prec_recall_f1(*event_tpfpfn)
        macro_event_prf1 = tuple(event_prf1_stat)
        if avg_type.lower() == 'micro':
            event_prf1_stat = micro_event_prf1
        elif avg_type.lower() == 'macro':
            event_prf1_stat = macro_event_prf1
        else:
            raise Exception('Unsupported average type {}'.format(avg_type))

        per_event_metric.append(event_prf1_stat)

        event_eval_dict = {
            'EventType': event_type_roles_list[event_idx][0],
            'MacroPrecision': macro_event_prf1[0],
            'MacroRecall': macro_event_prf1[1],
            'MacroF1': macro_event_prf1[2],
            'MicroPrecision': micro_event_prf1[0],
            'MicroRecall': micro_event_prf1[1],
            'MicroF1': micro_event_prf1[2],
            'TP': event_tpfpfn[0],
            'FP': event_tpfpfn[1],
            'FN': event_tpfpfn[2],
        }
        event_role_eval_dicts.append((event_eval_dict, role_eval_dicts))

    micro_g_prf1 = get_prec_recall_f1(*g_tpfpfn_stat)
    macro_g_prf1 = tuple(s / num_events for s in g_prf1_stat)
    if avg_type.lower() == 'micro':
        g_metric = micro_g_prf1
    else:
        g_metric = macro_g_prf1

    g_eval_dict = {
        'MacroPrecision': macro_g_prf1[0],
        'MacroRecall': macro_g_prf1[1],
        'MacroF1': macro_g_prf1[2],
        'MicroPrecision': micro_g_prf1[0],
        'MicroRecall': micro_g_prf1[1],
        'MicroF1': micro_g_prf1[2],
        'TP': g_tpfpfn_stat[0],
        'FP': g_tpfpfn_stat[1],
        'FN': g_tpfpfn_stat[2],
        'classification': event_mcml_prf1,
        'entity': ent_prf1,
    }
    event_role_eval_dicts.append(g_eval_dict)

    if not dict_return:
        return g_metric, per_event_metric, per_role_metric
    else:
        return event_role_eval_dicts

def get_mcml_prf1(pred_event_types, gold_event_types, event_type_roles_list):
    """get p r f1 measures of classification results"""
    len_events = len(event_type_roles_list)
    event_tp_fp_fn = [[0] * 3 for _ in range(len_events)]
    event_p_r_f1 = [[0.0] * 3 for _ in range(len_events)]
    tot_tp_fp_fn = [0] * 3
    tot_p_r_f1 = [0.0] * 3
    for preds, golds in zip(pred_event_types, gold_event_types):
        for event_idx, (pred, gold) in enumerate(zip(preds, golds)):
            if pred == 0:
                if gold == 0:  # TN
                    pass
                else:  # FN
                    event_tp_fp_fn[event_idx][2] += 1
            else:
                if gold == 0:  # FP
                    event_tp_fp_fn[event_idx][1] += 1
                else:  # TP: if both pred and gold contains paths for this event, then it's TP
                    event_tp_fp_fn[event_idx][0] += 1
    for event_idx, tp_fp_fn in enumerate(event_tp_fp_fn):
        tot_tp_fp_fn[0] += tp_fp_fn[0]
        tot_tp_fp_fn[1] += tp_fp_fn[1]
        tot_tp_fp_fn[2] += tp_fp_fn[2]
        prec, rec, f1 = get_prec_recall_f1(*tp_fp_fn)
        event_p_r_f1[event_idx][0] = prec
        event_p_r_f1[event_idx][1] = rec
        event_p_r_f1[event_idx][2] = f1

    micro_p, micro_r, micro_f1 = get_prec_recall_f1(*tot_tp_fp_fn)
    tot_p_r_f1[0] = micro_p
    tot_p_r_f1[1] = micro_r
    tot_p_r_f1[2] = micro_f1
    macro_p = sum([x[0] for x in event_p_r_f1]) / len_events
    macro_r = sum([x[1] for x in event_p_r_f1]) / len_events
    macro_f1 = sum([x[2] for x in event_p_r_f1]) / len_events

    results = {
        "MacroPrecision": macro_p,
        "MacroRecall": macro_r,
        "MacroF1": macro_f1,
        "MicroPrecision": micro_p,
        "MicroRecall": micro_r,
        "MicroF1": micro_f1,
        "TP": tot_tp_fp_fn[0],
        "FP": tot_tp_fp_fn[1],
        "FN": tot_tp_fp_fn[2],
        "Events": [{
            "EventType": event_type_roles_list[event_idx][0],
            "Precision": event_p_r_f1[event_idx][0],
            "Recall": event_p_r_f1[event_idx][1],
            "F1": event_p_r_f1[event_idx][2],
            "TP": event_tp_fp_fn[event_idx][0],
            "FP": event_tp_fp_fn[event_idx][1],
            "FN": event_tp_fp_fn[event_idx][2],
        } for event_idx in range(len_events)]
    }
    return results


def get_prec_recall_f1(tp, fp, fn):
    a = tp + fp
    prec = tp / a if a > 0 else 0
    b = tp + fn
    rec = tp / b if b > 0 else 0
    if prec > 0 and rec > 0:
        f1 = 2.0 / (1 / prec + 1 / rec)
    else:
        f1 = 0
    return prec, rec, f1


def get_ent_prf1(pred_spans_token_tuple_list, gold_spans_token_tuple_list):
    """get p r f1 measures of entity prediction results"""
    tot_tp_fp_fn = [0] * 3
    tot_p_r_f1 = [0.0] * 3

    for preds, golds in zip(pred_spans_token_tuple_list, gold_spans_token_tuple_list):
        pred_event_ents=gold_event_ents=set()
        for i_ in preds:
            pred_event_ents|=set(i_)
        for i_ in golds:
            gold_event_ents|=set(i_)
        # preds=[ set(i_) for i_ in preds ]
        # golds=[ set(i_) for i_ in golds ]# doc
        # pred_event_ents = set(preds)
        # gold_event_ents = set(golds)
        tot_tp_fp_fn[0] += len(pred_event_ents & gold_event_ents)  # TP
        tot_tp_fp_fn[1] += len(pred_event_ents - gold_event_ents)  # FP
        tot_tp_fp_fn[2] += len(gold_event_ents - pred_event_ents)  # FN

    micro_p, micro_r, micro_f1 = get_prec_recall_f1(*tot_tp_fp_fn)
    tot_p_r_f1[0] = micro_p
    tot_p_r_f1[1] = micro_r
    tot_p_r_f1[2] = micro_f1

    results = {
        "MicroPrecision": micro_p,
        "MicroRecall": micro_r,
        "MicroF1": micro_f1,
        "TP": tot_tp_fp_fn[0],
        "FP": tot_tp_fp_fn[1],
        "FN": tot_tp_fp_fn[2]
    }
    return results


def agg_ins_event_role_tpfpfn_stats(pred_record_mat, gold_record_mat, event_role_num_list):
    """
    Aggregate TP,FP,FN statistics for a single instance.
    A record_mat should be formated as
    [(Event Index)
        [(Record Index)
            ((Role Index)
                argument 1, ...
            ), ...
        ], ...
    ], where argument 1 should support the '=' operation and the empty argument is None.
    """
    assert len(pred_record_mat) == len(gold_record_mat)
    # tpfpfn_stat: TP, FP, FN
    event_role_tpfpfn_stats = []
    for event_idx, (pred_records, gold_records) in enumerate(zip(pred_record_mat, gold_record_mat)):
        role_num = event_role_num_list[event_idx]
        role_tpfpfn_stats = agg_event_role_tpfpfn_stats(pred_records, gold_records, role_num)
        event_role_tpfpfn_stats.append(role_tpfpfn_stats)

    return event_role_tpfpfn_stats


def agg_event_role_tpfpfn_stats(pred_records, gold_records, role_num):
    """
    Aggregate TP,FP,FN statistics for a single event prediction of one instance.
    A pred_records should be formated as
    [(Record Index)
        ((Role Index)
            argument 1, ...
        ), ...
    ], where argument 1 should support the '=' operation and the empty argument is None.
    """
    role_tpfpfn_stats = [[0] * 3 for _ in range(role_num)]

    if gold_records is None:
        if pred_records is not None:  # FP
            for pred_record in pred_records:
                assert len(pred_record) == role_num
                for role_idx, arg_tup in enumerate(pred_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][1] += 1
        else:  # ignore TN
            pass
    else:
        if pred_records is None:  # FN
            for gold_record in gold_records:
                assert len(gold_record) == role_num
                for role_idx, arg_tup in enumerate(gold_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][2] += 1
        else:  # True Positive at the event level
            # sort predicted event records by the non-empty count
            # to remove the impact of the record order on evaluation
            pred_records = sorted(pred_records,
                                  key=lambda x: sum(1 for a in x if a is not None),
                                  reverse=True)
            gold_records = list(gold_records)

            while len(pred_records) > 0 and len(gold_records) > 0:
                pred_record = pred_records[0]
                assert len(pred_record) == role_num

                # pick the most similar gold record
                _tmp_key = lambda gr: sum([1 for pa, ga in zip(pred_record, gr) if pa == ga])
                best_gr_idx = gold_records.index(max(gold_records, key=_tmp_key))
                gold_record = gold_records[best_gr_idx]

                for role_idx, (pred_arg, gold_arg) in enumerate(zip(pred_record, gold_record)):
                    if gold_arg is None:
                        if pred_arg is not None:  # FP at the role level
                            role_tpfpfn_stats[role_idx][1] += 1
                        else:  # ignore TN
                            pass
                    else:
                        if pred_arg is None:  # FN
                            role_tpfpfn_stats[role_idx][2] += 1
                        else:
                            if pred_arg == gold_arg:  # TP
                                role_tpfpfn_stats[role_idx][0] += 1
                            else:
                                role_tpfpfn_stats[role_idx][1] += 1
                                role_tpfpfn_stats[role_idx][2] += 1

                del pred_records[0]
                del gold_records[best_gr_idx]

            # remaining FP
            for pred_record in pred_records:
                assert len(pred_record) == role_num
                for role_idx, arg_tup in enumerate(pred_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][1] += 1
            # remaining FN
            for gold_record in gold_records:
                assert len(gold_record) == role_num
                for role_idx, arg_tup in enumerate(gold_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][2] += 1

    return role_tpfpfn_stats









####################CFEED#############################


def CFEED_evaluation(event_type,sentence,pred_results,gold_results):

    doc_pred_spans = []
    doc_pred_spans_qs = []
    for seq,seq_role in zip(sentence,pred_results):


        for index_q,qs in enumerate(seq_role):
            words=[]
            one_enti=''
            for tag_sent,pred_sentence in zip(seq,qs):
                for index,word in enumerate(pred_sentence):
                    if word=="B":
                        if one_enti:
                            words.append(one_enti)
                            one_enti=''
                        else:one_enti+=tag_sent[index]
                    if word=="I":
                        one_enti+=tag_sent[index]
                    if word=="O":
                        if one_enti:
                            words.append(one_enti)
                            one_enti=''
                if one_enti:
                    words.append(one_enti)
                    one_enti=''
            doc_pred_spans_qs.append(list(set(words)))

        doc_pred_spans.append(doc_pred_spans_qs)
        doc_pred_spans_qs=[]


    doc_gold_spans = []
    doc_gold_spans_qs = []
    for seq,seq_role in zip(sentence,gold_results):
        for index_q,qs in enumerate(seq_role):
            words=[]
            one_enti=''
            for tag_sent,gold_sentence in zip(seq,qs):
                for index,word in enumerate(gold_sentence):
                    if word=="B":
                        if one_enti:
                            words.append(one_enti)
                            one_enti=''
                        else:
                            if index>len(tag_sent):
                                print()
                            one_enti+=tag_sent[index]
                    if word=="I":
                        one_enti+=tag_sent[index]
                    if word=="O":
                        if one_enti:
                            words.append(one_enti)
                            one_enti=''
                if one_enti:
                    words.append(one_enti)
                    one_enti=''
            doc_gold_spans_qs.append(list(set(words)))

        doc_gold_spans.append(doc_gold_spans_qs)
        doc_gold_spans_qs=[]

    """
    correct_predict_num是TP
    find_num是TP+FP
    total_num是TP+FN
    """
    total_num=0
    find_num=0
    correct_golden_num=0
    correct_predict_num=0
    for gold,pred in zip(doc_gold_spans,doc_pred_spans):

        total_, find_, correct_golden_, correct_predict_=compare(gold, pred, event_type)
        total_num += total_
        find_num += find_
        correct_golden_num += correct_golden_
        correct_predict_num += correct_predict_

    if correct_golden_num ==0:
        return -1,-1,-1
    r = correct_golden_num / total_num
    p = correct_predict_num / find_num
    f = (p * r) / (p + r) * 2
    return p,r,f


def compare(golden_events, predict_events, event_type):
    total_num = 0
    find_num = 0#预测实体数量
    correct_golden_num = 0
    correct_predict_num = 0


    for g_role,p_role in zip(golden_events,predict_events):
        for g_tag in g_role:
            total_num += 1
            # for p_tag in p_role:
            #     if g_tag in p_tag:
            #         correct_golden_num += 1

    for g_role,p_role in zip(golden_events,predict_events):
        for p_tag in p_role:
            find_num += 1
            if p_tag in g_role:
                correct_predict_num += 1
    return total_num, find_num, correct_predict_num, correct_predict_num


def statisatic_pro(golden_events, golden_dict):
    golden_events_list = []
    for golden_event in golden_events:
        for arguments in golden_event:
            if arguments[0] in golden_dict.keys():
                if arguments[1] not in golden_events_list:
                    golden_dict[arguments[0]].append(arguments[1])
                    golden_events_list.append(arguments[1])
            else:
                golden_dict[arguments[0]] = [arguments[1]]
    return golden_dict



