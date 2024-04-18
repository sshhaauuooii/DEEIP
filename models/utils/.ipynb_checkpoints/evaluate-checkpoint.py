
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
    batch=[]
    for i in range(len(p[0])):
        for q in p:
            batch.append(q[i])
        r_batch_p.append(batch)
        batch = []






    return r_batch_p, g

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


# def get_eval_results(doc_pred_spans, doc_gold_events, to_print=False):
#     ## get pred spans (tokenized and headnoun)
#     doc_pred_spans_head_noun = dict()
#     for index_id,doc_id in enumerate(doc_pred_spans):
#         doc_pred_spans_head_noun[index_id] = dict()
#         for tag in tags_to_extract:
#             doc_pred_spans_head_noun[index_id][tag] = list()
#             for idx, span in enumerate(doc_pred_spans[index_id][tag]):
#                 span_tokenized = tokenizer.tokenize(span)  # to normalize (remove other diff between pred and gold)
#                 doc_pred_spans[index_id][tag][idx] = span_tokenized
#
#                 head_noun = list()
#                 noun_chunks = list(nlp(" ".join(span_tokenized).replace(" ##", '')).noun_chunks)
#                 for noun_chunk in noun_chunks:
#                     head_noun.append(noun_chunk.root.text)
#
#                 doc_pred_spans_head_noun[index_id][tag].append({"span": span_tokenized, "hn": head_noun})
#
#             doc_pred_spans[index_id][tag] = remove_duplicate(doc_pred_spans[index_id][tag])
#             doc_pred_spans_head_noun[index_id][tag] = remove_duplicate_head_noun(doc_pred_spans_head_noun[index_id][tag])
#
#     ## get gold event (tokenized and headnoun)
#     doc_gold_events_head_noun = dict()
#     for index_id in doc_gold_events:
#         doc_gold_events_head_noun[index_id] = dict()
#         for tag in tags_to_extract:
#             doc_gold_events_head_noun[index_id][tag] = list()
#             for idx, event in enumerate(doc_gold_events[index_id]["roles"][tag]):
#                 event_tokenized = []
#                 event_head_noun = []
#
#                 for span in event:
#                     span_tokenized = tokenizer.tokenize(span)
#                     event_tokenized.append(span_tokenized)
#
#                     head_noun = list()
#                     noun_chunks = list(nlp(" ".join(span_tokenized).replace(" ##", '')).noun_chunks)
#                     for noun_chunk in noun_chunks:
#                         head_noun.append(noun_chunk.root.text)
#                     event_head_noun.append({"span": span_tokenized, "hn": head_noun})
#
#                 doc_gold_events[index_id]["roles"][tag][idx] = event_tokenized
#                 doc_gold_events_head_noun[index_id][tag].append(event_head_noun)
#
#     # ## get gold event (tokenized and headnoun)
#     # doc_gold_events_exact = dict()
#     # doc_gold_events_head_noun = dict()
#     # for doc_id in doc_gold_events:
#     #     doc_gold_events_exact[doc_id] = dict()
#     #     doc_gold_events_head_noun[doc_id] = dict()
#     #     for tag in tags_to_extract:
#     #         doc_gold_events_exact[doc_id][tag] = list()
#     #         doc_gold_events_head_noun[doc_id][tag] = list()
#     #         for event in doc_gold_events[doc_id]["roles"][tag]:
#     #             event_tokenized = []
#     #             event_head_noun = []
#
#     #             for span in event:
#     #                 span_tokenized = tokenizer.tokenize(span)
#     #                 event_tokenized.append(span_tokenized)
#
#     #                 head_noun = list()
#     #                 noun_chunks = list(nlp(" ".join(span_tokenized).replace(" ##", '')).noun_chunks)
#     #                 for noun_chunk in noun_chunks:
#     #                     head_noun.append(noun_chunk.root.text)
#     #                 event_head_noun.append({"span": span_tokenized, "hn": head_noun})
#
#     #             doc_gold_events_exact[doc_id][tag].append(event_tokenized)
#     #             doc_gold_events_head_noun[doc_id][tag].append(event_head_noun)
#
#     ##
#     ## report exact
#     ##
#     prec_marco, recall_marco = 0, 0
#     if to_print:
#         print("Exact Match\n", "precision, recall, F-1")
#     final_print = []
#     for tag in tags_to_extract:
#         gold_event_num, right_event_num, pred_span_num, right_span_num = 0, 0, 0, 0
#         for index_gold,doc_id in enumerate(doc_gold_events):
#             #if index_gold>1:continue
#             gold_events = doc_gold_events[doc_id]["roles"][tag]
#             #if doc_id not in doc_pred_spans: continue
#             pred_spans = doc_pred_spans[index_gold][tag]
#
#             # for recall
#             for event in gold_events:
#                 gold_event_num += 1
#                 if match_exact(pred_spans, event):
#                     right_event_num += 1
#             # for prec
#             for span in pred_spans:
#                 pred_span_num += 1
#                 all_events = list()
#                 for event in gold_events:
#                     all_events += event
#                 if match_exact([span], all_events):
#                     right_span_num += 1
#
#         recall, prec = -1, -1
#         if gold_event_num: recall = (right_event_num + 0.0) / gold_event_num
#         if pred_span_num: prec = (right_span_num + 0.0) / pred_span_num
#
#         if prec <= 0 or recall <= 0:
#             f_measure = -1
#         else:
#             prec *= 100
#             recall *= 100
#             f_measure = 2 * prec * recall / (prec + recall)
#
#         prec_marco += prec
#         recall_marco += recall
#
#         if to_print:
#             print("%s\n%.4f %.4f %.4f" % (tag2category[tag], prec, recall, f_measure))
#         final_print += [prec, recall, f_measure]
#
#     prec_marco = prec_marco / 5
#     recall_marco = recall_marco / 5
#     f_measure_marco = 2 * prec_marco * recall_marco / (prec_marco + recall_marco)
#     final_print += [prec_marco, recall_marco, f_measure_marco]
#     # print("\n\nmarco avg")
#     if to_print:
#         print("Macro:")
#         print("%.4f %.4f %.4f" % (prec_marco, recall_marco, f_measure_marco))
#         print("\nfinal_print")
#         for num in final_print: print(num, end=" ")
#
#     ##
#     ## report head noun
#     ##
#     doc_pred_spans, doc_gold_events = doc_pred_spans_head_noun, doc_gold_events_head_noun
#
#     prec_marco, recall_marco = 0, 0
#     if to_print:
#         print("\n\nHead Noun Match\n", "precision, recall, F-1")
#     final_print = []
#     for tag in tags_to_extract:
#         gold_event_num, right_event_num, pred_span_num, right_span_num = 0, 0, 0, 0
#         for index_gold,doc_id in enumerate(doc_gold_events):
#             #if index_gold > 1: continue
#             gold_events = doc_gold_events[doc_id][tag]
#             #if doc_id not in doc_pred_spans: continue
#             pred_spans = doc_pred_spans[index_gold][tag]
#
#             # for recall
#             for event in gold_events:
#                 gold_event_num += 1
#                 if match_noun(pred_spans, event):
#                     right_event_num += 1
#             # for prec
#             for span in pred_spans:
#                 pred_span_num += 1
#                 all_events = list()
#                 for event in gold_events:
#                     all_events += event
#                 if match_noun([span], all_events):
#                     right_span_num += 1
#
#         recall, prec = -1, -1
#         if gold_event_num: recall = (right_event_num + 0.0) / gold_event_num
#         if pred_span_num: prec = (right_span_num + 0.0) / pred_span_num
#
#         if prec <= 0 or recall <= 0:
#             f_measure = -1
#         else:
#             prec *= 100
#             recall *= 100
#             f_measure = 2 * prec * recall / (prec + recall)
#
#         prec_marco += prec
#         recall_marco += recall
#
#         if to_print:
#             print("%s\n%.4f %.4f %.4f" % (tag2category[tag], prec, recall, f_measure))
#         final_print += [prec, recall, f_measure]
#
#     prec_marco = prec_marco / 5
#     recall_marco = recall_marco / 5
#     f_measure_marco = 2 * prec_marco * recall_marco / (prec_marco + recall_marco)
#     final_print += [prec_marco, recall_marco, f_measure_marco]
#     if to_print:
#         print("Macro:")
#         print("%.4f %.4f %.4f" % (prec_marco, recall_marco, f_measure_marco))
#         print("\nfinal_print")
#         for num in final_print: print(num, end=" ")
#
#     return prec_marco, recall_marco, f_measure_marco

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

    # ## get gold event (tokenized and headnoun)
    # doc_gold_events_exact = dict()
    # doc_gold_events_head_noun = dict()
    # for doc_id in doc_gold_events:
    #     doc_gold_events_exact[doc_id] = dict()
    #     doc_gold_events_head_noun[doc_id] = dict()
    #     for tag in tags_to_extract:
    #         doc_gold_events_exact[doc_id][tag] = list()
    #         doc_gold_events_head_noun[doc_id][tag] = list()
    #         for event in doc_gold_events[doc_id]["roles"][tag]:
    #             event_tokenized = []
    #             event_head_noun = []

    #             for span in event:
    #                 span_tokenized = tokenizer.tokenize(span)
    #                 event_tokenized.append(span_tokenized)

    #                 head_noun = list()
    #                 noun_chunks = list(nlp(" ".join(span_tokenized).replace(" ##", '')).noun_chunks)
    #                 for noun_chunk in noun_chunks:
    #                     head_noun.append(noun_chunk.root.text)
    #                 event_head_noun.append({"span": span_tokenized, "hn": head_noun})

    #             doc_gold_events_exact[doc_id][tag].append(event_tokenized)
    #             doc_gold_events_head_noun[doc_id][tag].append(event_head_noun)

    ##
    ## report exact
    ##
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
