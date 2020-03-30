import os
import re
import ast
import yaml
import glob
import math
import random
import argparse
import datetime
import numpy as np
from tools import kflod, stopWords
from prettytable import PrettyTable
from concurrent.futures import ProcessPoolExecutor, Executor, as_completed


def task(prob, alpha):
    tot = 0
    TP = 0 # True Positive
    TN = 0 # True Negative
    FP = 0 # False Positive
    FN = 0 # False Negative

    fold_k = int(prob.split('_')[2].split('.')[0])
    with open(prob, 'r') as f:
        probs = yaml.load(f, Loader=yaml.FullLoader)
    with open('./DataSet/kflod.yaml', 'r') as f:
        flods = yaml.load(f, Loader=yaml.FullLoader)
        for item in flods:
            if item['flod'] == fold_k:
                label = item['label']
                ipath = item['path']
                with open(ipath.replace('..', './trec06p'), 'r', encoding='utf-8') as f_item:
                    try:
                        content = f_item.read()
                        # Replace http str
                        # regex = r'http[s]?://(?:[0-9]|[a-zA-Z]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                        # pattern = re.compile(regex)
                        # content = re.sub(pattern, 'http', content)
                        # Replace html str
                        regex = r'</?\w+[^>]*>'
                        pattern = re.compile(regex)
                        content = re.sub(pattern, ' html ', content)
                        # Meta data: received from
                        regex = r'Received:(?:\s)+from\s[a-zA-Z0-9\.\-]*?\s'
                        pattern = re.compile(regex)
                        email_from = re.findall(pattern, content)
                        mails = []
                        for email in email_from:
                            mails += email.split('from')[-1].strip(' ').strip('\n').strip('\t').split('.')
                        # Meta data: Mailer
                        regex = r'Mailer:\s(.*?)\s'
                        pattern = re.compile(regex)
                        mailer = re.findall(pattern, content)
                        # Skip meta message
                        # regex = r'Received(?:.|\n)*?\n\n'
                        # pattern = re.compile(regex)
                        # content = re.sub(pattern, '', content)
                        # Replace long nonsense str, e.g. media encoding bytes
                        regex = r'\S{20,}'
                        pattern = re.compile(regex)
                        content = re.sub(pattern, 'nonsense', content)
                        # Match words
                        regex = r'[a-zA-Z]+'
                        pattern = re.compile(regex)
                        content = re.findall(pattern, content)
                    except UnicodeDecodeError:
                        # print('Skip file ' + ipath.replace('..', './trec06p') + ' because it cannot be encoded by utf-8')
                        continue
                    else:
                        prob_predict = {}
                        tot_label = 0
                        tot_dim = 0
                        for value in probs['label_classes'].values():
                            tot_label += value
                        for key in probs['label_classes'].keys():
                            tot_dim += probs['features'][key + '_dim']
                        for key, value in probs['label_classes'].items():
                            prob_predict[key] = math.log(probs['label_classes'][key] / (tot_label + len(probs['label_classes']) * alpha))
                        
                        for word in content:
                            for key in probs['label_classes'].keys():
                                # Skip Stop words:
                                if probs['features'][key].get(word, -1) != 0:
                                    # Increase weight of html
                                    w_html = 1
                                    if(word == 'html'):
                                        w_html = 10
                                    prob_predict[key] += (w_html * math.log(probs['features'][key].get(word, 0) + alpha) - \
                                        math.log(probs['label_classes'][key] + tot_dim * alpha))

                        for word in mails:
                            for key in probs['label_classes'].keys():
                                # Skip Stop words:
                                if probs['features'][key].get(word, -1) != 0:
                                    # Increase weight of mails
                                    w_mails = 2
                                    prob_predict[key] += (w_mails * math.log(probs['features'][key].get(word, 0) + alpha) - \
                                        math.log(probs['label_classes'][key] + tot_dim * alpha))

                        for word in mailer:
                            for key in probs['label_classes'].keys():
                                # Skip Stop words:
                                if probs['features'][key].get(word, -1) != 0:
                                    # Increase weight of mailer
                                    w_mailer = 200
                                    prob_predict[key] += (w_mailer * math.log(probs['features'][key].get(word, 0) + alpha) - \
                                        math.log(probs['label_classes'][key] + tot_dim * alpha))
                        
                        label_predict = ''
                        max_prediction = -math.inf
                        for key, value in prob_predict.items():
                            if value > max_prediction:
                                max_prediction = value
                                label_predict = key
                        if(label == label_predict and label_predict == 'spam'):
                            TP += 1
                        elif(label == label_predict and label_predict == 'ham'):
                            TN += 1
                        elif(label != label_predict and label_predict == 'spam'):
                            FP += 1
                        elif(label != label_predict and label_predict == 'ham'):
                            FN +=1
                        tot += 1
        
    accuracy = (TP + TN) / tot
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print('Testing on flod {} Accuracy: {} Precision: {} Recall: {:.16f} F1: {}'.format(fold_k, accuracy, precision, recall, f1))
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}      


def predict(alpha, record):
    _probs = glob.glob('./Prob/*.yaml')
    acc = []
    pre = []
    rec = []
    F1 = []

    with ProcessPoolExecutor() as executor:
        for results in executor.map(task, _probs, [alpha]*len(_probs)):
            acc.append(results['accuracy'])
            pre.append(results['precision'])
            rec.append(results['recall'])
            F1.append(results['f1'])

    # Print mean, min and max of acc, pre, rec and F1
    table = PrettyTable(['Index', 'Accuracy', 'Precision', 'Recall', 'F1'])
    table.padding_width = 1
    table.add_row(['Mean'] + [np.mean(x) for x in [acc, pre, rec, F1]])
    table.add_row(['Min'] + [np.min(x) for x in [acc, pre, rec, F1]])
    table.add_row(['Max'] + [np.max(x) for x in [acc, pre, rec, F1]])
    print(table)
    
    # Record exp data
    if record:
        if not os.path.exists('./Record'):
            os.mkdir('./Record')
        records = glob.glob('./Record/*.yaml')
        with open('./Record/record_ver_' + str(len(records)) + '.yaml', 'w') as f:
            yaml.dump({'acc': acc, 'pre': pre, 'rec': rec, 'F1': F1}, f, default_flow_style=False)      


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", default=1e-30, type=float, help="alpha value for zero-probabilities smooth")
    parser.add_argument("--record", default=True, type=ast.literal_eval, choices=[True, False], help="Record prediction result")
    parser.add_argument("--seed", default='2017013632', type=str, help="Seed for random")
    args = parser.parse_args()

    random.seed(args.seed)
    print('===> Predicting')
    start = datetime.datetime.now()
    predict(alpha=args.alpha, record=args.record)
    end = datetime.datetime.now()
    print('Time cost: {}'.format(end -start))
    print('===> Completed!')
    print('-' * 20)