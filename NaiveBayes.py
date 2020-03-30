import os
import re
import yaml
import random
import datetime
import argparse
from tools import kflod, stopWords
from concurrent.futures import ProcessPoolExecutor, Executor


def task(i, K, sample, rank, fuzzy):
    pro = 'Training on flod '
    for j in range(K):
        if j != i:
            pro += (str(j) + ' ')
    print(pro + '...', end='')

    label_classes = {'ham':0, 'spam':0}
    features = {'ham':{}, 'spam':{}}
    with open('./DataSet/kflod.yaml', 'r') as f:
        flods = yaml.load(f, Loader=yaml.FullLoader)
        for item in flods:
            if item['flod'] != i and random.random() <= sample:
                label = item['label']
                ipath = item['path']
                with open(ipath.replace('..', './trec06p'), 'r', encoding='utf-8') as f_item:
                    try:
                        content = f_item.read()
                        # Replace http str
                        # regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                        # pattern = re.compile(regex)
                        # content = re.sub(pattern, ' http ', content)
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
                        content = re.sub(pattern, '', content)
                        # Match words
                        regex = r'[a-zA-Z]+'
                        pattern = re.compile(regex)
                        content = re.findall(pattern, content)           
                    except UnicodeDecodeError:
                        # print('Skip file ' + ipath.replace('..', './trec06p') + ' because it cannot be encoded by utf-8')
                        continue
                    else:
                        content += (mails + mailer)
                        for word in content:
                            if not word in features[label].keys():
                                features[label][word] = 1
                            else:
                                features[label][word] += 1
                        label_classes[label] += 1 


    # Select top rank% features for valid features
    for _key, value in features.items():
        features[_key] = dict(sorted(value.items(), key = lambda x: x[1], reverse = True)[:int(len(value)*rank)])
    
    # Add attribution of feature dimension
    features_dict = {}
    for key, value in features.items():
        features_dict[key + '_dim'] = len(value)
    features.update(features_dict)

    # Delete Fuzzy features
    for key in list(features['ham'].keys()):
        key_rate = features['spam'].get(key, 0) / features['ham'][key]
        classes_rate = label_classes['spam'] / label_classes['ham']
        if((key_rate > (1-fuzzy)*classes_rate) and (key_rate < (1+fuzzy)*classes_rate)):
            del features['ham'][key]
            del features['spam'][key]

    for key in list(features['spam'].keys()):
        key_rate = features['ham'].get(key, 0) / features['spam'][key]
        classes_rate = label_classes['ham'] / label_classes['spam']
        if((key_rate > (1-fuzzy)*classes_rate) and (key_rate < (1+fuzzy)*classes_rate)):
            del features['ham'][key]
            del features['spam'][key]

    # Delete stop words
    stop_words = stopWords()
    for key in label_classes.keys():
        for words in stop_words:
            features[key][words] = 0

    # Debug: print current most used words
    # for _key, value in features.items():
    #     print(sorted(value.items(), key = lambda x: x[1], reverse = True)[:20])

    prob = {'label_classes': label_classes, 'features': features}     
    if not os.path.exists('./Prob'):
        os.mkdir('./Prob')
    with open('./Prob/prob_fold_' + str(i) + '.yaml', 'w') as f:
        yaml.dump(prob, f, default_flow_style=False)
    print(' Completed!')


def naiveBayes(K, sample, rank, fuzzy):
    if not os.path.exists('./DataSet/kflod.yaml'):
        kflod(K=K, fpath='./trec06p/label/index')

    with ProcessPoolExecutor() as executor:
        executor.map(task, range(K), [K]*K, [sample]*K, [rank]*K, [fuzzy]*K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=5, type=int, help="Number of fold")
    parser.add_argument("--sample", default=1.0, type=float, help="Sample specified rate of data for training")
    parser.add_argument("--rank", default=0.2, type=float, help="Select rank percent of words data for predicting")
    parser.add_argument("--fuzzy", default=0.25, type=float, help="fuzzy rate of features to be ignored")
    parser.add_argument("--seed", default='2017013632', type=str, help="Seed for random")
    args = parser.parse_args()
    assert args.sample > 0 and args.sample <= 1.0 , 'sample should between 0 and 1.0'
    assert args.rank > 0 and args.rank <= 1.0 , 'rank should between 0 and 1.0'
    assert args.fuzzy >= 0 and args.fuzzy < 1.0 , 'fuzzy should between 0 and 1.0'

    random.seed(args.seed)
    print('===> Training')
    start = datetime.datetime.now()
    naiveBayes(K=args.k, sample=args.sample, rank=args.rank, fuzzy=args.fuzzy)
    end = datetime.datetime.now()
    print('Time cost: {}'.format(end -start))
    print('===> Completed!')
    print('-' * 20)
