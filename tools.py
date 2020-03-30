import os
import yaml
import random
import datetime
import argparse


def kflod(K, fpath):
    flods = []
    size = {}
    for i in range(K):
        size[str(i)] = 0
    with open(fpath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            label = line.split(' ')[0]
            path = line.split(' ')[1]
            flod = random.randint(0, K - 1)
            flods.append({'flod':flod, 'label':label, 'path': path})
            size[str(flod)] += 1

    if not os.path.exists('./DataSet'):
        os.mkdir('./DataSet')
    with open('./DataSet/kflod.yaml', 'w') as f:
        yaml.dump(flods, f, default_flow_style=False)

    for key, value in size.items():
        print("Fold {} size: {} items".format(key, value))
    

def stopWords(path='./StopWords/stopwords.txt'):
    stop_words = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip('\n'))
    return stop_words


def read_record(path):
    record = {}
    with open(path, 'r') as f:
        record = yaml.load(f, Loader=yaml.FullLoader)
    return record
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=5, type=int, help="Number of fold")
    parser.add_argument("--fold_path", default='./trec06p/label/index', type=str, help="Path of label index")
    parser.add_argument("--stopwords_path", default='./StopWords/stopwords.txt', type=str, help="Path of stopwords")
    parser.add_argument("--seed", default='2017013632', type=str, help="Seed for random")
    args = parser.parse_args()

    random.seed(args.seed)
    print('===> Dividing Train Set')
    start = datetime.datetime.now()
    kflod(K=args.k, fpath=args.fold_path)
    end = datetime.datetime.now()
    print('Time cost: {}'.format(end -start))
    print('===> Completed!')
    print('-' * 20)
