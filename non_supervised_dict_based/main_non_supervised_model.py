from non_supervised_model import VSMRetrieval
import csv, os


TITLE_WEIGHT = 5
CONTENT_PROP = 0.035


def get_submission_file_path(title_weight):
    return 'results/submission_weight_{}.csv'.format(title_weight)


def read_test_queries():
    queries = []
    with open('../data/test_data/test_querys2.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            queries.append((row[0], row[1]))
    return queries


def generate_submission(title_weight, build_everything=True):
    if build_everything:
        vsm = VSMRetrieval('../data/test_data/test_docs.csv', title_weight=title_weight, content_prop=CONTENT_PROP)
    else:
        vsm = VSMRetrieval('', title_weight=title_weight,content_prop=CONTENT_PROP)
    queries = read_test_queries()
    with open(get_submission_file_path(title_weight), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('query_id', 'doc_id'))
        for query in queries:
            docs = vsm.get_topK(query[0])
            for doc in docs:
                writer.writerow((query[1], doc[1]))


# weight_list = [1, 5, 10, 20, 100]
weight_list = [TITLE_WEIGHT]
if __name__ == "__main__":
    for weight in weight_list:
        print("now generating submission with weight = ", weight)
        if weight == weight_list[0]:
            generate_submission(weight, True)
        else:
            generate_submission(weight, False)
        
