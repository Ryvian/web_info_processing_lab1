from non_supervised_model import VSMRetrieval
import csv, os


def table_exists(title_weight):
    return os.path.exists('pickled_data/TF_iDF_{}.pickle'.format(title_weight))

if __name__ == "__main__":
    with open('data/query_doc_relativeness.csv', encoding='utf-8') as f:
        query_dict = {}
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            query = row[0]
            doc_id = row[2]
            label = row[3]
            if label == "3":
                if query in query_dict:
                    query_dict[query].append(doc_id)
                else:
                    query_dict[query] = [doc_id]
    for query, docs in query_dict.items():
        if len(docs) == 0:
            print('warning: zero docs:', query)
    weight_list = [1, 5, 10, 15, 20]
    prop_list = []
    for title_weight in weight_list:
        if table_exists(title_weight):
            vsm = VSMRetrieval(title_weight=title_weight, load_object=True)
        else:
            vsm = VSMRetrieval(title_weight=title_weight)
        num = 0
        total = 0
        for query, real_doc_list in query_dict.items():
            results = vsm.get_topK(query, topK=len(real_doc_list))
            for result in results:
                if result in real_doc_list:
                    num += 1
            total += len(real_doc_list)
        prop_list.append(num / total)
    with open('results/non_supervised_test.txt', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('title weight', 'proportion'))
        for i, weight in enumerate(weight_list):
            writer.writerow((weight, prop_list[i]))

