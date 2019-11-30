import jieba
import pandas as pd
import utils
import numpy as np
from non_supervised_model import VSMRetrieval
from datetime import datetime


seg_list = jieba.lcut_for_search("10月份药品养护汇总分析 - 豆丁网。Apple公司发布的新产品 iPad Pro 2019")
print(seg_list)

data = {
    'a': [1, 2, 3]
}
data2 = {
    'b': [3,4,5]
}
# df = pd.DataFrame.from_dict(data, orient='index', columns=['A', 'B', 'C'])
# df.to_hdf('test.hdf', key='df', format='table', mode='w', append=True)
# df2 = pd.DataFrame.from_dict(data2, orient='index', columns=['A', 'B', 'C'])
# df2.to_hdf('test.hdf', key='df', format='table', append=True)
# df3 = pd.read_hdf('test.hdf', key='df')
# print(df3, type(df3))

vsm = VSMRetrieval(title_weight=1, content_prop=0)
string = "年均增长率怎么算"
print("length of word_list:", len(vsm.iDF))
print("length of docs_ids:", len(vsm.TF_iDF))
# seg_list = jieba.cut_for_search(string)
# for word in seg_list:
#     print(word, word in vsm.word_set)
# vector = vsm.get_relation_vector(string)
# print("norm of vector:", np.linalg.norm(vector))
# for doc, word_values in vsm.TF_iDF.items():
#     if len(word_values) == 0:
#         print("zero column in TF_iDF:", doc)
before = datetime.now()
docs = vsm.get_topK(string)
after = datetime.now()
print('test time: {}'.format(str(datetime.now())))
print('time used:', str(after - before))
print(docs)