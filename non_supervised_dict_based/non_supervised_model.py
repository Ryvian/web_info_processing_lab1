"""
python3
using VSM
"""

import os, csv, pickle, math, sys, datetime
import numpy as np
import pandas as pd
import jieba
from numba import jitclass, njit
import numba
import random
# using pathos for "multiprocess"  here
# install by: pip install pathos
from multiprocess import Pool
from itertools import chain

NUM_PROCESSES = 4

spec = [
    ('iDF', numba.types.DictType(
        numba.types.unicode_type,
        numba.types.float64
    )),
    ('word_list', numba.types.unicode_type[:]),
    ('TF_iDF', numba.types.DictType(
        numba.types.unicode_type,
        numba.types.DictType(
            numba.types.unicode_type,
            numba.types.float64
        )
    ))

]



class VSMRetrieval:
    """
    build TF-iDF model, use VSM to get the relevant docs of a query

    """
    def __init__(self, filename='', title_weight=1, content_prop=1, load_object=False):
        """
        filename: *(default: '')*. The original csv file you want to load doc 
        data from to build the model
            if filename is '':
                if load_object is False:
                    load pickled data for title_TF and content_TF.
                    Others are rebuilt
                if load_object is True:
                    load all data for this object from history, if the corresponding 
                    title_weight history exists.
            else:
                rebuild everything according to the data in the file path

        content_prop:
            float: 0~1
            the proportion of the words in content considered in processing
            the lower the quicker (but less informative or accurate)

        title_weight:
            >=0
            the weight of words in title relative to that in content, used when calculate TF
        """
        #build TF, iDF, TF-iDF table, and pickle them
        # jieba.enable_parallel(8)
        if filename != '':
            title_TF = dict()
            TF = dict()
            with open(filename, encoding='utf-8') as f:
                csv.field_size_limit(500 * (2 ** 20))
                reader = csv.reader(f)
                header = next(reader)
                doc_ids = []
                doc_id_set = set()
                for row in reader:
                    doc_id = row[0]
                    if doc_id in doc_id_set:
                        continue
                    doc_ids.append(doc_id)
                    doc_id_set.add(doc_id)
                    # title
                    seg_list = jieba.cut_for_search(row[2])
                    for word in seg_list:
                        if word in title_TF:
                            if doc_id in title_TF[word]:
                                title_TF[word][doc_id] += 1
                            else:
                                title_TF[word][doc_id] = 1
                        else:
                            title_TF[word] = {doc_id: 1}
                    #content
                    seg_list = jieba.cut_for_search(row[3])
                    for word in seg_list:
                        if word in TF:
                            if doc_id in TF[word]:
                                TF[word][doc_id] += 1
                            else:
                                TF[word][doc_id] = 1
                        else:
                            TF[word] = {doc_id: 1}
            # del doc_id_set
            with open('pickled_data/title_TF.pickle', 'wb') as f:
                pickle.dump(title_TF, f)
            with open('pickled_data/content_TF.pickle', 'wb') as f:
                pickle.dump(TF, f)
            with open('pickled_data/doc_ids.pickle', 'wb') as f:
                pickle.dump(doc_ids, f)
            self.merge_TF_table(title_TF, TF, title_weight, content_prop)
            #build iDF table
            self.iDF = self.build_iDF(TF, doc_ids)
            # self.iDF = numba.typed.Dict.empty(
            #     key_type=numba.types.unicode_type,
            #     value_type=numba.types.float64
            # )
        
            self.TF_iDF = self.build_TF_iDF_table(TF, doc_ids)
            self.pickle(title_weight)

        else:
            #load from pickled file directly if corresponding data exists
            if load_object:
               self.load(title_weight)

            #only load title_TF and content_TF, rebuild others. 
            # title_weight and content_prop has effect
            else:
                with open('pickled_data/title_TF.pickle', 'rb') as f:
                    title_TF = pickle.load(f)
                    print('title_TF length:', len(title_TF))
                with open('pickled_data/content_TF.pickle', 'rb') as f:
                    TF = pickle.load(f)
                    print('content_TF length:', len(TF))
                with open('pickled_data/doc_ids.pickle', 'rb') as f:
                    doc_ids = pickle.load(f)
                print('before merge')
                self.merge_TF_table(title_TF, TF, title_weight, content_prop=content_prop)
                print('after merge')
                self.iDF = self.build_iDF(TF,doc_ids)
                print('before build TF_iDF')
                self.TF_iDF = self.build_TF_iDF_table(TF, doc_ids)
                print('after build TF_iDF')
                self.pickle(title_weight=title_weight)
      
    def build_iDF(self, TF, doc_ids):
        iDF = {}
        for key in TF:
            idf = math.log10(len(doc_ids) / len(TF[key]))
            iDF[key] = idf
        return iDF

    def merge_TF_table(self, title_TF: dict, content_TF: dict, title_weight: float, content_prop: float):
        """
        merge title_TF and content_TF in place, with title_TF multipilied by title_weight
        result are in content_TF
        """
        del_num = round(len(content_TF) * (1 - content_prop))
        del_word_list = random.sample(content_TF.keys(), del_num)
        for word in del_word_list:
            del content_TF[word]
        for word, TF_dict in title_TF.items():
            if word not in content_TF:
                new_TF_dict = {}
                for doc, tf in TF_dict.items():
                    new_TF_dict[doc] = title_weight * tf
                content_TF[word] = new_TF_dict
            else:
                for doc, tf in TF_dict.items():
                    if doc not in content_TF[word]:
                        content_TF[word][doc] = title_weight * tf
                    else:
                        content_TF[word][doc] += title_weight * tf

    def build_TF_iDF_table(self: dict, TF: dict, doc_ids: list):        
        print("TF length:", len(TF))
        print('before TF_iDF dict is built')
        i = 0
        #element of TF_iDF: TF_iDF[doc_id][word]
        #to better later performance
        TF_iDF = dict()
        for doc in doc_ids:
            TF_iDF[doc] = {}
        for word, TF_dict in TF.items():
            for doc in doc_ids:
                if doc in TF_dict:
                    TF_iDF[doc][word] = (
                        self.calc_log_tf(TF_dict[doc]) * self.iDF[word]
                    )
        return TF_iDF

    def get_TF_iDF_value(self, word: str, doc_id: str):
        if word in self.TF_iDF[doc_id]:
            return self.TF_iDF[doc_id][word]
        else:
            return 0.0

    @staticmethod
    @njit
    def calc_log_tf(tf: float):
        """
        calculate the logarithm TF
        """
        if tf == 0:
            return 0.0
        return 1 + math.log10(tf)

    def pickle(self, title_weight, only_TF_iDF=False):
        with open('pickled_data/TF_iDF_{}.pickle'.format(title_weight), 'wb') as f:
            pickle.dump(self.TF_iDF, f)
        if not only_TF_iDF:
            with open('pickled_data/iDF.pickle', 'wb') as f:
                pickle.dump(self.iDF, f)
    
    def load(self, title_weight):
        with open('pickled_data/iDF.pickle', 'rb') as f:
            self.iDF = pickle.load(f)
        with open('pickled_data/TF_iDF_{}.pickle'.format(title_weight), 'rb') as f:
            self.TF_iDF = pickle.load(f)

    def get_TF_iDF_doc_vector(self, doc: str):
        vector = []
        for word in self.iDF:
            vector.append(self.get_TF_iDF_value(word, doc))
        return np.array(vector, dtype=np.float64)

    def get_relation_vector(self, s: str) -> np.array:
        """
        calculate relation vector for a string
        """
        vector = {}
        for word in self.iDF:
            vector[word] = 0.0
        seg_list = jieba.cut_for_search(s)
        for word in seg_list:
            if word in self.iDF:
                vector[word] += 1.0
        vector_list = []
        for word in vector:
            vector_list.append(self.calc_log_tf(vector[word]) * self.iDF[word])
        vector = np.array(vector_list, dtype=np.float64)
        return vector

    @staticmethod
    @njit
    def get_similarity(vec1: np.array, vec2: np.array, s, doc):
        """
        cosine similarity
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0.0:
            print('warning: questionable query, vector norm == 0: '+ s)
            return -1
        if norm2 == 0.0:
            print('Warning: questionable doc, vector norm == 0: '+ doc)
            return -1
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def get_topK(self, s: str, topK=20):
        """
        get the most similar K docs' ids, using VSM and cosine similarity
        """
        s_vector = self.get_relation_vector(s)

        def get_topK_aux(keys):
            doc_list = []
            for doc in keys:
                doc_vector = self.get_TF_iDF_doc_vector(doc)
                sim = self.get_similarity(s_vector, doc_vector, s, doc)
                #maintain top K list
                if len(doc_list) < topK:
                    doc_list.append((sim, doc))
                    doc_list.sort(key=lambda x: x[0], reverse=True)
                elif sim > doc_list[-1][0]:
                    doc_list[-1] = (sim, doc)
                    doc_list.sort(key=lambda x: x[0], reverse=True)
            return doc_list

        pool = Pool(processes=NUM_PROCESSES)
        chunk_size = len(self.TF_iDF) // NUM_PROCESSES
        keys = list(self.TF_iDF.keys())
        result_list = []
        for i in range(NUM_PROCESSES-1):
            res = pool.apply_async(get_topK_aux, (keys[i*chunk_size:(i+1)*chunk_size],))
            result_list.append(res)
        res = pool.apply_async(get_topK_aux, (keys[(NUM_PROCESSES-1)*chunk_size:],))
        result_list.append(res)
        pool.close()
        pool.join()
        doc_list = []
        for res in result_list:
            doc_list += res.get()
        doc_list.sort(key=lambda x: x[0], reverse=True)
        return doc_list[:topK]


if __name__ == "__main__":
    vsm = VSMRetrieval('data/doc_data.csv', title_weight=5)

    # vsm = VSMRetrieval(title_weight=5)
    # vsm.pickle()





