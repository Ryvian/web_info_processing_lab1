import os, csv, pickle, math, sys, datetime
import numpy as np
import pandas as pd
import jieba

def wide_df_to_hdf(filename, data, columns=None, maxColSize=2000, mode='a',append=False, **kwargs):
    """Write a `pandas.DataFrame` with a large number of columns
    to one HDFStore.

    Parameters
    -----------
    filename : str
        name of the HDFStore
    data : pandas.DataFrame
        data to save in the HDFStore
    columns: list
        a list of columns for storing. If set to `None`, all 
        columns are saved.
    maxColSize : int (default=2000)
        this number defines the maximum possible column size of 
        a table in the HDFStore.

    """
    import numpy as np
    from collections import ChainMap
    store = pd.HDFStore(filename, **kwargs)
    if columns is None:
        columns = data.columns
    colSize = columns.shape[0]
    if colSize > maxColSize:
        numOfSplits = np.ceil(colSize / maxColSize).astype(int)
        colsSplit = [
            columns[i * maxColSize:(i + 1) * maxColSize]
            for i in range(numOfSplits)
        ]
        _colsTabNum = ChainMap(*[
            dict(zip(columns, ['data{}'.format(num)] * colSize))
            for num, columns in enumerate(colsSplit)
        ])
        colsTabNum = pd.Series(dict(_colsTabNum)).sort_index()
        for num, cols in enumerate(colsSplit):
            store.put('data{}'.format(num), data[cols], format='table', mode=mode, append=append)
        store.put('colsTabNum', colsTabNum, format='fixed')
    else:
        store.put('data', data[columns], format='table', mode=mode, append=append)
    store.close()


def read_hdf_wide_df(filename, columns=None, **kwargs):
    """Read a `pandas.DataFrame` from a HDFStore.

    Parameter
    ---------
    filename : str
        name of the HDFStore
    columns : list
        the columns in this list are loaded. Load all columns, 
        if set to `None`.

    Returns
    -------
    data : pandas.DataFrame
        loaded data.

    """
    store = pd.HDFStore(filename)
    data = []
    colsTabNum = store.select('colsTabNum')
    if colsTabNum is not None:
        if columns is not None:
            tabNums = pd.Series(
                index=colsTabNum[columns].values,
                data=colsTabNum[columns].data).sort_index()
            for table in tabNums.unique():
                data.append(
                    store.select(table, columns=tabsNum[table], **kwargs))
        else:
            for table in colsTabNum.unique():
                data.append(store.select(table, **kwargs))
        data = pd.concat(data, axis=1).sort_index(axis=1)
    else:
        data = store.select('data', columns=columns)
    store.close()
    return data


class VSMRetrieval:
    """
    build TF-iDF model, use VSM to get the ranking of docs of a request

    """
    def __init__(self, filename='', title_weight=1, load_object=False):
        #chunk size of rows when building TF_iDF table
        self.TF_iDF_BUILD_CHUNK = 2000
        self.TF_iDF_STORE_PATH = "pickled_data/TF_iDF_table.hdf5"
        """
        filename: the original csv file you want to load data from
            if filename is '':
                if load_object is False:
                    load pickled data for title_TF and content_TF. Others are rebuilt
                if load_object is True:
                    load all data for this object from history. Note title_weight
                    here won't have effect

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
                for row in reader:
                    doc_id = row[0]
                    doc_ids.append(doc_id)
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
            with open('pickled_data/title_TF.pickle', 'wb') as f:
                pickle.dump(title_TF, f)
            with open('pickled_data/content_TF.pickle', 'wb') as f:
                pickle.dump(TF, f)
            with open('pickled_data/doc_ids.pickle', 'wb') as f:
                pickle.dump(doc_ids, f)
            self.merge_TF_table(title_TF, TF, title_weight)

            #build iDF table
            self.iDF = dict()
            for key in TF:
                idf = math.log10(len(doc_ids) / len(TF[key]))
                self.iDF[key] = idf
            
            self.TF_iDF = self.build_TF_iDF_table(TF, doc_ids)
            self.pickle()

        else:
            #load directly. title_weight here won't have effect
            if load_object:
                with open('pickled_data/iDF.pickle', 'rb') as f:
                    self.iDF = pickle.load(f)
                self.TF_iDF = pd.read_hdf(self.TF_iDF_STORE_PATH)

            #only load title_TF and content_TF, rebuild others. title_weight has effect
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
                self.merge_TF_table(title_TF, TF, title_weight)
                print('after merge')
                #build iDF table
                self.iDF = dict()
                for key in TF:
                    idf = math.log10(len(doc_ids) / len(TF[key]))
                    self.iDF[key] = idf
                print('before build TF_iDF')
                self.TF_iDF = self.build_TF_iDF_table(TF, doc_ids)
                print('after build TF_iDF')
                
    
    def merge_TF_table(self, title_TF: dict, content_TF: dict, title_weight: float):
        """
        merge title_TF and content_TF in place, with title_TF multipilied by title_weight
        result are in content_TF
        """
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

    def build_TF_iDF_table(self, TF: dict, doc_ids: list):
        TMP_PATH = self.TF_iDF_STORE_PATH
        if os.path.exists(TMP_PATH):
            os.remove(TMP_PATH)

        def save_table_tmp(table: pd.DataFrame):
            wide_df_to_hdf(TMP_PATH, table, append=True)

        def load_table_tmp():
            return read_hdf_wide_df(TMP_PATH)

        TF_iDF = dict()
        print("TF length:", len(TF))
        print('before TF_iDF dict is built')
        i = 0
        for word, TF_dict in TF.items():
            TF_iDF_list = []
            for doc in doc_ids:
                if doc in TF_dict:
                    TF_iDF_list.append(
                        self.calc_log_tf(TF_dict[doc]) * self.iDF[word]
                    )
                else:
                    TF_iDF_list.append(0)
            TF_iDF[word] = TF_iDF_list

            if (i % self.TF_iDF_BUILD_CHUNK) == self.TF_iDF_BUILD_CHUNK - 1:
                TF_iDF_df = pd.DataFrame.from_dict(TF_iDF, orient='index', columns=doc_ids)
                del TF_iDF
                save_table_tmp(TF_iDF_df)
                del TF_iDF_df
                TF_iDF = dict()
                print('Until', i, 'th row of TF_iDF is saved')
            i = i + 1

        TF_iDF_df = pd.DataFrame.from_dict(TF_iDF, orient='index', columns=doc_ids)
        save_table_tmp(TF_iDF_df)
        del TF_iDF
        del TF_iDF_df
        print('after TF_iDF dict is built')
        return load_table_tmp()
        
    def calc_log_tf(self, tf):
        """
        calculate the logarithm TF
        """
        if tf == 0:
            return 0.0
        return 1 + math.log10(tf)

    def pickle(self, both=False):
        with open('pickled_data/iDF.pickle', 'wb') as f:
            pickle.dump(self.iDF, f)
        if both:
            wide_df_to_hdf(self.TF_iDF_STORE_PATH, self.TF_iDF, mode='w')

    def get_relation_vector(self, s: str) -> pd.Series:
        """
        calculate relation vector for a string
        """
        words = self.TF_iDF.index
        vector = {}
        for word in words:
            vector[word] = 0.0
        vector = pd.Series(vector)
        seg_list = jieba.cut_for_search(s)
        for word in seg_list:
            vector[word] += 1
        for word in vector.index:
            vector[word] = self.calc_log_tf(vector[word]) * self.iDF[word]
        return vector            

    def get_similarity(self, vec1: pd.Series, vec2: pd.Series):
        """
        cosine similarity
        """
        vec1 = vec1.values
        vec2 = vec2.values
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def get_topK(self, s: str, topK=20):
        """
        get the most similar K docs' ids, using VSM and cosine similarity
        """
        doc_list = []
        s_vector = self.get_relation_vector(s)
        for doc in self.TF_iDF.cloumns:
            doc_vector = self.TF_iDF[doc]
            sim = self.get_similarity(doc_vector, s_vector)
            #maintain top K list
            if len(doc_list) < topK:
                doc_list.append((sim, doc))
                doc_list.sort(key=lambda x: x[0], reverse=True)
            elif sim > doc_list[-1][0]:
                doc_list[-1] = (sim, doc)
                doc_list.sort(key=lambda x: x[0], reverse=True)
        return doc_list


if __name__ == "__main__":
    vsm = VSMRetrieval(title_weight=10)
    vsm.pickle()


