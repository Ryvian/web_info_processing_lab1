import os, pickle


DF_CHUNK_SIZE = 512

class DistributedFile:
    """save and read data in/from multiple files. 
    Used for the TF_iDF table here (word_list length: 336082, doc numbers: 13139). Need about 35 GB memory
    """
    def __init__(self, name='data', dir_path='./pickled_data'):
        self.index = {}
        self.name = name
        self.obj = {}
        self.obj_index = -1
        self.dir_path = dir_path
        self.dump_index = 0
        self.mode = ''

    def dump(self, data: dict,  func=lambda x: x, chunk_size=DF_CHUNK_SIZE, append=False):
        self.mode = 'w'
        if not append:
            self.dump_index = 0
        else:
            self.dump_index += 1
        to_dump = {}
        for i, key in enumerate(data):
            self.index[key] = self.dump_index
            to_dump[key] = func(data[key])
            if (i % chunk_size) == chunk_size - 1:
                with open(os.path.join(
                        self.dir_path, 
                        'distributed_file_{}_{}.pickle'.format(self.name, self.dump_index)), 'wb') as f:
                    pickle.dump(to_dump, f)
                del to_dump
                to_dump = {}
                self.dump_index += 1
        if len(to_dump) != 0:
            with open(os.path.join(
                    self.dir_path,
                    'distributed_file_{}_{}.pickle'.format(self.name, self.dump_index)), 'wb') as f:
                    pickle.dump(to_dump, f)
            del to_dump
        else:
            self.dump_index -= 1
        with open(os.path.join(
                self.dir_path, 
                'index_distributed_file_{}.pickle'.format(self.name)), 'wb') as f:
                    pickle.dump(self.index, f)
    
    def read(self, key):
        self.mode = 'r'
        if self.obj_index == -1:
            with open(os.path.join(
                        self.dir_path, 
                        'index_distributed_file_{}.pickle'
                        .format(self.name)), 'rb') as f:
                self.index = pickle.load(f)
            self.obj_index = self.index[key]
            with open(os.path.join(
                        self.dir_path, 
                        'distributed_file_{}_{}.pickle'
                        .format(self.name, self.obj_index)), 'rb') as f:
                self.obj = pickle.load(f)
            return self.obj[key]
        else:
            if self.index[key] == self.obj_index:
                return self.obj[key]
            else:
                self.obj_index = self.index[key]
                with open(os.path.join(
                            self.dir_path, 
                            'distributed_file_{}_{}.pickle'
                            .format(self.name, self.obj_index)), 'rb') as f:
                    self.obj = pickle.load(f)
                return self.obj[key]
