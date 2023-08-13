import pickle
import os

def save_objects(obj,file_path):
    dir_path=os.path.dirname(file_path)
    os.makedirs(dir_path,exist_ok=True)
    pickle.dump(obj,open(file_path,'wb'))

def load_object(file_path):
    obj=pickle.load(open(file_path,'rb'))
    return obj