import numpy as np
import json

def serialize_array(ar):
    shape = ar.shape
    type = ar.dtype
    cl = ar.__class__
    ar = ar.tolist()
    ar = {'data':ar, "type":type, 'shape':shape, 'class': cl}
    return ar

def deserialize_array(serialized_array):
    ar = serialized_array['data']
    if serialized_array['class'] == np.ndarray:
        ar = np.array(ar)
    ar.shape = serialized_array['shape']
    ar.dtype = serialized_array['type']
    return ar


def serialize_dict(dict):
    serial = {}
    for elem in dict:
        if isinstance(dict[elem], np.ndarray):
            serial[elem] = serialize_array(dict[elem])
        else:
            serial[elem] = dict[elem]
    return serial

def deserialize_dict(seri):
    dict = {}
    for elem in seri:
        print(elem)
        if 'class' in seri[elem] and seri[elem]['class'] == np.ndarray:
            dict[elem] = deserialize_array(seri[elem])
        else:
            dict[elem] = seri[elem]
    return dict


if __name__ == '__main__':
    b = np.array([[1,2,3], [1,2,3]])
    ser = serialize_array(b)
    ar = deserialize_array(ser)

    d = {'a': ar}
    seri = serialize_dict(d)
    dd = deserialize_dict(seri)