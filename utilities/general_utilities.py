import importlib.util


def import_from_path(path):
    spec = importlib.util.spec_from_file_location('module_spec', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def append_key_dict(d, k):
    return {k+kk: vv for kk, vv in d.items()}


def choose_1_from_dict_of_lists(dict_of_lists, choice, result, i_choice=0):
    """
    :param dict_of_lists: {a: {b:[1,2,3], c:[11,22]}, d:{e:{f:[111, 222]}}}
    :param choice: [2,1,2]
    :param result: must have all the keys in dict_of_lists
    :param i_choice: the index on the choices
    :return: {a: {b: 2, c:11}, d:{e:{f:222}}}
    """

    for k in sorted(dict_of_lists.keys()):
        dtype = type(dict_of_lists[k])
        if dtype is dict:
            i_choice = choose_1_from_dict_of_lists(dict_of_lists[k], choice, result[k], i_choice)
        elif dtype is list:
            result[k] = dict_of_lists[k][choice[i_choice]]
            i_choice += 1
        else:
            raise NotImplementedError
    return i_choice


def __get_list_size_of_dict_of_lists(dict_of_lists, result):
    for k in sorted(dict_of_lists.keys()):
        dtype = type(dict_of_lists[k])
        if dtype is dict:
            __get_list_size_of_dict_of_lists(dict_of_lists[k], result)
        elif dtype is list:
            result.append(len(dict_of_lists[k]))
        else:
            raise NotImplementedError
    return result


def get_list_size_of_dict_of_lists(dict_of_lists):
    return __get_list_size_of_dict_of_lists(dict_of_lists, [])


def is_power2(num):
    """
    Author: A.Polino
    states if a number is a power of two
    """
    return num != 0 and ((num & (num - 1)) == 0)


