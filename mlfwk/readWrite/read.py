from pandas import read_csv
from mlfwk.utils import get_project_root


def load_base(path='', type=None):
    path = get_project_root() + '/mlfwk/datasets/' + path

    if type == 'csv':
        base_result = read_csv(path)
    else:
        base_result = None

    return base_result

