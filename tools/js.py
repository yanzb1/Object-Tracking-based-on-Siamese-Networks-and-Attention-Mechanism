import json
import os, sys

json_path = '/media/sci/移动硬盘'
json_path1 = 'VOT2016_1.json'
txt_path = 'list.txt'
dict = {}


def get_json_data(json_path):
    with open(json_path) as f:

        params = json.load(f)
        # add '/color/'to the path
        file = open("list.txt")
        while 1:
            lines = file.readlines(1000)
            if not lines:
                break
            for line in lines:
                line = line[:-1]
                root = (params[line]["img_names"])
                # print(len(root))
                while 1:
                    for i in range(len(root)):
                        kind, jpg = root[i].split("/")
                        root[i] = kind + '/color/' + jpg
                    # print(root)
                    break

        file.close()
        # print("params",params)
        dict = params
    f.close()
    return dict


def write_json_data(dict):
    with open(json_path1, 'w') as r:
        json.dump(dict, r)
    r.close()


if __name__ == "__main__":
    the_revised_dict = get_json_data(json_path)

    write_json_data(the_revised_dict)
# -*-ing:utf-8-*-
