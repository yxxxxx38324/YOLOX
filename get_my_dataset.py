import os
import csv
import json
import random
basePath = '/media/young/MyPassport/跨越险阻'
categoryDic = {"H":1, "V":2, "B":3}

def get_csv_dic():
    dataFolders = [x for x in os.listdir(basePath) if x.split('.')[-1] != 'zip']
    all_data = {}
    for folder in dataFolders:
        gtPath = basePath + '/' + folder + '/' + folder + '-' + 'Groundtruth' + '/gt_detection.csv' 
        if ( not os.path.exists(gtPath)):
            continue
        all_data[folder] = {}
        f = open(gtPath, "r")
        csv_list = []
        for row in f:
            if row.split(",")[3].replace(" ", "") == 'C':
                csv_list.append(row)
        f.close()

        all_timestamp = []
        for row in csv_list:
            all_timestamp.append(row.split(",")[1].replace(" ", ""))
        all_timestamp = list(set(all_timestamp))
        all_timestamp.sort(key = int)
        for timestamp in all_timestamp:
            all_data[folder][timestamp] = []
            for row in csv_list:
                # print(row)
                if (row.split(",")[1].replace(" ", "") == timestamp):
                    row_dic = {}
                    row_dic["category"] = row.split(",")[4].replace(" ", "")
                    row_dic["box"] = [float(x) for x in row.split("\"")[1][1:-1].split(",")]
                    row_dic["position"] = [float(x) for x in row.split("\"")[3][1:-1].split(",")]
                    all_data[folder][timestamp].append(row_dic)
       

    return all_data
#2391 2044 1863 1572 2855 2839 = 13565
#2221 1671 1303 1282 1415 2031 = 9923

def timeStampToImg(result_folder):
    timeStampToImg_dic = {}
    csvpath = basePath + '/' + result_folder + '/' + result_folder + "-Camera-Timestamp.csv"
    with open(csvpath) as f:
        tsp = []
        for line in f:
            timestamp = line.split(',')[0]
            tsp.append(timestamp)
            img_name = line.split(',')[1].replace(" ", "")
            timeStampToImg_dic[timestamp] = img_name[0:-1]
        # print(len(set(tsp)) == len(tsp))
    return timeStampToImg_dic


def get_coco_json(all_data):
    all_json_data = {}

    all_json_data["train"] = {}
    all_json_data["test"] = {}

    #info
    all_json_data["train"]["info"] = {"year": 2021, "version": "1.0", "description": "For object detection", "date_created": "2021"}
    all_json_data["test"]["info"] = {"year": 2021, "version": "1.0", "description": "For object detection", "date_created": "2021"}

    #licenses
    all_json_data["train"]["licenses"] = [{"id": 1, "name": "GNU General Public License v3.0", "url": "https://github.com/zhiqwang/yolov5-rt-stack/blob/master/LICENSE"}]
    all_json_data["test"]["licenses"] = [{"id": 1, "name": "GNU General Public License v3.0", "url": "https://github.com/zhiqwang/yolov5-rt-stack/blob/master/LICENSE"}]

    #type
    all_json_data["train"]["type"] = "instances"
    all_json_data["test"]["type"] = "instances"

    #category
    all_json_data["train"]["categories"] = [
    {"id": 1, "name": "H", "supercategory": "1"}, 
    {"id": 2, "name": "V", "supercategory": "2"}, 
    {"id": 3, "name": "B", "supercategory": "3"}, 
    ]
    all_json_data["test"]["categories"] = [
    {"id": 1, "name": "H", "supercategory": "1"}, 
    {"id": 2, "name": "V", "supercategory": "2"}, 
    {"id": 3, "name": "B", "supercategory": "3"}
    ]

    #images and annotations
    all_json_data["train"]["images"] = []
    all_json_data["train"]["annotations"] = []

    all_json_data["test"]["images"] = []
    all_json_data["test"]["annotations"] = []

    img_num = {"train":1, "test":1}
    det_num = {"train":1, "test":1}
    for k in all_data.keys():
        timeStampToImg_dic = timeStampToImg(k)
        used_timestamp = []
        tsp_list = list(all_data[k].keys())
        random.shuffle(tsp_list)
        for timestamp in tsp_list:
            x = random.uniform(0,1)
            dataset_type = "train" if x > 0.2 else "test"

            image_info = {}
            image_info["date_captured"] = "2021"
            image_info["file_name"] = str(img_num[dataset_type]) + ".png"
            image_info["id"] = img_num[dataset_type]
            image_info["height"] = 768
            image_info["width"] = 1024

            src_img = basePath + '/' + k + '/' + k+"-Camera" + '/' + timeStampToImg_dic[timestamp]
            dst_img = basePath + '/' + 'coco_format_dataset' + '/' + dataset_type + '/' + image_info["file_name"]
            command = "cp" + " " + src_img + " " + dst_img
            print(command)
            os.system(command)
            
            all_json_data[dataset_type]["images"].append(image_info)

            for det in all_data[k][timestamp]:
                annotation = {}
                annotation["segmentation"] = det["box"]
                annotation["area"] = (det["box"][2]- det["box"][0]) * (det["box"][5] - det["box"][1])
                annotation["iscrowd"] = 0
                annotation["image_id"] = img_num[dataset_type]
                annotation["bbox"] = [det["box"][0], det["box"][1], det["box"][2]- det["box"][0], det["box"][5] - det["box"][1]]
                annotation["category_id"] = categoryDic[det["category"]]
                annotation["id"] = det_num[dataset_type]
                all_json_data[dataset_type]["annotations"].append(annotation)
                det_num[dataset_type] += 1
            img_num[dataset_type] += 1
            pass
    return all_json_data


if __name__ == "__main__":
    all_data = get_csv_dic()
    all_json_data = get_coco_json(all_data)
    with open(basePath + "/coco_format_dataset/annotations/instances_train2021.json", "w") as f:
        json.dump(all_json_data["train"],f)
    
    with open(basePath + "/coco_format_dataset/annotations/instances_test2021.json", "w") as f:
        json.dump(all_json_data["test"],f)
    
    annos = all_json_data["train"]["annotations"]
    category_nums_train = [0, 0, 0] #H V B 
    for anno in annos:
        category_nums_train[anno["category_id"]-1] += 1

    annos = all_json_data["test"]["annotations"]
    category_nums_test = [0, 0, 0] #H V B 
    for anno in annos:
        category_nums_test[anno["category_id"]-1] += 1
    
    
    print("training set info:")
    print("img num: {0}, det num: {1}".format(len(all_json_data["train"]["images"]), len(all_json_data["train"]["annotations"])))
    print("Human: {0}, Vehicle: {1}, Box: {2}}".format(category_nums_train[0], category_nums_train[1], category_nums_train[2]))

    print("test set info: ")
    print("img num: {0}, det num: {1}".format(len(all_json_data["test"]["images"]), len(all_json_data["test"]["annotations"])))
    print("Human: {0}, Vehicle: {1}, Box: {2}".format(category_nums_test[0], category_nums_test[1], category_nums_test[2]))

