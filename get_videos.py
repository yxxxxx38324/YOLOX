#coding:utf-8
import os
import cv2
import re
 
def get_images(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        if not files:
            continue
        for file in files:
            if file.endswith('.png'):
                file_list.append(os.path.join(root, file))
                #file_list.append(file)
    return file_list
 
def key_sort(image_path):
    pattern = re.compile("\d+")
    image_name = os.path.basename(image_path)
    return int(pattern.findall(image_name)[0])
 
def main():
	#图片路径
    path = "/media/young/MyPassport/跨越险阻/Raw-001/Raw-001-Camera"
	#将图片存入列表
    file_list = get_images(path)
	#按照图片名称排序
    file_list.sort(key=key_sort)
	#一秒25帧，代表1秒视频由25张图片组成
    fps = 15
	#视频分辨率
    img_size = (1024, 768) 
	#保存视频的路径
    save_path = "/media/young/MyPassport/跨越险阻/Raw-001/raw.avi"
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, img_size)
    for file_name in file_list:
        # print(file_name)
        img = cv2.imread(file_name)
        video_writer.write(img)
 
    video_writer.release()
 
if __name__ == "__main__":
    main()