import os

filepath1 = "C:\\Users\\zh\\Desktop\\data_province\\data\\"
filepath2 = "C:\\Users\\zh\\Desktop\\data_province\\txt\\"

path_list1 = os.listdir(filepath1) # data文件列表
print(path_list1)
for i in range(len(path_list1)):
    path_list2 = os.listdir(filepath1+path_list1[i])
    for j in range(len(path_list2)):
            with open(filepath2+path_list1[i]+".txt","a") as f:
                f.write(path_list2[j] + "\n")
                f.close()

# os.listdir(file)会历遍文件夹内的文件并返回一个列表
    #path_list = os.listdir(file_path1+filename)
    # print(path_list)
    # 定义一个空列表,我不需要path_list中的后缀名
    # path_name=[]
    # # 利用循环历遍path_list列表并且利用split去掉后缀名
    # for i in path_list:
    #     path_name.append(i.split(".")[0])
    #
    # # 排序一下
    # path_name.sort()

   # for file_name in path_list1:
   #      "a"表示以不覆盖的形式写入到文件中,当前文件夹如果没有"save.txt"会自动创建
   #      with open(file_path2+"\\txt\\"+filename+".txt","a") as f:
   #          f.write(file_name + "\n")
   #
   #      f.close()
