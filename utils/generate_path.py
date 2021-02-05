import pandas as pd

base_dir = "/home/workspace/data/IMG/"

csv_file = pd.read_csv("../../data/driving_log.processed.2.csv")

image_file_name = csv_file["image"].values
angles = csv_file["steering"].values


data_set = []

for i in range(len(image_file_name)):
    print("Number of data: \t{}\r".format(i+1), end="")
    name = base_dir + image_file_name[i].split("/")[-1]
    angle = angles[i]
    data_set.append([name, angle])
    
new_csv = pd.DataFrame(data_set, columns=["image", "steering"])

new_csv.to_csv("../../data/driving_log.processed.3.csv")

    
