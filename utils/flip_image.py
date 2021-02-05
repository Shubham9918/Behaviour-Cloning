import pandas as pd
from PIL import Image, ImageOps

base_dir = "/home/workspace/data/IMG/"

csv_file = pd.read_csv("../../data/driving_log.processed.1.csv")

data_set = []

for i in csv_file.index:
    print("Number of data: \t{}\r".format(i+1), end="")
    image_file_name = csv_file["image"][i]
    angle = csv_file["steering"][i]
    
    image = Image.open(base_dir+image_file_name.split("/")[-1])
    
    flippedimage= ImageOps.mirror(image)
    
    new_name = base_dir+image_file_name.replace(".jpg", "_flip.jpg").split("/")[-1]
    
    flippedimage.save(new_name)
    
    new_angle = -1 * angle
    
    data_set.append([image_file_name, angle])
    data_set.append([new_name, new_angle])
    
new_csv = pd.DataFrame(data_set, columns=["image", "steering"])

new_csv.to_csv("../../data/driving_log.processed.2.csv")
    
