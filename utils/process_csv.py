import pandas as pd

STEERING_COEFFICIENT = 0.225

csv_file = pd.read_csv("../../data/driving_log.csv")

steering_angle = csv_file["steering"].values
c_image = csv_file["center"].values
l_image = csv_file["left"].values
r_image = csv_file["right"].values


data_set = []

count = 0

for i in range(len(c_image)):
    count+=1
    print("Number of data: \t{}\r".format(count), end="")
    data_set.append([c_image[i], steering_angle[i]])
    
for i in range(len(l_image)):
    count+=1
    print("Number of data: \t{}\r".format(count), end="")
    data_set.append([l_image[i], steering_angle[i] - STEERING_COEFFICIENT])
    
for i in range(len(r_image)):
    count+=1
    print("Number of data: \t{}\r".format(count), end="")
    data_set.append([r_image[i], steering_angle[i] + STEERING_COEFFICIENT])
    
print()
    
new_csv = pd.DataFrame(data_set, columns=["image", "steering"])

new_csv.to_csv("../../data/driving_log.processed.1.csv")

    
    
    
