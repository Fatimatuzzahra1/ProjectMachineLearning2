import os

def read_files_in_leaf_folders(dataset_path):
    list_file_label = []
    ##dataset yang akan kita gunakan
    ##usahakan isinya link_foto, label
    folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    for folder in folders:
        leaf_folder_path = os.path.join(dataset_path, folder)
        files = [f for f in os.listdir(leaf_folder_path) if os.path.isfile(os.path.join(leaf_folder_path, f))]
        for file in files:
            file_path = os.path.join(leaf_folder_path, file)
            print(file_path)
            list_file_label.append([file_path, folder])

    return list_file_label
dataset_folder_path = 'Dataset'
hasil_baca = read_files_in_leaf_folders(dataset_folder_path)

import csv
csv_file_path = 'hasil_baca_file.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    for row in hasil_baca:
        csv_writer.writerow(row)
print(f"List has been written to {csv_file_path}")