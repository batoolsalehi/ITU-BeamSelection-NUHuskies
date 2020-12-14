from PIL import Image
import os 
from tqdm import tqdm


def show_all_files_in_directory(input_path):
    'This function reads the path of all files in directory input_path'
    files_list=[]
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".png"):
               files_list.append(os.path.join(path, file))
    return files_list


truck = show_all_files_in_directory('/Users/maryam/Desktop/camras/samples/truck') 
bus = show_all_files_in_directory('/Users/maryam/Desktop/camras/samples/bus')
car = show_all_files_in_directory('/Users/maryam/Desktop/camras/samples/car')
background = show_all_files_in_directory('/Users/maryam/Desktop/camras/samples/background')


save_path = '/Users/maryam/Desktop/camras/crops/car/'
window = 40
stride = 5 


count = 0
for sample in car:

    img=Image.open(sample)

    row_step = int((img.size[1]-window)/stride+1)
    col_step = int((img.size[0]-window)/stride+1)

    # If we choose not very large window and stride, it will be equal to stride
    row_jump = img.size[1]/row_step
    col_jump = img.size[0]/col_step
    all_strides = [(j*col_jump,k*row_jump) for j in range(col_step) for k in range(row_step)] 

    for strides_to_genearte in tqdm(all_strides):

        area = (strides_to_genearte[0], strides_to_genearte[1], strides_to_genearte[0]+window, strides_to_genearte[1]+window)
        crop = img.crop(area)

        if crop.size != (40,40):
            crop = crop.resize((40,40))

        crop.save(save_path+str(count)+'.png')
        count +=1

    print('{} crops has been created'.format(count))
