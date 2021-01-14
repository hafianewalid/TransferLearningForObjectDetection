import numpy as np
import Data
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

dataset_dir = "/opt/Datasets/Pascal-VOC2012/"
download = False

image_transform_params = {'image_mode': 'none'}


target_transform_params = {'target_mode': 'preprocessed'}


image_transform = transforms.ToTensor()



train_dataset, valid_dataset, test_dataset = Data.pascal_dataset(
        dataset_dir             = dataset_dir,
        image_transform_params  = image_transform_params,
        transform               = image_transform,
        target_transform_params = target_transform_params,
        download                = download)

print(" train dataset size : {} \n valid dataset size : {}\n test dataset size {}"
      .format(len(train_dataset),len(valid_dataset),len(test_dataset)))

print(train_dataset[0])
print(train_dataset[0][1]["annotation"]['size'])
print(train_dataset[0][1]["annotation"]['object'][0]['name'])

heights,widths,objects,bbox={},{},{},{}
for item in train_dataset:
     for obj in item[1]['annotation']['object']:
         h=obj['bndbox']['ymax']-obj['bndbox']['ymin']
         w=obj['bndbox']['xmax']-obj['bndbox']['xmin']
         obj_name=obj['name']

         if h not in heights :
             heights[h]=1
         else:
             heights[h]+=1
         if w not in widths :
             widths[w]=1
         else:
             widths[w]+=1

         if obj_name not in objects :
             objects[obj_name]=1
         else:
             objects[obj_name]+=1

         if obj_name not in bbox :
             bbox[obj_name]=[w*h]
         else:
             bbox[obj_name].append(w*h)


plt.style.use('bmh')
plt.subplot()
plt.bar(list(heights.keys()),heights.values(),width=10, color='r')
plt.bar(np.array(list(widths.keys()))+10,widths.values(),width=10, color='g')
plt.legend(handles=[mpatches.Patch(color='r',label='heights'),mpatches.Patch(color='g',label='widths')])
plt.title("Width and height histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.savefig("WH_histo.png")
plt.show()

plt.bar(list(objects.keys()),list(objects.values()),1,ec='w')
plt.xticks(rotation=70)
plt.title("Objects distribution")
plt.xlabel("class")
plt.ylabel("Frequency")
plt.savefig("distributions.png")
plt.show()

box_dist=[bbox[i] for i in bbox.keys()]
plt.boxplot(box_dist,labels=bbox.keys(),patch_artist=True)
plt.xticks(rotation=70)
plt.title("Bbox size variation")
plt.ylabel("Bbox size")
plt.xlabel("Class")
plt.savefig("box_size.png")
plt.show()





