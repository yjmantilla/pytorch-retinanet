import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


#parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

model_path =r"Y:\code\pytorch-retinanet\model_final.pt"
csv_annotations_path = r"Y:\code\hippoID\hippos_detection_valid.csv" 
class_list_path =r"Y:\code\hippoID\hippos_detection_classes.csv"
iou_threshold='0.1'
dataset_val = CSVDataset(csv_annotations_path,class_list_path,transform=transforms.Compose([Normalizer(), Resizer()]))
# Create the model
#retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
retinanet=torch.load(model_path)

use_gpu = True

if use_gpu:
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()

# I think the loaded model is already data parallel
if not isinstance(retinanet,torch.nn.parallel.DataParallel):
    if torch.cuda.is_available():
        retinanet.load_state_dict(torch.load(model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(model_path))
        retinanet = torch.nn.DataParallel(retinanet)

retinanet.training = False
retinanet.eval()
retinanet.module.freeze_bn()
#retinanet.freeze_bn()
print(csv_eval.evaluate(dataset_val, retinanet,iou_threshold=float(iou_threshold)))





#retinanet=torch.load(model_path)