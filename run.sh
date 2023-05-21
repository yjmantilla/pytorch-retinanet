python train.py --dataset csv --csv_train "Y:\code\hippoID\hippos_detection.csv"  --csv_classes "Y:\code\hippoID\hippos_detection_classes.csv" --epochs 3 --depth 50
python train.py --dataset csv --csv_train "Y:\code\hippoID\hippos_detection.csv"  --csv_classes "Y:\code\hippoID\hippos_detection_classes.csv" --custom_model "Y:\code\pytorch-retinanet\model_final.pt"
python train.py --dataset csv --csv_train "Y:\code\hippoID\hippos_detection.csv"  --csv_classes "Y:\code\hippoID\hippos_detection_classes.csv" --epochs 3 --depth 50
python csv_validation.py --csv_annotations_path "Y:\code\hippoID\hippos_detection_valid.csv" --model_path "Y:\code\pytorch-retinanet\model_final.pt" --class_list_path "Y:\code\hippoID\hippos_detection_classes.csv"


python -u train.py --dataset csv --csv_train "Y:\code\hippoID\hippos_detection.csv"  --csv_classes "Y:\code\hippoID\hippos_detection_classes.csv" --epochs 3 --depth 18
python -u train.py --dataset csv --csv_train "Y:\code\hippoID\hippos_id.csv"  --csv_classes "Y:\code\hippoID\hippos_id_classes.csv" --custom_model "Y:\code\pytorch-retinanet\models\2023-05-21_model-final_epochs-5_resnet-18.pt" --epochs 10
python -u visualize_single_image.py --image_dir "Y:\code\hippoID\data\hippoID-12\train" --model_path "Y:\code\pytorch-retinanet\model_final.pt" --class_list "Y:\code\hippoID\hippos_id_classes.csv" --threshold 0.3
python -u csv_validation.py --csv_annotations_path "Y:\code\hippoID\hippos_id.csv" --model_path "Y:\code\pytorch-retinanet\model_final.pt" --class_list_path "Y:\code\hippoID\hippos_id_classes.csv"
python -u visualize_max.py --image_dir "Y:\code\hippoID\data\visualize" --model_path "Y:\code\pytorch-retinanet\model_final.pt" --class_list "Y:\code\hippoID\hippos_id_classes.csv"

python visualize_single_image.py --image_dir "Y:\code\hippoID\data\hippo-pretrain-2-retina\New folder" --model_path "Y:\code\pytorch-retinanet\model_final.pt" --class_list "Y:\code\hippoID\hippos_detection_classes.csv" --threshold 0.2
python visualize_single_image.py --image_dir "Y:\code\pytorch-retinanet\models\coco_resnet_50_map_0_335_state_dict.pt" --class_list "Y:\code\hippoID\hippos_detection_classes.csv" --threshold 0.2

python visualize_single_image.py --image_dir "Y:\OneDriveFS\OneDrive - Universite de Montreal\projects\hippoID\hippoDB\sourcedata\big" --model_path "Y:\code\pytorch-retinanet\model_final.pt" --class_list "Y:\code\hippoID\hippos_detection_classes.csv" --threshold 0.2

python -u resize.py "Y:\OneDriveFS\OneDrive - Universite de Montreal\projects\hippoID\hippoDB\sourcedata" "Y:\OneDriveFS\OneDrive - Universite de Montreal\projects\hippoID\hippoDB\resized_data" 416
python -u resize.py "Y:\OneDriveFS\OneDrive - Universite de Montreal\projects\hippoID\hippoDB\sourcedata" "Y:\OneDriveFS\OneDrive - Universite de Montreal\projects\hippoID\hippoDB\jpg_data"
    parser.add_argument('--csv_annotations_path', help='Path to CSV annotations')
    parser.add_argument('--model_path', help='Path to model', type=str)
    parser.add_argument('--images_path',help='Path to images directory',type=str)
    parser.add_argument('--class_list_path',help='Path to classlist csv',type=str)
    parser.add_argument('--iou_threshold',help='IOU threshold used for evaluation',type=str, default='0.5')
