# Remote directories
# tensor dataset
obverse_train_dir = './data/ccc_tensors/train/obverses'
obverse_train_csv = './data/ccc_tensors/train/train_labels.csv'
obverse_validate_dir = './data/ccc_tensors/val/obverses'
obverse_validate_csv = './data/ccc_tensors/val/val_labels.csv'
obverse_test_dir = './data/ccc_tensors/test/obverses'
obverse_test_csv = './data/ccc_tensors/test/test_labels.csv'

obverse_train = './data/ccc_tensors/obverse/train_dataset.csv'
obverse_val = './data/ccc_tensors/obverse/val_dataset.csv'
obverse_test = './data/ccc_tensors/obverse/test_dataset.csv'

obverse_tensors = './data/ccc_tensors/obverse/data/'

reverse_train = './data/ccc_tensors/reverse/train_dataset.csv'
reverse_val = './data/ccc_tensors/reverse/val_dataset.csv'
reverse_test = './data/ccc_tensors/reverse/test_dataset.csv'

reverse_tensors = './data/ccc_tensors/reverse/data/'

combined_train = './data/ccc_tensors/combined/train_dataset.csv'
combined_val = './data/ccc_tensors/combined/val_dataset.csv'
combined_test = './data/ccc_tensors/combined/test_dataset.csv'

combined_tensors = './data/ccc_tensors/combined/data/'


reverse_train_dir = '../../data/processed/ccc_images_cropped_final/train/Obverses/'
reverse_train_csv = '../../data/processed/ccc_images_cropped_final/train/obverse_labels.csv'
reverse_validate_dir = '../../data/processed/ccc_images_cropped_final/validate/Reverses/'
reverse_validate_csv = '../../data/processed/ccc_images_cropped_final/validate/reverse_labels.csv'
reverse_test_dir = '../../data/processed/ccc_images_cropped_final/test/Reverses/'
reverse_test_csv = '../../data/processed/ccc_images_cropped_final/test/reverse_labels.csv'

sam_checkpoint = './content/weights/sam_vit_b_01ec64.pth'

# SAM
sam_bce_out = 'runs/sam/bce'
sam_contrastive_out = 'runs/sam/contrastive'
sam_triplet_out = 'runs/sam/triplet'