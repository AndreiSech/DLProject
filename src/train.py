import argparse
import os
import dataprocessing as dp
import model
import torch.optim

if __name__ == '__main__':
    """$ train.py --help
    
    usage: train.py [(-h | --help)] [(-dp | --datapath) input_dataset_dir_path] [(-s | --save_to) output_dir]
                    [(-m | --model) (ResNet | VGG)] [(-p | --pretrained) (False | True)] [(-a | --augmentation) (False | True)]
    
    optional arguments:
        -dp, --datapath     Input directory data with labeled pictures.
                            By default is "input\hymenoptera_data\train"                            
        -s, --save_to       Output directory for trained model
                            By default is "models"
        -m, --model         Model kind: 'ResNet' to apply ResNet18 (by default), 'VGG' for VGG16
        -p, --pretrained    True or False (by default)
		-a, --augmentation	True (by default) or False"""

    parser = argparse.ArgumentParser(description="Images classificator")

    parser.add_argument('-dp', '--datapath', help="Dataset path", type=str, default="..\input\hymenoptera_data\\train")
    parser.add_argument('-s', '--save_to', help="Dir to save trained model", type=str, default='..\models')
    parser.add_argument('-m', '--model', help="Model: ResNet or VGG", type=str, default='ResNet')
    parser.add_argument('-p', '--pretrained', help="Transfer learning usage", type=bool, default=False)
    parser.add_argument('-a', '--augmentation', help="Data augmentation", type=bool, default=True)

    # Parse the arguments
    args = parser.parse_args()

    data_path = args.datapath
    if not os.path.exists(data_path):
        parser.error("Please provide existing datapath")

    dir_to_save = args.save_to
    if not os.path.isdir(dir_to_save):
        os.makedirs(dir_to_save)
        print("Directory {} was created".format(dir_to_save))

    model_kind = args.model
    if model_kind not in ('ResNet', 'VGG'):
        parser.error("Model should be 'ResNet' or 'VGG' only")

    use_pretrained = args.pretrained
    use_augmentation = args.augmentation

    print(args)
    #print(data_path, dir_to_save, model_kind, use_pretrained, use_augmentation)

    # Load dataset
    loader = dp.DataLoader()
    train_loader = loader.load_train_data(data_path, use_augmentation)

    # Model training
    cnn = model.get_model(model_kind, use_pretrained, 2)
    trainer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
    model.train(cnn, train_loader, trainer)

    # Save trained model
    model.save_model_state(cnn, model_kind, use_pretrained, dir_to_save)