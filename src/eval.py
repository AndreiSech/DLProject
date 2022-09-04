import argparse
import os
import dataprocessing as dp
import model

if __name__ == '__main__':
    """$ eval.py --help

    usage: eval.py [(-h | --help)] [(-dp | --datapath) input_dataset_dir_path]
                    [(-m | --model) (ResNet | VGG)] [(-p | --pretrained) (False | True)]

    optional arguments:
        -dp, --datapath     Input directory data with labeled pictures.
                            By default is "input\hymenoptera_data\val"
        -m, --model         Model kind: 'ResNet' to apply ResNet18 (by default), 'VGG' for VGG16
        -p, --pretrained    True or False (by default)"""

    parser = argparse.ArgumentParser(description="Classificator validation")

    parser.add_argument('-dp', '--datapath', help="Dataset path", type=str, default="..\input\hymenoptera_data\\train")
    parser.add_argument('-m', '--model', help="Model: ResNet or VGG", type=str, default='ResNet')
    parser.add_argument('-p', '--pretrained', help="Transfer learning usage", type=bool, default=False)

    # Parse the arguments
    args = parser.parse_args()

    data_path = args.datapath
    if not os.path.exists(data_path):
        parser.error("Please provide existing datapath")

    model_kind = args.model
    if model_kind not in ('ResNet', 'VGG'):
        parser.error("Model should be 'ResNet' or 'VGG' only")

    use_pretrained = args.pretrained

    print(args)
    #print(data_path, model_kind, use_pretrained)

    # Load dataset
    loader = dp.DataLoader()
    val_loader = loader.load_test_data(data_path)

    # Load saved model
    recovered_model = model.get_model(model_kind, use_pretrained, 2)
    recovered_model = model.load_model_state(recovered_model, model_kind, use_pretrained)
    recovered_model.eval()

    # Validation
    model.evaluate_accuracy(val_loader, recovered_model)
