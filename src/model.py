import torch
import torch.nn as nn
import torchvision
import time
import os

from factory import ObjectFactory
import ResNet
import VGG

def get_model(model_kind, use_pretrained, num_classes):

    if (use_pretrained):
        if model_kind == 'ResNet':
            cnn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        elif model_kind == 'VGG':
            cnn = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)

        for param in cnn.parameters():
            param.requires_grad = False

        if model_kind == 'ResNet':
            num_features = cnn.fc.in_features
            cnn.fc = nn.Linear(in_features=num_features, out_features=num_classes)
        elif model_kind == 'VGG':
            cnn.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

    else:
        config = {
            'num_classes': num_classes
        }
        print(config)

        factory = ObjectFactory()
        factory.register_builder('ResNet18', ResNet.ResNet18)
        factory.register_builder('VGG16', VGG.VGG16)

        if model_kind == 'ResNet':
            cnn = factory.create('ResNet18', **config)
        elif model_kind == 'VGG':
            cnn = factory.create('VGG16', **config)

    return cnn


def evaluate_accuracy(data_iter, net):
    with torch.no_grad():
        total_records = correct_records = 0
        for index, (images, target) in enumerate(data_iter):
            output = net(images)
            _, prediction = torch.max(output, 1)
            accuracy = torch.sum(prediction == target)
            correct_records += accuracy
            total_records += images.shape[0]
        print(f'Total records {total_records} and correct records is {correct_records}')
        print(f'Total accuracy is {(correct_records / total_records) * 100}')


def train(net, train_iter, trainer, num_epochs=20):
    loss = nn.CrossEntropyLoss(reduction='sum')
    net.train()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for index, (X, y) in enumerate(train_iter):
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().item()
            n += y.shape[0]
            print("step {:d}. time since epoch: {:.3f}. Train acc: {:.3f}. Train Loss: {:.3f}".format(index, time.time() - start,
                                                                                                 (y_hat.argmax(axis=1) == y).sum().item() /
                                                                                                 y.shape[0], l.item()))
        print("Epoch {:d}, loss {:.4f}, train acc {:.3f}, time {:.1f} sec".format(epoch + 1, train_l_sum / n,
                                                                                                train_acc_sum / n,
                                                                                                time.time() - start))

def save_model_state(model, model_kind, use_pretrained, filedir = "..\models"):
    filename = "{}_{}".format('ResNet18' if model_kind == 'ResNet' else 'VGG16',
                              'pretrained' if use_pretrained else 'from_scratch')

    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    torch.save(model.state_dict(), os.path.join(filedir, filename))
    print("Model saved to ", os.path.join(filedir, filename))

def load_model_state(model, model_kind, use_pretrained, filedir = "..\models"):
    filename = "{}_{}".format('ResNet18' if model_kind == 'ResNet' else 'VGG16',
                              'pretrained' if use_pretrained else 'from_scratch')

    if not os.path.exists(os.path.join(filedir, filename)):
        print("Please provide existing file instead of {} or retrain model".format(filename))
        exit(0)
    else:
        model.load_state_dict(torch.load(os.path.join(filedir, filename)))
    return model