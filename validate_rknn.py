import torch
from rknn.api import RKNN
import cv2
import numpy as np
from cifar import load_torchscript_model, evaluate_model, prepare_dataloader
from tqdm import tqdm
import torchvision


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

mean = [m*255 for m in mean]
std = [s*255 for s in std]


def evaluate_model_rknn(test_loader, device, criterion=None):

    running_loss = 0
    running_corrects = 0

    for inputs, labels in tqdm(test_loader):

        inputs = inputs.to(device).numpy()
        labels = labels.to(device).numpy()
        inputs = np.transpose(inputs, (0, 2, 3, 1))

        outputs = rknn.inference(inputs=[inputs[0]])
        # np.array(outputs[0][0])
        _, preds = torch.max(torch.tensor(outputs[0]), 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.shape[0]
        running_corrects += torch.sum(preds == torch.tensor(labels))

    eval_loss = running_loss / len(test_loader)
    eval_accuracy = running_corrects / len(test_loader)

    return eval_loss, eval_accuracy


def evaluate_model_torch(model, test_loader, device, criterion=None):

    running_loss = 0
    running_corrects = 0

    for i, inputs in tqdm(enumerate(test_loader.data)):

        labels = test_loader.targets[i]
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        for j in range(3):
            inputs[:, :, j] = (inputs[:, :, j] - mean[j]) / std[j]

        inputs = np.transpose(inputs, (2, 0, 1))

        inputs = torch.tensor(inputs).to(device)
        outputs = model(inputs[np.newaxis, :])

        # np.array(outputs[0][0])
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.shape[0]
        running_corrects += torch.sum(preds == labels)

    eval_loss = running_loss / len(test_loader.data)
    eval_accuracy = running_corrects / len(test_loader.data)

    return eval_loss, eval_accuracy


def prepare_dataloader_rknn(num_workers=8,
                       eval_batch_size=256):


    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    test_set = torchvision.datasets.CIFAR10(root="data",
                                            train=False,
                                            download=True,
                                            )


    test_sampler = torch.utils.data.SequentialSampler(test_set)

    # test_loader = torch.utils.data.DataLoader(dataset=test_set,
    #                                           batch_size=eval_batch_size,
    #                                           sampler=test_sampler,
    #                                           num_workers=num_workers)

    return test_set


if __name__ == '__main__':
    rknn = RKNN(verbose=False, verbose_file='./verbose_log.txt')
    quantized_model_filepath = './saved_models/resnet18_quantized_cifar10.pt'
    cpu_device = torch.device("cpu:0")

    # pre-process config
    print('--> Set config model')
    rknn.config(quantize_input_node=False,
                merge_dequant_layer_and_output_node=False,
                # mean_values=[mean],
                # std_values=[std],
                # optimization_level=0,
                # quantized_dtype='dynamic_fixed_point-i8',
                target_platform='rv1126',
                # model_data_format='nchw',
                )
    print('done')

    # Load Pytorch model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=quantized_model_filepath, input_size_list=[[3, 32, 32]])
    if ret != 0:
        print('Load Pytorch model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset='dataset.txt', pre_compile=True)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./saved_models/resnet18_quant.rknn')
    if ret != 0:
        print('Export resnet18_quant.rknn failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    # ret = rknn.init_runtime()
    ret = rknn.init_runtime(target='rv1126', device_id='fa647528590c7546')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    _, test_loader = prepare_dataloader(num_workers=8,
                                        eval_batch_size=256)
    _, rknn_loader = prepare_dataloader(num_workers=8, eval_batch_size=1)

    quantized_jit_model = load_torchscript_model(
        model_filepath=quantized_model_filepath, device=cpu_device)

    _, int8_rknn_accuracy = evaluate_model_rknn(rknn_loader, cpu_device)

    _, int8_eval_accuracy = evaluate_model(model=quantized_jit_model,
                                           test_loader=test_loader,
                                           device=cpu_device,
                                           criterion=None)

    print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))
    print("INT8 RKNN evaluation accuracy: {:.3f}".format(int8_rknn_accuracy))