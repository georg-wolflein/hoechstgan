from hoechstgan.models import create_model
from hoechstgan.data import *

from omegaconf import DictConfig
from hoechstgan.util.logging import ModelLogger
from pathlib import Path


# take list of dicts of clients blocks parameters
# return dicts of averaged parameteres
def fed_avg(clients_disc1_param, clients_disc2_param, clients_gen_param):
    # initialize parameters with last model parameters for the size of the dictionnarie, it will be all overwritten
    global_disc1_param = clients_disc1_param[0]
    global_disc2_param = clients_disc2_param[0]
    global_gen_param = clients_gen_param[0]

    # avg parameters of discriminator 1
    for layer in global_disc1_param:
        for i in range(1, len(clients_disc1_param)):
            global_disc1_param[layer] = global_disc1_param[layer] + \
                clients_disc1_param[i][layer]
        global_disc1_param[layer] = global_disc1_param[layer] / \
            len(clients_disc1_param)

    # avg parameters of discriminator 2
    for layer in global_disc2_param:
        for i in range(1, len(clients_disc2_param)):
            global_disc2_param[layer] = global_disc2_param[layer] + \
                clients_disc2_param[i][layer]
        global_disc2_param[layer] = global_disc2_param[layer] / \
            len(clients_disc2_param)

    # avg parameters of generator
    for layer in global_gen_param:
        for i in range(1, len(clients_gen_param)):
            global_gen_param[layer] = global_gen_param[layer] + \
                clients_gen_param[i][layer]
        global_gen_param[layer] = global_gen_param[layer] / \
            len(clients_gen_param)

    return global_disc1_param, global_disc2_param, global_gen_param


# same as fed_avg but take weights the parameters with the length of the local database
def fed_weighted_avg(clients_disc1_param, clients_disc2_param, clients_gen_param, lendata):
    # initialize parameters with last model parameters for the size of the dictionnarie, it will be all overwritten
    global_disc1_param = clients_disc1_param[0] * lendata[0]
    global_disc2_param = clients_disc2_param[0] * lendata[0]
    global_gen_param = clients_gen_param[0] * lendata[0]

    lentotal = 0
    for i in lendata:
        lentotal += i

    # avg parameters of discriminator 1
    for layer in global_disc1_param:
        for i in range(1, len(clients_disc1_param)):
            global_disc1_param[layer] = global_disc1_param[layer] + \
                (clients_disc1_param[i][layer] * lendata[i])
        global_disc1_param[layer] = global_disc1_param[layer] / lentotal

    # avg parameters of discriminator 2
    for layer in global_disc2_param:
        for i in range(1, len(clients_disc2_param)):
            global_disc2_param[layer] = global_disc2_param[layer] + \
                (clients_disc2_param[i][layer] * lendata[i])
        global_disc2_param[layer] = global_disc2_param[layer] / lentotal

    # avg parameters of generator
    for layer in global_gen_param:
        for i in range(1, len(clients_gen_param)):
            global_gen_param[layer] = global_gen_param[layer] + \
                (clients_gen_param[i][layer] * lendata[i])
        global_gen_param[layer] = global_gen_param[layer] / lentotal

    return global_disc1_param, global_disc2_param, global_gen_param


# train all the clients and return lists of new weights
def client_training(global_disc1_param, global_disc2_param, global_gen_param, clients_gen_param, clients_disc1_param, clients_disc2_param, cfg, epoch, data, continuous_epoch, model):

    # load global parameters into client model
    model.G.netG.load_state_dict(global_gen_param)
    model.D.netD1.load_state_dict(global_disc1_param)
    model.D.netD2.load_state_dict(global_disc2_param)

    # train model
    model.set_input(data)  # preprocess data
    # Compute loss functions, get gradients, update weights
    model.optimize_parameters(epoch=continuous_epoch)

    # put client parameters in lists
    clients_gen_param.append(model.G.netG.state_dict())
    clients_disc1_param.append(model.D.netD1.state_dict())
    clients_disc2_param.append(model.D.netD2.state_dict())


# separate the database in 4 parts
def create_4_clients(cfg: DictConfig):
    # initialization
    data_loader1 = CustomDatasetDataLoader(cfg)
    data_loader2 = CustomDatasetDataLoader(cfg)
    data_loader3 = CustomDatasetDataLoader(cfg)
    data_loader4 = CustomDatasetDataLoader(cfg)

    dataset1 = data_loader1.load_data()
    dataset2 = data_loader2.load_data()
    dataset3 = data_loader3.load_data()
    dataset4 = data_loader4.load_data()

    json_path1 = []
    json_path2 = []
    json_path3 = []
    json_path4 = []

    # separate database in 4 subdatabase
    for json in dataset1.dataset.json_paths:
        if '1007' in str(json) or '1014' in str(json):
            json_path1.append(json)
        if '1025' in str(json) or '1027' in str(json):
            json_path2.append(json)
        if '1029' in str(json) or '1067' in str(json):
            json_path3.append(json)
        if '1073' in str(json) or '977' in str(json):
            json_path4.append(json)

    dataset1.dataset.json_paths = json_path1
    dataset2.dataset.json_paths = json_path2
    dataset3.dataset.json_paths = json_path3
    dataset4.dataset.json_paths = json_path4

    # print lenght of each subdatabase
    print("the client 1 has " + str(len(dataset1.dataset.json_paths))+" patches")
    print("the client 2 has " + str(len(dataset2.dataset.json_paths))+" patches")
    print("the client 3 has " + str(len(dataset3.dataset.json_paths))+" patches")
    print("the client 4 has " + str(len(dataset4.dataset.json_paths))+" patches")

    return dataset1, dataset2, dataset3, dataset4


# separate the database in 8 parts
def create_8_clients(cfg: DictConfig):
    # initialization
    data_loader1 = CustomDatasetDataLoader(cfg)
    data_loader2 = CustomDatasetDataLoader(cfg)
    data_loader3 = CustomDatasetDataLoader(cfg)
    data_loader4 = CustomDatasetDataLoader(cfg)
    data_loader5 = CustomDatasetDataLoader(cfg)
    data_loader6 = CustomDatasetDataLoader(cfg)
    data_loader7 = CustomDatasetDataLoader(cfg)
    data_loader8 = CustomDatasetDataLoader(cfg)

    dataset1 = data_loader1.load_data()
    dataset2 = data_loader2.load_data()
    dataset3 = data_loader3.load_data()
    dataset4 = data_loader4.load_data()
    dataset5 = data_loader5.load_data()
    dataset6 = data_loader6.load_data()
    dataset7 = data_loader7.load_data()
    dataset8 = data_loader8.load_data()

    json_path1 = []
    json_path2 = []
    json_path3 = []
    json_path4 = []
    json_path5 = []
    json_path6 = []
    json_path7 = []
    json_path8 = []

    # separate database in 4 subdatabase
    for json in dataset1.dataset.json_paths:
        if '1007' in str(json):
            json_path1.append(json)
        if '1014' in str(json):
            json_path2.append(json)
        if '1025' in str(json):
            json_path3.append(json)
        if '1027' in str(json):
            json_path4.append(json)
        if '1029' in str(json):
            json_path5.append(json)
        if '1067' in str(json):
            json_path6.append(json)
        if '1073' in str(json):
            json_path7.append(json)
        if '977' in str(json):
            json_path8.append(json)

    dataset1.dataset.json_paths = json_path1
    dataset2.dataset.json_paths = json_path2
    dataset3.dataset.json_paths = json_path3
    dataset4.dataset.json_paths = json_path4
    dataset5.dataset.json_paths = json_path5
    dataset6.dataset.json_paths = json_path6
    dataset7.dataset.json_paths = json_path7
    dataset8.dataset.json_paths = json_path8

    # print lenght of each subdatabase
    print("the client 1 has " + str(len(dataset1.dataset.json_paths))+" patches")
    print("the client 2 has " + str(len(dataset2.dataset.json_paths))+" patches")
    print("the client 3 has " + str(len(dataset3.dataset.json_paths))+" patches")
    print("the client 4 has " + str(len(dataset4.dataset.json_paths))+" patches")
    print("the client 5 has " + str(len(dataset5.dataset.json_paths))+" patches")
    print("the client 6 has " + str(len(dataset6.dataset.json_paths))+" patches")
    print("the client 7 has " + str(len(dataset7.dataset.json_paths))+" patches")
    print("the client 8 has " + str(len(dataset8.dataset.json_paths))+" patches")

    return dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8


def plot_test():
    pass
