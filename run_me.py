import os

import numpy as np

import torch
from torch.autograd import Variable

from preparing_data import read_pickled_data_gzip
from training import training_model
from testing import measure_active_units_model, measure_marginal_LL_model
from generation import generation_model

from datetime import datetime

### Saving the progress in a logfile
def update_log_file(filename, text):
    with open(filename, "a") as fl:
        fl.write(text)

### This function train, testing and generating based on the given model and parameters
def run_me(USE_CUDA, layers, n_pseudo_inputs, x_dim, h_dim, z_dim, n_batch, eta, epochs, early_stopping,
           warm_up, device, train_data, valid_data, test_data, He_bool, weighted_vp_bool):

    text = "\n{}\nmodel_name:{}\t\tLayers:{}\t\tpseudo_inputs:{}\t\tGPU:{}\t\teta:{}\t\twarm_up:{}\n".format(
        str(datetime.now()), model_name, layers, n_pseudo_inputs, USE_CUDA, eta, warm_up)
    text += "early_stopping:{}\t\tbatch_size:{}\t\tvectorized_input_dim:{}\t\tlatent_dim:{}\t\thidden_units:{}\n".format(
        early_stopping, n_batch, x_dim, z_dim, h_dim)
    text += "epochs:{}\t\tHe_initialization:{}\t\tweighted_VP:{}\n".format(epochs, He_bool, bool(weighted_vp_bool))
    text += "\nStart Training at {}".format(str(datetime.now()))
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

    ############################## TRAINING ##############################
    model, train_loss_lst, valid_loss_lst = training_model(model_name, train_data, valid_data, device, layers,
                                                           x_dim, h_dim, z_dim, n_batch, eta, epochs,
                                                           n_pseudo_inputs, early_stopping, warm_up, weighted_vp_bool,
                                                           He_bool, plot_show_bool=False)
    ############################## Training ##############################

    text = "\nEnd of training at {}\tbest_validation_loss:{}\tbest_epoch:{}\n".format(
        str(datetime.now()), np.min(valid_loss_lst[warm_up:]), len(valid_loss_lst) - early_stopping)
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

    text = "\nStart generating digits by sampling from the prior (time:{})".format(str(datetime.now()))
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

    ############################## GENERATING ##############################
    generated_samples = generation_model(model, model_name)
    ############################## Generating ##############################

    text = "\nEnd of generating digits by sampling from the prior (time:{})\n".format(str(datetime.now()))
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

    text = "\nStart measureing the active units for the validation dataset at {}\n".format(str(datetime.now()))
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

    X = Variable(torch.tensor(valid_data).float().to(device))

    ############################## ACTIVE UNITS (VALIDATION) ##############################
    val_active_units_per_layer = measure_active_units_model(X, model)
    ############################## Active Units (Validation) ##############################

    text = "\nEnd of measuring the active units for the validation dataset at {}".format(str(datetime.now()))
    text += "\nActive units per layer {}\n".format(str(val_active_units_per_layer)[1:-1])
    text += "\nStart measureing the active units for the testing dataset at {}\n".format(str(datetime.now()))
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

    X = Variable(torch.tensor(test_data).float().to(device))

    ############################## ACTIVE UNITS (TESTING) ##############################
    test_active_units_per_layer = measure_active_units_model(X, model)
    ############################## Active Units (Testing) ##############################

    text = "\nEnd of measuring the active units for the testing dataset at {}".format(str(datetime.now()))
    text += "\nActive units per layer {}\n".format(str(test_active_units_per_layer)[1:-1])

    text += '\nStart measuring the averaged marginal log-likelihood for the validation dataset at {}'.format(
        str(datetime.now()))
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

    X = Variable(torch.tensor(valid_data).float().to(device))

    ############################## MARGINAL LL (VALIDATION) ##############################
    val_avg_marginal_LL = measure_marginal_LL_model(X, model, model_name, "validation")
    ############################## Marginal LL (Validation) ##############################

    text = "\nEnd of measuring the averaged marginal log-likelihood for the validation dataset at {}\n".format(
        str(datetime.now()))
    text += "The averaged marginal log-likelihood for the validation dataset is {}\n".format(
        val_avg_marginal_LL)

    text += '\nStart measuring the averaged marginal log-likelihood for the testing dataset at {}'.format(
        str(datetime.now()))
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

    X = Variable(torch.tensor(test_data).float().to(device))

    ############################## MARGINAL LL (TESTING) ##############################
    test_avg_marginal_LL = measure_marginal_LL_model(X, model, model_name, "test")
    ############################## Marginal LL (Testing) ##############################

    text = "\nEnd of measuring the averaged marginal log-likelihood for the testing dataset at {}\n".format(
        str(datetime.now()))
    text += "The averaged marginal log-likelihood for the testing dataset is {}\n\n".format(
        test_avg_marginal_LL)
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)


if __name__ == "__main__":
    ### Prepraing data
    download_bool = False  # True if you need to download all the data (especially for the first time)
    if download_bool:
        download_binarirzed_mnist()  # Downloading data
        read_original_data_and_write_pickled_data_gzip()  # Creating the numpy arrays and save the arrays to pickles
    train_data, valid_data, test_data = read_pickled_data_gzip()  # Read the data from the saved (compressed) pickles
    print("\nShapes for train:{} valid:{} test:{}\n".format(train_data.shape, valid_data.shape, test_data.shape))

    ### GPU or CPU
    print("Using Torch Version:{}".format(torch.__version__))
    USE_CUDA = torch.cuda.is_available()  # Do you have a GPU or CPU
    device = torch.device("cuda" if USE_CUDA else "cpu")  #
    print("Device in use {}".format(device))
    if USE_CUDA:  # If you have GPU, then how many GPUs do you have
        num_of_gpu = torch.cuda.device_count()  # How many GPUs?
        print("Available GPUs:{}".format(num_of_gpu))
        for n_gpu in range(num_of_gpu):
            print("GPU:{} is a '{}'".format(n_gpu, torch.cuda.get_device_name(n_gpu)))  # Which GPU do you have

    ### Configuration of the model
    layers = 2
    N = train_data.shape[0]  # Number of training data
    x_dim = train_data.shape[1]
    h_dim = 300
    z_dim = 40
    n_batch = 100
    eta = 1e-4  # Learning rate
    epochs = int(1e4)
    n_pseudo_inputs = 500  # Number of pseudo inputs (500)
    early_stopping = 50  # How many epochs should I wait? (if None then I do not apply early stopping)
    warm_up = 0
    He_bool = True
    weighted_vp_bool = True

    for layers in range(1,5):

        ### Create the name of the model based on the given configurations
        model_name = ("VAE_" if layers == 1 else "HVAE_L_{}_".format(layers)) + (
            "SG_" if n_pseudo_inputs == 0 else "VP_") + ("WG_" if bool(weighted_vp_bool) else "") + "He_new"

        ### Create a directory to save the results
        if not os.path.exists(os.path.join("Results", model_name)):
            os.makedirs(os.path.join("Results", model_name))
        else:
            print("Skipping the model:{}".format(model_name))
            continue

        run_me(USE_CUDA, layers, n_pseudo_inputs, x_dim, h_dim, z_dim, n_batch, eta, epochs, early_stopping,
               warm_up, device, train_data, valid_data, test_data, He_bool, weighted_vp_bool)
