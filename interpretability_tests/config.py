import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# makign class config 
class config:
    # setting the batch size;
    batch_size = 512
    shuffle = True
    # setting the number of workers for the dataloader = number of cpu threads// 2
    num_workers = torch.get_num_threads() // 2
    # pin memory None as we are not training the model to keep the data on the SMs of the GPU
    pin_memory = False
    persistent_workers = True
    prefetch_factor = 4
    # setting the device
    device = device
    fig_save_folderpath = '/home/mahesh/ayan/Time-Series-Library-main/scripts/imputation_analysis/Deep-Fakes/interpretablility_results/alexnet/'
    test_dir = "/home/mahesh/ayan/Time-Series-Library-main/scripts/imputation_analysis/Deep-Fakes/data/real-vs-fake/valid/"
    model_chkpt = "/home/mahesh/ayan/Time-Series-Library-main/scripts/imputation_analysis/Deep-Fakes/checkpoints_2/alexnet/best_model_optimizer_1.pth"