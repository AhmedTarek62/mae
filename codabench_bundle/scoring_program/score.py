#!/usr/bin/env python

# Scoring program for the AutoML challenge
# Isabelle Guyon and Arthur Pesah, ChaLearn, August 2014-November 2016

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

# Some libraries and options
import os
from sys import argv

import yaml
import torch
import importlib, sys
from segmentation.dataset import TaskDataset
from segmentation.finetuning_engine import evaluate
# Default I/O directories:

# Debug flag 0: no debug, 1: show all scores, 2: also show version amd listing of dir
debug_mode = 1

# Constant used for a missing score
missing_score = -0.999999

# Version number
scoring_version = 1.0

dataset_path = "/app/testsets"
submission_path = "/app/input/res"
# =============================== MAIN ========================================

if __name__ == "__main__":
    
    # Load the model class
    model_class_path = os.path.join(submission_path, "model.py")
    spec = importlib.util.spec_from_file_location('model', model_class_path)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules['model'] = model_module
    spec.loader.exec_module(model_module)
    print('> Successfully loaded the model definition!')

    function_name = 'vit_small_patch16' # TODO: This should be a user input as well
    TaskModel = getattr(model_module, function_name)
    model_obj = TaskModel()
    print('> Successfully loaded the model object!')

    # Load the model checkpoint
    checkpoint_path = os.path.join(submission_path, "checkpoint.pth")
    msg = model_obj.load_model_checkpoint(checkpoint_path=checkpoint_path)
    print('> Successfully loaded the model checkpoint!')

    # model = torch.load(model_path, map_location=torch.device('cpu'))
    # print(model)
    model_obj.eval()

    # Dataset
    # print(f"For positioning task found {len(list(os.listdir(os.path.join(dataset_path, 'positioning/test'))))} files!")
    # print(f"For sensing task found {len(list(os.listdir(os.path.join(dataset_path, 'sensing/test'))))} files!")
    # print(f"For segmentation task found {len(list(os.listdir(os.path.join(dataset_path, 'segmentation/Test/LTE_NR'))))} files!")
    # print(f"For signal_identification task found {len(list(os.listdir(os.path.join(dataset_path, 'signal_identification/test'))))} files!")
    # print(f"For channel_estimation task found {len(list(os.listdir(os.path.join(dataset_path, 'channel_estimation/val'))))} files!")
    
    data_for_testing = "segmentation" # TODO: This changes from a task to another!
    segmentation_data_path = os.path.join(dataset_path, data_for_testing)
    dataset_val = TaskDataset(segmentation_data_path, split="val")
    print(f'> Successfully loaded the {data_for_testing} dataset!')

    batch_size = 64
    num_workers = 10
    pin_mem = True

    ## 2. Create the dataloader (Validation)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=False
        )
    
    
    n_parameters = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
    print('Number of params (M): %.2f' % (n_parameters / 1.e6))
    num_parameters = n_parameters / 1.e6

    criterion = torch.nn.CrossEntropyLoss()
    val_stats = evaluate(data_loader_val, model_obj, criterion, "cpu")

    output_dir = '/app/output'
    os.makedirs(output_dir, exist_ok=True)

    score_file = open(os.path.join(output_dir, 'scores.txt'), 'w')

    score_file.write("pred_score: %0.3f\n" % val_stats["avg_acc"])
    # Write score corresponding to selected task and metric to the output file
    for i in range(4):
        score_file.write(f"elsayed_score{i}" + ": %0.12f\n" % 10)

    # End loop for solution_file in solution_names

    # Read the execution time and add it to the scores:
    try:
        # metadata = yaml.load(open(os.path.join(input_dir, 'res', 'metadata'), 'r'))
        score_file.write("Duration: %0.6f\n" % 20)
    except:
        score_file.write("Duration: 0\n")

    score_file.write(f"Parameters: {n_parameters}")
    

    score_file.close()

    

 