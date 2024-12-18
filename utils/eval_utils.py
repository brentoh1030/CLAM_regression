import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index  # For regression tasks


def initiate_model(args, ckpt_path, device='cuda'):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)

    if args.task_type == 'classification':
        patient_results, test_error, auc_score, df, _ = summary(model, loader, args)
        print('test_error: ', test_error)
        print('auc: ', auc_score)
        return model, patient_results, test_error, auc_score, df
    else:  # Regression
        patient_results, risk_scores, event_times, censorships, c_index, df = summary(model, loader, args)
        print('C-index: ', c_index)
        return model, patient_results, risk_scores, event_times, censorships, df

def summary(model, loader, args):
    model.eval()

    if args.task_type == 'classification':
        acc_logger = Accuracy_Logger(n_classes=args.n_classes)
        test_error = 0.
        all_probs = np.zeros((len(loader), args.n_classes))
        all_labels = np.zeros(len(loader))
        all_preds = np.zeros(len(loader))
    else:  # Regression
        risk_scores = []
        event_times = []
        censorships = []

    test_loss = 0.

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    for batch_idx, batch_data in enumerate(loader):
        if args.task_type == 'classification':
            data, label = batch_data
            label = label.to(device)
        else:  # Regression
            data, event_time, censorship = batch_data
            event_time, censorship = event_time.to(device), censorship.to(device)

        data = data.to(device)
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            if args.task_type == 'classification':
                logits, Y_prob, Y_hat, _, _ = model(data)
                acc_logger.log(Y_hat, label)

                probs = Y_prob.cpu().numpy()
                all_probs[batch_idx] = probs
                all_labels[batch_idx] = label.item()
                all_preds[batch_idx] = Y_hat.item()
                error = calculate_error(Y_hat, label)
                test_error += error

                patient_results.update({
                    slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}
                })
                
            else:  # Regression
                risk_score, _, _ = model(data)
                risk_scores.extend(risk_score.cpu().numpy())
                event_times.extend(event_time.cpu().numpy())
                censorships.extend(censorship.cpu().numpy())

                patient_results.update({
                    slide_id: {
                        'slide_id': np.array(slide_id),
                        'risk_score': risk_score.cpu().numpy(),
                        'event_time': event_time.item(),
                        'censorship': censorship.item()
                    }
                })

    del data
    if args.task_type == 'classification':
        test_error /= len(loader)

        aucs = []
        if len(np.unique(all_labels)) == 1:
            auc_score = -1

        else: 
            if args.n_classes == 2:
                auc_score = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                for class_idx in range(args.n_classes):
                    if class_idx in all_labels:
                        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                        aucs.append(auc(fpr, tpr))
                    else:
                        aucs.append(float('nan'))
                if args.micro_average:
                    binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                    fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                    auc_score = auc(fpr, tpr)
                else:
                    auc_score = np.nanmean(np.array(aucs))

        results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
        for c in range(args.n_classes):
            results_dict.update({'p_{}'.format(c): all_probs[:,c]})
        df = pd.DataFrame(results_dict)
        return patient_results, test_error, auc_score, df, acc_logger
    else:
        c_index = concordance_index(
            event_times=np.array(event_times),
            predicted_scores=np.array(risk_scores),
            event_observed=np.array(censorships)
        )

        df = pd.DataFrame({
            'slide_id': slide_ids,
            'risk_score': risk_scores,
            'event_time': event_times,
            'censorship': censorships
        })

        return patient_results, risk_scores, event_times, censorships, c_index, df
