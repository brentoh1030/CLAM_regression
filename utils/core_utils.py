import numpy as np
import torch
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from lifelines.utils import concordance_index

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cox_partial_likelihood_loss(risk_scores, times, events):
    idx = torch.argsort(times, descending=True)
    sorted_events = events[idx]
    sorted_risk = risk_scores[idx].view(-1)

    max_risk = torch.max(sorted_risk)
    exp_risk = torch.exp(sorted_risk - max_risk)
    cum_sum = torch.cumsum(exp_risk, dim=0)
    log_cum_sum = torch.log(cum_sum) + max_risk
    
    event_mask = (sorted_events == 1)
    loss = -(torch.mean(sorted_risk[event_mask] - log_cum_sum[event_mask]))
    #print("Sorted Risk Scores:", sorted_risk)
    #print("Exp(Risk):", exp_risk)
    #print("Cumulative Sum of Exp(Risk):", cum_sum)
    #print("Log Cumulative Sum:", log_cum_sum)
    #print("Event Mask:", event_mask)
    #print("Loss Contributions:", sorted_risk[event_mask] - log_cum_sum[event_mask])

    return loss

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.task_type == 'classification':
        if args.bag_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
            if device.type == 'cuda':
                loss_fn = loss_fn.cuda()
        else:
            loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = cox_partial_likelihood_loss
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim,
                  "task_type": args.task_type}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))


    if args.task_type == 'classification':
        _, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
        print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

        results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

        for i in range(args.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

            if writer:
                writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

        if writer:
            writer.add_scalar('final/val_error', val_error, 0)
            writer.add_scalar('final/val_auc', val_auc, 0)
            writer.add_scalar('final/test_error', test_error, 0)
            writer.add_scalar('final/test_auc', test_auc, 0)
            writer.close()
        return results_dict, test_auc, val_auc, 1-test_error, 1-val_error

    else:
        _, val_c_index = summary(model, val_loader)
        _, test_c_index = summary(model, test_loader)
        print('Validation C-index: {:.4f}'.format(val_c_index))
        print('Test C-index: {:.4f}'.format(test_c_index))
        if writer:
            writer.add_scalar('final/val_c_index', val_c_index, 0)
            writer.add_scalar('final/test_c_index', test_c_index, 0)
            writer.close()
        return test_c_index, val_c_index

def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    model.train()
    accumulation_steps=16
    optimizer.zero_grad()
    # Initialize loggers for classification or placeholders for regression
    if model.task_type == 'classification':
        acc_logger = Accuracy_Logger(n_classes=n_classes)
        inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    accumulated_risk_scores = []
    accumulated_event_times = []
    accumulated_censorship = []
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, batch_data in enumerate(loader):
        #print(f"Batch {batch_idx}: img shape = {batch_data[0].shape}, event_time shape = {batch_data[1].shape}, censorship shape = {batch_data[2].shape}")
        if model.task_type == 'classification':
            data, label = batch_data
            label = label.to(device)
        elif model.task_type == 'regression':
            data, event_time, censorship = batch_data
            if not torch.isfinite(batch_data[0]).all():
                print("NaN or Inf found in input features")
            if not torch.isfinite(batch_data[1]).all():
                print("NaN or Inf found in event_time")
            if not torch.isfinite(batch_data[2]).all():
                print("NaN or Inf found in censorship")
            event_time, censorship = event_time.to(device), censorship.to(device)

        data = data.to(device)

        # Model forward pass
        if model.task_type == 'classification':
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
        else:  # Regression
            risk_scores, _, instance_dict = model(data, label=None, instance_eval=True)
            accumulated_risk_scores.append(risk_scores)
            accumulated_event_times.append(event_time)
            accumulated_censorship.append(censorship)
            #print("Risk scores before loss:", risk_scores)
        
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            
            # Concatenate accumulated data
            risk_scores_batch = torch.cat(accumulated_risk_scores, dim=0)
            event_times_batch = torch.cat(accumulated_event_times, dim=0)
            censorship_batch = torch.cat(accumulated_censorship, dim=0)

            # Compute Cox loss
            loss = loss_fn(risk_scores_batch, event_times_batch, censorship_batch) / accumulation_steps
            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad()  # Reset gradients

            # Clear accumulated data
            accumulated_risk_scores = []
            accumulated_event_times = []
            accumulated_censorship = []
            train_loss += loss.item() * accumulation_steps  # Undo scaling for logging
            print(f"Batch {batch_idx + 1}/{len(loader)}, Avg Loss: {train_loss / (batch_idx + 1):.4f}")

        # Compute losses
        #if model.task_type == 'classification':
        #    loss = loss_fn(logits, label)
        #    loss_value = loss.item()
        #    acc_logger.log(Y_hat, label)

            # Classification error calculation
         #   error = calculate_error(Y_hat, label)
         #   train_error += error

        #else:  # Regression
        #    loss = loss_fn(risk_scores, event_time, censorship) / accumulation_steps
            #loss_value = loss.item()

        # Instance-level loss (common for both tasks if instance loss is enabled)
        #if instance_dict:
        #    instance_loss = instance_dict['instance_loss']
        #    inst_count += 1
        #    instance_loss_value = instance_loss.item()
        #    train_inst_loss += instance_loss_value
        #    total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss

        #    if model.task_type == 'classification':
        #        inst_preds = instance_dict['inst_preds']
        #        inst_labels = instance_dict['inst_labels']
        #        inst_logger.log_batch(inst_preds, inst_labels)
        #else:
        #    total_loss = loss

        #train_loss += loss_value

    # calculate loss and error for epoch
    train_loss /= len(loader)
    if inst_count > 0:
        train_inst_loss /= inst_count

    print(f"\nEpoch {epoch}: Train Loss: {train_loss:.4f}")
    if model.task_type == 'classification':
        train_error /= len(loader)
        print('train_error: {:.4f}'.format(train_error))

        # Log classification accuracy per class
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
            if writer and acc is not None:
                writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    # Log metrics to TensorBoard
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        if inst_count > 0:
            writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

        if model.task_type == 'classification':
            writer.add_scalar('train/error', train_error, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    model.eval()
    if model.task_type == 'classification':
        acc_logger = Accuracy_Logger(n_classes=n_classes)
        inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    accumulation_steps = 16

    if model.task_type == 'classification':
        prob = np.zeros((len(loader), n_classes))
        labels = np.zeros(len(loader))
    else:
        accumulated_risk_scores = []
        accumulated_event_times = []
        accumulated_censorships = []

    sample_size = model.k_sample

    with torch.inference_mode():
        for batch_idx, batch_data in enumerate(loader):
            if model.task_type == 'classification':
                data, label = batch_data
                label = label.to(device)
            elif model.task_type == 'regression':
                data, event_time, censorship = batch_data
                event_time, censorship = event_time.to(device), censorship.to(device)

            data = data.to(device)

            if model.task_type == 'classification':
                logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
                acc_logger.log(Y_hat, label)
                loss = loss_fn(logits, label)

                inst_preds = instance_dict['inst_preds']
                inst_labels = instance_dict['inst_labels']
                inst_logger.log_batch(inst_preds, inst_labels)

                prob[batch_idx] = Y_prob.cpu().numpy()
                labels[batch_idx] = label.item()

                error = calculate_error(Y_hat, label)
                val_error += error
            else:  # Regression
                risk_score, _, instance_dict = model(data, label=None, instance_eval=True)
                # Accumulate for the entire dataset
                accumulated_risk_scores.append(risk_score.cpu())
                accumulated_event_times.append(event_time.cpu())
                accumulated_censorships.append(censorship.cpu())

            if instance_dict:
                instance_loss = instance_dict['instance_loss']
                inst_count += 1
                val_inst_loss += instance_loss.item()

    # Concatenate all accumulated data
    accumulated_risk_scores = torch.cat(accumulated_risk_scores, dim=0).to(device)
    accumulated_event_times = torch.cat(accumulated_event_times, dim=0).to(device)
    accumulated_censorships = torch.cat(accumulated_censorships, dim=0).to(device)
    # Compute loss for the whole dataset
    val_loss = loss_fn(accumulated_risk_scores, accumulated_event_times, accumulated_censorships).item()
 
    #val_loss /= len(loader) / accumulation_steps

    if model.task_type == 'classification':
            val_error /= len(loader)
            if n_classes == 2:
                auc = roc_auc_score(labels, prob[:, 1])
            else:
                aucs = []
                binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
                for class_idx in range(n_classes):
                    if class_idx in labels:
                        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                        aucs.append(calc_auc(fpr, tpr))
                    else:
                        aucs.append(float('nan'))
                auc = np.nanmean(np.array(aucs))

            print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))

            if inst_count > 0:
                val_inst_loss /= inst_count
                for i in range(2):
                    acc, correct, count = inst_logger.get_summary(i)
                    print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

            for i in range(n_classes):
                acc, correct, count = acc_logger.get_summary(i)
                print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
                if writer and acc is not None:
                    writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

            if writer:
                writer.add_scalar('val/loss', val_loss, epoch)
                writer.add_scalar('val/auc', auc, epoch)
                writer.add_scalar('val/error', val_error, epoch)
                writer.add_scalar('val/inst_loss', val_inst_loss, epoch)

    else:  # Regression
        # Compute Concordance Index for regression tasks
        c_index = concordance_index(
            event_times=accumulated_event_times.cpu().numpy(),  # Move to CPU before converting to NumPy
            predicted_scores=accumulated_risk_scores.cpu().numpy(),  # Move to CPU before converting to NumPy
            event_observed=accumulated_censorships.cpu().numpy()  # Move to CPU before converting to NumPy
            #event_times=event_times.cpu().numpy(),  # Move to CPU before converting to NumPy
            #predicted_scores=risk_scores.cpu().numpy(),  # Move to CPU before converting to NumPy
            #event_observed=censorships.cpu().numpy()  # Move to CPU before converting to NumPy
        )

        print('\nVal Set, val_loss: {:.4f}, C-index: {:.4f}'.format(val_loss, c_index))

        if writer:
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/c_index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes=None):
    model.eval()

    if model.task_type == 'classification':
        acc_logger = Accuracy_Logger(n_classes=n_classes)
        test_loss = 0.
        test_error = 0.

        all_probs = np.zeros((len(loader), n_classes))
        all_labels = np.zeros(len(loader))

        slide_ids = loader.dataset.slide_data['slide_id']
        patient_results = {}

        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.inference_mode():
                logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()
            
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
            error = calculate_error(Y_hat, label)
            test_error += error

        test_error /= len(loader)

        if n_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
            aucs = []
        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))

            auc = np.nanmean(np.array(aucs))


        return patient_results, test_error, auc, acc_logger
    
    else:
        all_risk_scores = []
        all_event_times = []
        all_censorships = []
        slide_ids = loader.dataset.slide_data['slide_id']
        patient_results = {}

        for batch_idx, (data, event_time, censorship) in enumerate(loader):
            data = data.to(device)
            event_time = event_time.to(device)  # Observed survival times
            censorship = censorship.to(device)  # Censorship flags (1 if event occurred, 0 if censored)
            slide_id = slide_ids.iloc[batch_idx]

            with torch.inference_mode():
                risk_scores, _, _ = model(data)  # Predict risk scores

            all_risk_scores.extend(risk_scores.cpu().numpy())
            all_event_times.extend(event_time.cpu().numpy())
            all_censorships.extend(censorship.cpu().numpy())
            
            patient_results.update({slide_id: {
                'slide_id': np.array(slide_id), 
                'risk_score': risk_scores.cpu().numpy(),
                'event_time': event_time.cpu().numpy(),
                'censorship': censorship.cpu().numpy()
            }})

        # Compute Concordance Index using lifelines
        c_index = concordance_index(
            event_times=np.array(all_event_times),
            predicted_scores=np.array(all_risk_scores),
            event_observed=np.array(all_censorships)
        )

        return patient_results, c_index

