import datetime
import numpy as np
import os
import random
import shutil
import sys
import time
import torch
from typing import List

from .common import dump_json, print_out
from .pytorch import move_to_gpu, rescale_gradients

__all__ = ["set_random_seed", "create_output_dir", "train", "final_evaluate", "generate_predictions"]


'''Update date: 2019-March-01'''
def set_random_seed(seed=13370):
    if seed > 0:
        random.seed(seed)
        np.random.seed(int(seed / 10))
        torch.manual_seed(int(seed / 100))
        torch.cuda.manual_seed(int(seed / 100))
        torch.cuda.manual_seed_all(int(seed / 100))


'''Update date: 2019-March-01'''
def create_output_dir(output_dir):
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise ValueError("%s is not empty." % output_dir)
    os.makedirs(output_dir, exist_ok=True)


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py (batch_loss)
Update date: 2019-March-03'''
def _batch_loss(model, batch):
    batch = move_to_gpu(batch, cuda_device=0)
    output_dict = model(**batch)
    loss = output_dict.get("loss", 0.0)
    return loss


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py (_validation_loss)
Update date: 2019-April-20'''
def _get_val_loss(model, iterator, data):
    model.eval()
    generator = iterator(data, shuffle=False)
    total_loss, batch_counter = 0.0, 0
    for batch in generator:
        batch_counter += 1
        _loss = _batch_loss(model, batch)
        if isinstance(_loss, float):
            total_loss += _loss
        else:
            total_loss += _loss.item()
    loss = float(total_loss / batch_counter) if batch_counter > 0 else 0.0
    return loss


'''Update date: 2019-April-20'''
def _is_best_model_so_far(this_epoch_score: float, score_per_epoch: List[float]):
    if not score_per_epoch:
        return True
    else:
        return this_epoch_score > max(score_per_epoch)


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py
Update date: 2019-April-20'''
def _output_metrics_to_console(train_metrics, dev_metrics={}):
    metric_names = list(train_metrics.keys()) + list(dev_metrics.keys())
    metric_names = list(set(metric_names))
    train_metrics = ["%s: %s" % (k, str(train_metrics.get(k, 0))) for k in metric_names]
    print_out(" # Train set \n     %s" % ("; ".join(train_metrics)))
    dev_metrics = ["%s: %s" % (k, str(dev_metrics.get(k, 0))) for k in metric_names]
    print_out(" # Dev set \n     %s" % ("; ".join(dev_metrics)))


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py (_save_checkpoint)
Update date: 2019-April-20'''
def _save_checkpoint(model_dir, model, epoch, is_best=False):
    model_path = os.path.join(model_dir, "epoch_%s.th" % epoch)
    torch.save(model.state_dict(), model_path)

    if is_best:
        print_out(" # Best dev performance so far. Copying weights to %s/best.th" % model_dir)
        shutil.copyfile(model_path, os.path.join(model_dir, "best.th"))

    return model_path


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py
Update date: 2019-April-20'''
def _should_early_stop(score_per_epoch: List[float], patience=0):
    if patience > 0 and patience < len(score_per_epoch):
        return max(score_per_epoch[-patience:]) <= max(score_per_epoch[:-patience])
    return False


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py (_train_epoch)
Update date: 2019-April-20'''
def _train_epoch(model, optimizer, iterator, data, shuffle=True, lr_scheduler=None):
    model.train()

    total_loss = 0.0
    generator = iterator(data, shuffle=shuffle)
    num_batches = iterator.get_num_batches(data)
    batch_counter = 0
    summary_interval = max(min(50, int(num_batches / 5)), 1)

    for batch in generator:
        batch_counter += 1
        optimizer.zero_grad()
        loss = _batch_loss(model, batch)
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        loss.backward()
        total_loss += loss.item()
        rescale_gradients(model)
        optimizer.step()

        metrics = model.get_metrics(reset=False)
        metrics["loss"] = float(total_loss / batch_counter) if batch_counter > 0 else 0.0

        if batch_counter % summary_interval == 0 or batch_counter == num_batches:
            print_out("%d out of %d batches, loss: %.3f" % (batch_counter, num_batches, metrics["loss"]))

    metrics = model.get_metrics(reset=True)
    metrics["loss"] = float(total_loss / batch_counter) if batch_counter > 0 else 0.0
    return metrics


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py (train)
Update date: 2019-April-20'''
def train(model, optimizer, train_data, train_iterator, dev_data, dev_iterator, args, lr_scheduler=None):
    model_dir = args["output_dir"]
    max_epoches = args["max_epochs"]
    max_save_models = args["max_save_models"]
    patience = args["early_stop_patience"]
    validation_metric = args["validation_metric"]

    validation_metric_per_epoch = []
    metrics = {}
    training_start_time = time.time()

    model_paths = []

    for epoch in range(0, max_epoches):
        epoch_start_time = time.time()
        print_out("*" * 20 + "\n" + "Epoch %d/%d\n" % (epoch + 1, max_epoches) + "*" * 20)
        train_metrics = _train_epoch(model, optimizer, train_iterator, train_data, lr_scheduler=lr_scheduler)
        with torch.no_grad():
            val_loss = _get_val_loss(model, dev_iterator, dev_data)
            val_metrics = model.get_metrics(reset=True)
            val_metrics["loss"] = val_loss
            this_epoch_val_metric = val_metrics[validation_metric]
            is_best = _is_best_model_so_far(this_epoch_val_metric, validation_metric_per_epoch)
            validation_metric_per_epoch.append(this_epoch_val_metric)

        _output_metrics_to_console(train_metrics, val_metrics)

        if lr_scheduler is not None: lr_scheduler.step(this_epoch_val_metric, epoch)

        training_elapsed_time = time.time() - training_start_time
        metrics["training_elapsed_time"] = time.strftime("%H:%M:%S", time.gmtime(training_elapsed_time))
        metrics["epoch"] = epoch

        for k, v in train_metrics.items():
            metrics["training_" + k] = v
        for k, v in val_metrics.items():
            metrics["validation_" + k] = v

        if is_best:
            metrics["best_epoch"] = epoch
            for k, v in val_metrics.items():
                metrics["best_validation_" + k] = v

        model_path = _save_checkpoint(model_dir, model, epoch, is_best)
        model_paths.append([time.time(), model_path])

        if len(model_paths) > max_save_models:
            paths_to_remove = model_paths.pop(0)
            for f in paths_to_remove[1:]:
                os.remove(f)

        if _should_early_stop(validation_metric_per_epoch, patience):
            print_out(" # Ran out of patience. Stopping training.")
            break

        epoch_elapsed_time = time.time() - epoch_start_time
        estimated_time_remaining = training_elapsed_time * (max_epoches - epoch - 1) / (epoch + 1)
        print_out(" # Epoch duration: %s, estimated training time remaining: %s" % (
            str(time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed_time))),
            str(datetime.timedelta(seconds=int(estimated_time_remaining)))))
        sys.stdout.flush()

    return metrics, model_paths


'''Update May-3-2019'''
def generate_predictions(model, instances, iterator, output_namespace="preds"):
    pred_instances = []
    with torch.no_grad():
        model.eval()
        generator = iterator(instances, shuffle=False)
        for batch in generator:
            preds = model(**move_to_gpu(batch)).get(output_namespace)
            for i, pred in enumerate(preds):
                pred_instance = {"pred": pred}
                for k, v in batch.items():
                    if isinstance(v, dict):
                        for k, v in v.items():
                            pred_instance[k] = v[i]
                    else:
                        pred_instance[k] = v[i]
                pred_instances.append(pred_instance)
    return pred_instances


'''Update May-3-2019'''
def final_evaluate(model, vocab, test_data, iterator, eval_func, metrics, args, model_paths=None):
    if args["max_save_models"] == 1:
        best_model_state_path = os.path.join(args["output_dir"], "best.th")
        metrics["model_name"] = "best.th"
        model.load_state_dict(torch.load(best_model_state_path))
        test_metrics = eval_func(model, vocab, test_data, iterator)
        for key, value in test_metrics.items():
            metrics["test_" + key] = value
        dump_json("%s.json" % args["output_dir"], metrics)
    else:
        for model_path in model_paths:
            metrics["model_name"] = model_path[1].split("/")[1]
            model.load_state_dict(torch.load(model_path[1]))
            test_metrics = eval_func(model, vocab, test_data, iterator)
            for key, value in test_metrics.items():
                metrics["test_" + key] = value
            dump_json("%s.json" % args["output_dir"], metrics)