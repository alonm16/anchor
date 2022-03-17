import os
import abc
import sys
import tqdm
import torch
from typing import Any, Callable
from pathlib import Path
from torch.utils.data import DataLoader
import torchtext
from train_results import FitResult, BatchResult, EpochResult
import models as models

from sklearn.metrics import r2_score
regression = False

class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device="cpu"):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        model.to(self.device)

    def fit(
            self,
            dl_train: torchtext.legacy.data.BucketIterator,
            dl_test: torchtext.legacy.data.BucketIterator,
            num_epochs,
            params,
            checkpoints: str = None,
            early_stopping: int = None,
            print_every=1,
            post_epoch_fn=None,
            **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        epochs_without_improvement = 0

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f"{checkpoints}.pt"
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f"*** Loading checkpoint file {checkpoint_filename}")
                saved_state = torch.load(checkpoint_filename, map_location=self.device)
                best_acc = saved_state.get("best_acc", best_acc)
                epochs_without_improvement = saved_state.get(
                    "ewi", epochs_without_improvement
                )
                self.model.load_state_dict(saved_state["model_state"])

        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f"--- EPOCH {epoch + 1}/{num_epochs} ---", verbose)

            # TODO:
            #  Train & evaluate for one epoch
            #  - Use the train/test_epoch methods.
            #  - Save losses and accuracies in the lists above.
            #  - Implement early stopping. This is a very useful and
            #    simple regularization technique that is highly recommended.
            # ====== YOUR CODE: ======
            train_result = self.train_epoch(dl_train, **kw)
            cur_losses, cur_accuracy = train_result.losses, train_result.accuracy
            cur_avg_loss = sum(cur_losses) / len(cur_losses)
            train_loss.append(cur_avg_loss)
            train_acc.append(cur_accuracy)

            test_result = self.test_epoch(dl_test, **kw)
            cur_losses, cur_accuracy = test_result.losses, test_result.accuracy
            cur_avg_loss = sum(cur_losses) / len(cur_losses)
            test_loss.append(cur_avg_loss)
            test_acc.append(cur_accuracy)

            actual_num_epochs += 1

            if not best_acc or cur_accuracy > best_acc:
                best_acc = cur_accuracy
                epochs_without_improvement = 0
                save_checkpoint = True
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping:
                break

                # ========================

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(
                    best_acc=best_acc,
                    ewi=epochs_without_improvement,
                    model_state=self.model.state_dict(),
                    parameters=params
                )
                torch.save(saved_state, checkpoint_filename)
                print(
                    f"*** Saved checkpoint {checkpoint_filename} " f"at epoch {epoch + 1}"
                )

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: torchtext.legacy.data.BucketIterator, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: torchtext.legacy.data.BucketIterator, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
            dl: torchtext.legacy.data.BucketIterator,
            forward_fn: Callable[[Any], BatchResult],
            verbose=True,
            max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        
        num_samples = len(dl) * dl.batch_size
        num_batches = len(dl)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            
            if regression:
                accuracy = num_correct/num_batches
                pbar.set_description(
                    f"{pbar_name} "
                    f"(Avg. Loss {avg_loss:.3f}, "
                    f"R2 {accuracy:.3f})"
                )
            else:
                accuracy = 100.0 * num_correct / num_samples
                            
                pbar.set_description(
                    f"{pbar_name} "
                    f"(Avg. Loss {avg_loss:.3f}, "
                    f"Accuracy {accuracy:.3f})"
                )

        return EpochResult(losses=losses, accuracy=accuracy)


class SentimentTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        super().__init__(model, loss_fn, optimizer, device)

    def train_batch(self, batch) -> BatchResult:
        x, y = batch.text, batch.label
        x = x.to(self.device)  # (S, B, E)
        y = y.to(self.device)  # (B,)
        # Forward pass
        if isinstance(self.model, models.VanillaGRU):
            y_pred_log_proba = self.model(x)
        elif isinstance(self.model, models.MultiHeadAttentionNet):
            y_pred_log_proba, _ = self.model(x)

       
        # Backward pass
        self.optimizer.zero_grad()
        loss = self.loss_fn(y_pred_log_proba, y)

        loss.backward()
        # self.model.embedding_layer.weight.grad[self.model.indices_to_zero] = 0
        
        #self.model.embedding_layer.weight.grad[self.model.indices_to_zero] = 0
        
        # Weight updates
        self.optimizer.step()

        y_pred = torch.argmax(y_pred_log_proba, dim=1)
        num_correct = torch.sum(y_pred == y).float().item()

        # ========================
        if regression: 
            pred =  y_pred_log_proba.reshape(-1).detach().cpu().numpy()
            y =y.cpu()
            score = r2_score(y, pred)
            # print(f'y {y}')
            # print(f' pred {pred}')
            # print(f'score {score}')
            return BatchResult(loss.item(), score)


        return BatchResult(loss.item(), num_correct)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch.text, batch.label
        x = x.to(self.device)  # (S, B, E)
        y = y.to(self.device)  # (B,)

        with torch.no_grad():
            if isinstance(self.model, models.VanillaGRU):
                y_pred_log_proba = self.model(x)
            elif isinstance(self.model, models.MultiHeadAttentionNet):
                y_pred_log_proba, _ = self.model(x)
            loss = self.loss_fn(y_pred_log_proba, y)
            y_pred = torch.argmax(y_pred_log_proba, dim=1)
            num_correct = torch.sum(y_pred == y).float()
            
        if regression: 
            pred =  y_pred_log_proba.reshape(-1).detach().cpu().numpy()
            y =y.cpu()
            score = r2_score(y, pred)
            # print(f'y {y}')
            # print(f' pred {pred}')
            # print(f'score {score}')
            return BatchResult(loss.item(), score)

        return BatchResult(loss.item(), num_correct.item())
    
