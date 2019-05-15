import numpy as np
import torch

class BeamSearch:
    def __init__(self, end_index, max_steps=50, beam_size=10, beam_size_per_node=None):
        self._end_index = end_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.beam_size_per_node = beam_size_per_node or beam_size

    def search(self, start_predictions, start_state, step_function):
        batch_size = 1 # forget batching
        predictions = []
        backpointers = []

        start_class_log_probabilities, state = step_function(start_predictions, start_state)
        print(start_class_log_probabilities)
        print(state)
        raise ValueError

        num_classes = start_class_log_probabilities.size()[1]

        # Make sure `beam_size_per_node` is not larger than `num_classes`.
        if self.beam_size_per_node > num_classes:
            raise ValueError(f"Target vocab size ({num_classes:d}) too small "
                                     f"relative to beam_size_per_node ({self.beam_size_per_node:d}).\n"
                                     f"Please decrease beam_size or beam_size_per_node.")

        # shape: (batch_size, beam_size), (batch_size, beam_size)
        start_top_log_probabilities, start_predicted_classes = \
                start_class_log_probabilities.topk(self.beam_size)
        if self.beam_size == 1 and (start_predicted_classes == self._end_index).all():
            raise ValueError("Empty sequences predicted. You may want to increase the beam size or ensure "
                          "your step function is working properly.",
                          RuntimeWarning)
            return start_predicted_classes.unsqueeze(-1), start_top_log_probabilities

        last_log_probabilities = start_top_log_probabilities

        # shape: [(batch_size, beam_size)]
        predictions.append(start_predicted_classes)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        log_probs_after_end = start_class_log_probabilities.new_full(
                (batch_size * self.beam_size, num_classes),
                float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.

        # Set the same state for each element in the beam.
        for key, state_tensor in state.items():
            _, *last_dims = state_tensor.size()
            # shape: (batch_size * beam_size, *)
            state[key] = state_tensor.\
                    unsqueeze(1).\
                    expand(batch_size, self.beam_size, *last_dims).\
                    reshape(batch_size * self.beam_size, *last_dims)

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size)


            if (last_predictions == self._end_index).all():
                break


            class_log_probabilities, state = step_function(last_predictions, state)

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                    batch_size * self.beam_size,
                    num_classes
            )

            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            # shape: (batch_size * beam_size, num_classes)
            cleaned_log_probabilities = torch.where(
                    last_predictions_expanded == self._end_index,
                    log_probs_after_end,
                    class_log_probabilities
            )

            # shape (both): (batch_size * beam_size, beam_size_per_node)
            top_log_probabilities, predicted_classes = \
                cleaned_log_probabilities.topk(self.beam_size_per_node)

            # Here we expand the last log probabilities to (batch_size * beam_size, beam_size_per_node)
            # so that we can add them to the current log probs for this timestep.
            # This lets us maintain the log probability of each element on the beam.
            # shape: (batch_size * beam_size, beam_size_per_node)
            expanded_last_log_probabilities = last_log_probabilities.\
                    unsqueeze(2).\
                    expand(batch_size, self.beam_size, self.beam_size_per_node).\
                    reshape(batch_size * self.beam_size, self.beam_size_per_node)

            # shape: (batch_size * beam_size, beam_size_per_node)
            summed_top_log_probabilities = top_log_probabilities + expanded_last_log_probabilities

            # shape: (batch_size, beam_size * beam_size_per_node)
            reshaped_summed = summed_top_log_probabilities.\
                    reshape(batch_size, self.beam_size * self.beam_size_per_node)

            # shape: (batch_size, beam_size * beam_size_per_node)
            reshaped_predicted_classes = predicted_classes.\
                    reshape(batch_size, self.beam_size * self.beam_size_per_node)

            # Keep only the top `beam_size` beam indices.
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(self.beam_size)

            # Use the beam indices to extract the corresponding classes.
            # shape: (batch_size, beam_size)
            restricted_predicted_classes = reshaped_predicted_classes.gather(1, restricted_beam_indices)

            predictions.append(restricted_predicted_classes)

            # shape: (batch_size, beam_size)
            last_log_probabilities = restricted_beam_log_probs

            # The beam indices come from a `beam_size * beam_size_per_node` dimension where the
            # indices with a common ancestor are grouped together. Hence
            # dividing by beam_size_per_node gives the ancestor. (Note that this is integer
            # division as the tensor is a LongTensor.)
            # shape: (batch_size, beam_size)
            backpointer = restricted_beam_indices / self.beam_size_per_node

            backpointers.append(backpointer)

            # Keep only the pieces of the state tensors corresponding to the
            # ancestors created this iteration.
            for key, state_tensor in state.items():
                _, *last_dims = state_tensor.size()
                # shape: (batch_size, beam_size, *)
                expanded_backpointer = backpointer.\
                        view(batch_size, self.beam_size, *([1] * len(last_dims))).\
                        expand(batch_size, self.beam_size, *last_dims)

                # shape: (batch_size * beam_size, *)
                state[key] = state_tensor.\
                        reshape(batch_size, self.beam_size, *last_dims).\
                        gather(1, expanded_backpointer).\
                        reshape(batch_size * self.beam_size, *last_dims)

        if not torch.isfinite(last_log_probabilities).all():
            raise ValueError("Infinite log probabilities encountered. Some final sequences may not make sense. "
                          "This can happen when the beam size is larger than the number of valid (non-zero "
                          "probability) transitions that the step function produces.",
                          RuntimeWarning)

        # Reconstruct the sequences.
        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)

            reconstructed_predictions.append(cur_preds)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)

        return all_predictions, last_log_probabilities


def take_step(last_predictions: torch.Tensor,
              state):
    """
    Take decoding step.
    This is a simple function that defines how probabilities are computed for the
    next time step during the beam search.
    We use a simple target vocabulary of size 6. In this vocabulary, index 0 represents
    the start token, and index 5 represents the end token. The transition probability
    from a state where the last predicted token was token `j` to new token `i` is
    given by the `(i, j)` element of the matrix `transition_probabilities`.
    """
    log_probs_list = []
    for last_token in last_predictions:
        log_probs = torch.log(transition_probabilities[last_token.item()])
        log_probs_list.append(log_probs)

    print(log_probs_list)

    return torch.stack(log_probs_list), state

if __name__ == "__main__":
    transition_probabilities = torch.tensor(
        [[0.0, 0.4, 0.3, 0.2, 0.1, 0.0],  # start token -> jth token
         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # 1st token -> jth token
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 2nd token -> jth token
         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # ...
         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # ...
         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]  # end token -> jth token
    )

    end_index = transition_probabilities.size()[0] - 1
    beam_search = BeamSearch(end_index, max_steps=10, beam_size=3)
    expected_top_k = np.array(
        [[1, 2, 3, 4, 5],
         [2, 3, 4, 5, 5],
         [3, 4, 5, 5, 5]]
    )

    expected_log_probs = np.log(np.array([0.4, 0.3, 0.2]))

    expected_top_k = expected_top_k if expected_top_k is not None else expected_top_k
    expected_log_probs = expected_log_probs if expected_log_probs is not None else expected_log_probs
    state = {}

    batch_size = 1

    beam_search = beam_search or beam_search
    beam_size = beam_search.beam_size

    initial_predictions = torch.tensor([0] * 1)  # pylint: disable=not-callable
    top_k, log_probs = beam_search.search(initial_predictions, state, take_step)  # type: ignore
    print(top_k)
    print(log_probs)

    # top_k should be shape `(batch_size, beam_size, max_predicted_length)`.
    assert list(top_k.size())[:-1] == [batch_size, beam_size]
    np.testing.assert_array_equal(top_k[0].numpy(), expected_top_k)

    # log_probs should be shape `(batch_size, beam_size, max_predicted_length)`.
    assert list(log_probs.size()) == [batch_size, beam_size]
    np.testing.assert_allclose(log_probs[0].numpy(), expected_log_probs)