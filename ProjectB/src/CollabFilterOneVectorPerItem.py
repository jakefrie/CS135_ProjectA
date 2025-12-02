'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets

# Some packages you might need (uncomment as necessary)
## import pandas as pd
## import matplotlib

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state # inherited RandomState object

        # Use mean rating from training data as initial global mean
        ratings_train = train_tuple[2]
        mean_rating = ag_np.mean(ratings_train)

        K = self.n_factors

        # Global mean as 1D array of size (1,)
        mu = ag_np.array([mean_rating])

        # Per-user and per-item biases initialized near zero
        b_per_user = ag_np.zeros(n_users)
        c_per_item = ag_np.zeros(n_items)

        # Latent factors: small random normal values
        U = 0.01 * random_state.randn(n_users, K)
        V = 0.01 * random_state.randn(n_items, K)

        # TIP: use self.n_factors to access number of hidden dimensions
        self.param_dict = dict(
            mu=mu,
            b_per_user=ag_np.array(b_per_user),
            c_per_item=ag_np.array(c_per_item),
            U=ag_np.array(U),
            V=ag_np.array(V),
        )


    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''

        # Look up embeddings and biases
        # mu is shape (1,), so treat as scalar mu[0]
        u_vecs = U[user_id_N]        # (N, K)
        v_vecs = V[item_id_N]        # (N, K)

        dot_uv = ag_np.sum(u_vecs * v_vecs, axis=1)  # (N,)

        yhat_N = mu[0] + b_per_user[user_id_N] + c_per_item[item_id_N] + dot_uv

        # (Optional) you could clip to [1, 5] for evaluation if desired:
        # yhat_N = ag_np.clip(yhat_N, 1.0, 5.0)

        return yhat_N


    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        user_id_N, item_id_N, y_N = data_tuple

        # Predictions using current params
        yhat_N = self.predict(user_id_N, item_id_N, **param_dict)

        # Squared error term (sum over batch; base class later rescales)
        diff = yhat_N - y_N
        sq_err = ag_np.sum(diff ** 2)

        # L2 regularization on latent factors U and V
        U = param_dict['U']
        V = param_dict['V']
        reg = self.alpha * (ag_np.sum(U ** 2) + ag_np.sum(V ** 2))

        loss_total = sq_err + reg
        return loss_total  


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()

    Ks = [2, 10, 50]

    results = {}  # store traces per K

    for K in Ks:
        print(f"\nTraining model with K = {K}")

        model = CollabFilterOneVectorPerItem(
            n_epochs=100,
            batch_size=1000,
            step_size=0.1,
            n_factors=K,
            alpha=0.0
        )
        model.init_parameter_dict(n_users, n_items, train_tuple)
        model.fit(train_tuple, valid_tuple)

        # Save COPY of traces so they aren't overwritten
        results[K] = dict(
            epoch=list(model.trace_epoch),
            train=list(model.trace_rmse_train),
            valid=list(model.trace_rmse_valid)
        )

    # -------------------------------------------------------------------------
    # PLOTTING SECTION
    # -------------------------------------------------------------------------

    # Side-by-side plots: K = 2, 10, 50 (train + valid)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, K in zip(axes, Ks):
        ax.plot(results[K]['epoch'], results[K]['train'], label='Train RMSE')
        ax.plot(results[K]['epoch'], results[K]['valid'], label='Valid RMSE')
        ax.set_title(f"K = {K}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE")
        ax.legend()

    plt.tight_layout()
    plt.show()
