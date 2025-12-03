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
        U = 0.1 * random_state.randn(n_users, K)
        V = 0.1 * random_state.randn(n_items, K)

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
        if mu is None:
            mu = self.mu
        if b_per_user is None:
            b_per_user = self.b_per_user
        if c_per_item is None:
            c_per_item = self.c_per_item
        if U is None:
            U = self.U
        if V is None:
            V = self.V
        # Look up embeddings and biases
        # mu is shape (1,), so treat as scalar mu[0]
        u_vecs = U[user_id_N]        # (N, K)
        v_vecs = V[item_id_N]        # (N, K)

        dot_uv = ag_np.sum(u_vecs * v_vecs, axis=1)  # (N,)

        yhat_N = mu[0] + b_per_user[user_id_N] + c_per_item[item_id_N] + dot_uv

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

    # Hyperparameters for this experiment (Problem 1B)
    K = 50
    step_size = 0.6
    n_epochs = 1000
    batch_size = 1000

    # Try several positive alpha values (L2 strength)
    alpha_list = [1, 0.5, 0.1]

    # Store traces per alpha
    results = {}

    for alpha in alpha_list:
        print(f"\nTraining model with K={K}, alpha={alpha}")

        model = CollabFilterOneVectorPerItem(
            n_epochs=n_epochs,
            batch_size=batch_size,
            step_size=step_size,
            n_factors=K,
            alpha=alpha,
        )
        model.init_parameter_dict(n_users, n_items, train_tuple)
        model.fit(train_tuple, valid_tuple)

        # Save copies of the traces so they don't get overwritten
        results[alpha] = dict(
            epoch=list(model.trace_epoch),
            train=list(model.trace_rmse_train),
            valid=list(model.trace_rmse_valid),
        )

    # ------------------------------------------------------------------
    # Figure 1B: Trace plot showing RMSE vs epoch for alpha > 0
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    for alpha in alpha_list:
        epochs = results[alpha]['epoch']
        train_rmse = results[alpha]['train']
        valid_rmse = results[alpha]['valid']

        ax.plot(
            epochs,
            train_rmse,
            label=f"train, α={alpha}",
        )
        ax.plot(
            epochs,
            valid_rmse,
            linestyle='--',
            label=f"valid, α={alpha}",
        )

    ax.set_xlabel("Epochs completed")
    ax.set_ylabel("RMSE")
    ax.set_title("K = 50, step_size = 0.6, RMSE vs. epoch for α > 0")
    ax.legend()
    plt.tight_layout()
    plt.show()
