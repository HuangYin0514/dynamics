from jax import random, vmap, jit
import jax.numpy as np
import scipy.io

from deeponet.dataset import getBurgersEquationDataSet
from deeponet.model import PI_DeepONet


# Geneate test data corresponding to one input sample
def generate_one_test_data(idx,usol, m=101, P=101):

    u = usol[idx]
    u0 = u[0,:]

    t = np.linspace(0, 1, P)
    x = np.linspace(0, 1, P)
    T, X = np.meshgrid(t, x)

    s = u.T.flatten()
    u = np.tile(u0, (P**2, 1))
    y = np.hstack([T.flatten()[:,None], X.flatten()[:,None]])

    return u, y, s

# Geneate training data corresponding to N input sample
def compute_error(idx, usol, m, P):
    u_test, y_test, s_test = generate_one_test_data(idx, usol, m, P)

    u_test = u_test.reshape(P ** 2, -1)
    y_test = y_test.reshape(P ** 2, -1)
    s_test = s_test.reshape(P ** 2, -1)

    s_pred = model.predict_s(params, u_test, y_test)[:, None]
    error = np.linalg.norm(s_test - s_pred) / np.linalg.norm(s_test)

    return error

if __name__ == '__main__':
    ics_dataset, bcs_dataset, res_dataset = getBurgersEquationDataSet()

    # Initialize model
    m = 101  # number of sensors for input samples
    # branch_layers = [m, 100, 100, 100, 100, 100, 100, 100]
    # trunk_layers = [2, 100, 100, 100, 100, 100, 100, 100]
    branch_layers = [m, 10]
    trunk_layers = [2, 10]
    model = PI_DeepONet(branch_layers, trunk_layers)

    # Train
    # Note: may meet OOM issue if use Colab. Please train this model on the server.
    model.train(ics_dataset, bcs_dataset, res_dataset, nIter=1)



    # Predict
    params = model.get_params(model.opt_state)
    N_train = 10  # number of input samples used for training
    idx = random.randint(key=random.PRNGKey(12345), shape=(400,), minval=N_train, maxval=2000)
    k = 1500
    N_test = 100
    P_test = 101  # resolution of uniform grid for the test data
    idx = np.arange(k, k + N_test)
    # Load data
    path = 'data/Burger.mat'  # Please use the matlab script to generate data
    data = scipy.io.loadmat(path)
    usol = np.array(data['output'])
    errors = vmap(compute_error, in_axes=(0, None, None, None))(idx, usol, m, P_test)
    mean_error = errors.mean()
    print(mean_error)

    print("run is done.")