import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

data = scipy.io.loadmat("data/burgers_shock.mat")
Exact = np.real(data["usol"]).T
u_star = Exact.flatten()[:, None]

n_t = 100
n_x = 256
lb = np.array([-1.0, 0.0])
ub = np.array([1.0, 1.0])  # (X,T)
t = np.linspace(lb[1], ub[1], n_t).flatten()[:, None]
x = np.linspace(lb[0], ub[0], n_x).flatten()[:, None]

data = scipy.io.loadmat("result/Burgers_Equation/pred.mat")
u_pred = np.real(data["u_pred"])
u_pred = u_pred.flatten()[:, None]
u_pred = u_pred.reshape(n_t, n_x)

X_u_train = np.real(data["X_u_train"])


######################################################################
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)
h = ax.imshow(u_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)

ax.plot(
    X_u_train[:, 1],
    X_u_train[:, 0],
    'kx', label='Data (%d points)' % (X_u_train.shape[0]),
    markersize=4,  # marker size doubled
    clip_on=False,
    alpha=1.0
)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)

ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.9, -0.05),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)
ax.set_title('$u(t,x)$', fontsize=20)  # font size doubled
ax.tick_params(labelsize=15)

plt.savefig("result/Burgers_Equation/pred_img.jpg")
plt.show()

######################################################################
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)

gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x, Exact[25, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, u_pred[25, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = 0.25$', fontsize=15)
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, u_pred[50, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_title('$t = 0.50$', fontsize=15)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 2])
ax.plot(x, Exact[75, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, u_pred[75, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_title('$t = 0.75$', fontsize=15)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)


plt.savefig("result/Burgers_Equation/t_time_pred.jpg")
plt.show()


######################################################################
loss = np.real(data["loss"])[0]
loss_u = np.real(data["loss_u"])[0]
loss_f = np.real(data["loss_f"])[0]

show_interval = 0
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(loss[show_interval:], label="loss")
ax.plot(loss_u[show_interval:], label="loss_u")
ax.plot(loss_f[show_interval:], label="loss_f")
ax.set_xlabel('$iter$')
ax.set_ylabel('$loss$')
plt.legend()
plt.savefig("result/Burgers_Equation/loss.jpg")
plt.show()