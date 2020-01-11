__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'development'

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def plot_hyperparameter(result, param_name, pretty_name, negative=True, save=False):
    
    if negative:
        res = result.copy()
        res['mean_train_score'] = -res['mean_train_score']
        res['mean_test_score'] = -res['mean_test_score']
    else:
        res = result.copy()

    fig, ax = plt.subplots(1,2, figsize=(15,6))

    X_axis = res[param_name].astype(float)

    ax[0].plot(X_axis, res['mean_train_score'], label='Train', color='r', alpha=.6)
    ax[0].fill_between(X_axis, (res['mean_train_score'] - res['std_train_score']).astype(float),
                            (res['mean_train_score'] + res['std_train_score']).astype(float), alpha=0.1, color='r')
    ax[0].plot(X_axis, res['mean_test_score'], label='Test', color='g', alpha=.6)
    ax[0].fill_between(X_axis, (res['mean_test_score'] - res['std_test_score']).astype(float),
                            (res['mean_test_score'] + res['std_test_score']).astype(float), alpha=0.1, color='g')

    ax[1].plot(X_axis, res['mean_fit_time'], label='Train', color='r', alpha=.6)
    ax[1].fill_between(X_axis, (res['mean_fit_time'] - res['std_fit_time']).astype(float),
                            (res['mean_fit_time'] + res['std_fit_time']).astype(float), alpha=0.1, color='r')
    ax[1].plot(X_axis, res['mean_score_time'], label='Test', color='g', alpha=.6)
    ax[1].fill_between(X_axis, (res['mean_score_time'] - res['std_score_time']).astype(float),
                            (res['mean_score_time'] + res['std_score_time']).astype(float), alpha=0.1, color='g')

    ax[0].legend()
    ax[1].legend()
    ax[0].set_title('Score', fontsize=14)
    ax[1].set_title('Time', fontsize=14)
    fig.suptitle(f'{pretty_name}, Scores and Times', fontsize=18)
    
    if save:
        plt.savefig('plots/' + save)

    plt.show()


def plot_two_hyperparms(result, param_x, param_y, pretty_name, negative=True, save=False):
    
    if negative:
        res = result.copy()
        res['mean_test_score'] = -res['mean_test_score']
    else:
        res = result.copy()

    fig, ax = plt.subplots(1,2, figsize=(15,6))

    X_axis = res[param_x].astype(float)
    Y_axis = res[param_y].astype(float)

    xg, yg = np.meshgrid(np.linspace(X_axis.min(), X_axis.max(), 100),
                         np.linspace(Y_axis.min(), Y_axis.max(), 100))
    
    triangles = tri.Triangulation(X_axis, Y_axis)
    tri_interp = tri.CubicTriInterpolator(triangles, res['mean_test_score'])
    zg = tri_interp(xg, yg)
    
    ax[0].contourf(xg, yg, zg, 
                   norm=plt.Normalize(vmax=res['mean_test_score'].max(), vmin=res['mean_test_score'].min()),
                   cmap=plt.cm.terrain)
    
    tri_interp = tri.CubicTriInterpolator(triangles, res['mean_fit_time'])
    zg = tri_interp(xg, yg)
    
    ax[1].contourf(xg, yg, zg, 
                   norm=plt.Normalize(vmax=res['mean_fit_time'].max(), vmin=res['mean_fit_time'].min()), 
                   cmap=plt.cm.terrain)
    
    ax[0].set_xlabel(param_x.split('__')[-1].title(), fontsize=12)
    ax[1].set_xlabel(param_x.split('__')[-1].title(), fontsize=12)
    ax[0].set_ylabel(param_y.split('__')[-1].title(), fontsize=12)
    ax[1].set_ylabel(param_y.split('__')[-1].title(), fontsize=12)
    ax[0].set_title('Test Score', fontsize=14)
    ax[1].set_title('Fit Time', fontsize=14)
    fig.suptitle(f'{pretty_name}', fontsize=18)
    
    if save:
        plt.savefig('plots/' + save)
    
    plt.show()
    
    
def _label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))
    return ax


def plot_coefficients(target_name, est_coefs, annotate=False):
    coefs_real = pd.read_pickle('data/simulated/coefficients.pkl')
    coefs_real = coefs_real['tar_lin_unc']
    
    comparison = pd.merge(coefs_real, coefs_est.reset_index(), on='feat', how='left').fillna(0)
    
    fig, ax = plt.subplots(1,2, figsize=(13,6))
    
    ax[0].scatter(comparison.coef, comparison['mean'], color='k')
    ax[0] = _plot_diagonal(ax[0])
    if annotate:
        ax[0] = _label_point(comparison.coef, comparison['mean'], comparison.feat, ax[0])
        
    ax[0].set_xlabel('True Coefficient', fontsize=12)
    ax[0].set_ylabel('Estimated Coefficient', fontsize=12)
    ax[0].set_title('True vs Estimated', fontsize=14)
        
    ax[1].scatter(comparison.feat, comparison.coef, color='g', alpha=0.7)
    ax[1].scatter(comparison.feat, comparison['mean'], color='r', alpha=0.7)
    ax[1].errorbar(comparison.feat, comparison['mean'], yerr=comparison['std'], 
                   ls='none', color='r', alpha=0.3)
    
    ax[1].set_xticklabels(comparison.feat, rotation=70)
    ax[1].set_title('Coefficient values', fontsize=14)
    
    plt.show()
    