__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'development'


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

