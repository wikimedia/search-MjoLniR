from __future__ import absolute_import
import hyperopt
import mjolnir.training.hyperopt


def test_maximize(folds_b):
    "Not an amazing test...basically sees if the happy path doesnt blow up"
    def f(params):
        assert isinstance(params, dict)
        assert 'max_depth' in params
        assert params['num_rounds'] == 50
        return [{
            'train': [0.80],
            'test': [0.79],
        }]

    space = {
        'num_rounds': 50,
        'max_depth': hyperopt.hp.quniform('max_depth', 1, 20, 1)
    }

    # mostly hyperopt just calls cross_validate, of which the integration with
    # xgboost is separately tested. Instead of going all the way into xgboost
    # mock it out w/MockModel.
    best_params, trails = mjolnir.training.hyperopt.maximize(
        f, space, max_evals=5)
    assert isinstance(best_params, dict)
    # num_rounds should have been unchanged
    assert 'num_rounds' in best_params
    assert best_params['num_rounds'] == 50
    # should have max_evals evaluations
    assert len(trails.trials) == 5
