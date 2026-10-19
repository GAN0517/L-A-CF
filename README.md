# L-A

This package starts from an already integrated daily `data_model_ready.nc`. It contains the final three single-task land-atmosphere models (SM, ET, T), single-task MLP and RF baselines, a multi-task mediator LSTM, evaluation, and one-scenario counterfactual inference. It does not include upstream source-specific preprocessing, original experiment settings, trained checkpoints, data, PCMCI/SHAP analysis, or multi-seed orchestration.

## Layout

```text
L-A/
  modeling/                              final single-task model, train/evaluate
    counterfactual_mediator_lstm/       mediator train/evaluate and one-scenario inference
  baselines/mlp/                        single-task MLP train/evaluate
  baselines/rf/                         single-task RF train/evaluate
  validate_model_ready.py               input contract check
  DATA_CONTRACT.md
```

## Before running

Install `requirements.txt`. Edit the relevant YAML template: set `data.path`, the two external ET files, five static-field files, `train_end`, `val_end`, and your training settings. Values in these templates are examples, **not** the settings used for reported experiments. See `DATA_CONTRACT.md` for dimensions and units. Check the integrated input before training:

```bash
python validate_model_ready.py --config modeling/config_v4_single_swvl1.yaml
```

## Single runs

Run the three final targets separately by changing the config path among `config_v4_single_swvl1.yaml`, `config_v4_single_e_pos.yaml`, and `config_v4_single_evavt_pos.yaml`:

```bash
python modeling/train_arse_mlp.py --config modeling/config_v4_single_swvl1.yaml
python modeling/evaluate_arse_mlp.py --checkpoint PATH/TO/best_model.pt --split test
```

The MLP command trains the three specified single-task targets in one process. RF trains one target per invocation; repeat it with `e_pos` and `evavt_pos`:

```bash
python baselines/mlp/simple_mlp_single_task_loop/train_single_task_mlp_loop.py --config baselines/mlp/config_arse_mlp_three_head_full_rawres.yaml --targets swvl1 e_pos evavt_pos --use-static-land-features --static-source-dir data/static
python baselines/mlp/simple_mlp_single_task_loop/evaluate_single_task_mlp_loop.py --checkpoint-root PATH/TO/mlp_output --targets swvl1 e_pos evavt_pos --split test
python baselines/rf/rf/train_rf.py --config baselines/rf/common/config_baseline_common.yaml --model-config baselines/rf/rf/config_rf.yaml --target swvl1
python baselines/rf/rf/evaluate_rf.py --checkpoint PATH/TO/rf_checkpoint.joblib --split test
```

The mediator uses `r`, `t2m`, `skt`, `sshf_wm2`, and `slhf_wm2`. The sign-constraint coefficient and scales must be selected explicitly for a constrained run; the code uses the dataset's downward-positive sensible-heat convention.

```bash
python modeling/counterfactual_mediator_lstm/train_mediator_lstm.py --config modeling/config_v4_single_swvl1.yaml --model-types delta --h-sign-constraint-weight WEIGHT --h-sign-flux-scale FLUX_SCALE --h-sign-temp-scale TEMP_SCALE
python modeling/counterfactual_mediator_lstm/evaluate_mediator_lstm.py --checkpoint PATH/TO/mediator_best_model.pt --split test
```

Edit `modeling/counterfactual_mediator_lstm/configs/aod_ceres_scenario.yaml` to specify a single AOD perturbation and CERES response. The single-scenario inference compares each target's counterfactual prediction with the **same checkpoint's unchanged-input prediction** on aligned time/point samples; it does not compare with observations:

```bash
python modeling/counterfactual_mediator_lstm/run_v4_single_task_counterfactual_full_year.py --scenario-config modeling/counterfactual_mediator_lstm/configs/aod_ceres_scenario.yaml --data-path data/data_model_ready.nc --mediator-checkpoint PATH/TO/mediator_best_model.pt --swvl1-checkpoint PATH/TO/sm_best_model.pt --e-pos-checkpoint PATH/TO/et_best_model.pt --evavt-pos-checkpoint PATH/TO/t_best_model.pt --start-date YYYY-MM-DD --end-date YYYY-MM-DD --split all --output-dir outputs/counterfactual
```

The mediator-only scenario input generator is `modeling/counterfactual_mediator_lstm/run_hsign_multitask_mediator_scenario_input.py`. The `run_counterfactual_direct.py` file is retained as an import dependency for scenario application and mediator prediction, not as an alternate main workflow.
