# Regression Test

The regression test for the `ai8x-training` repository is tested when there is a pull request for the `develop` branch of `MaximIntegratedAI/ai8x-training` by triggering `test.yaml` GitHub actions.

## Last Tested Code

`last_dev.py` generates the log files for the last tested code. These log files are used for comparing the newly pushed code to check if there are any significant changes in the trained values. Tracking is done by checking the hash of the commit.

## Creating Test Scripts

The sample training scripts are under the `scripts` path. In order to create training scripts for regression tests, these scripts are rewritten by changing their epoch numbers by running `regression/create_test_script.py`. The aim of changing the epoch number is to keep the duration of the test under control. This epoch number is defined in `regression/test_config.yaml` for each model/dataset combination. Since the sizes of the models and the datasets vary, different epoch numbers can be defined for each of them in order to create a healthy test. If a new training script is added, the epoch number and threshold values must be defined in the `regression/test_config.yaml` file for the relevant model.

## Comparing Log Files

After running test scripts for newly pushed code, the log files are saved and compared to the last tested code’s log files by running `regression/log_comparison.py`, and the results are saved.

## Pass-Fail Decision

In the comparison, the test success criterion is that the difference does not exceed the threshold values defined in `regression/test_config.yaml` as a percentage. If all the training scripts pass the test, `pass_fail.py` completes with success. Otherwise, it fails and exits.

## ONNX Export

Scripts for ONNX export are created and run by running `create_onnx_scripts.py` by configuring `Onnx_Status: True` in `regression/test_config.yaml`. If it is set to `False`, ONNX export will be skipped.

## Configuration

In `regression/test_config.yaml`, the `Onnx_Status` and `Qat_Test` settings should be defined to `True` when ONNX export or QAT tests by using `policies/qat_policy.yaml` are desired. When `Qat_Test` is set to `False`, QAT will be done according to the main training script. All threshold values and test epoch numbers for each model/dataset combination are also configured in this file. In order to set up the test on a new system, `regression/paths.yaml` needs to be configured accordingly.

## Setting Up Regression Test

### GitHub Actions

GitHub Actions is a continuous integration (CI) and continuous deployment (CD) platform provided by GitHub. It allows developers to automate various tasks, workflows, and processes directly within their GitHub repositories. A GitHub Workflow is an automated process defined using a YAML file that helps automate various tasks in a GitHub repository.

In this project, with GitHub Actions, there is a 'test.yml' workflow that is triggered when a pull request is opened for the 'develop' branch of the 'MaximIntegratedAI/ai8x-training' repository. This workflow contains and runs the jobs and steps required for the regression test. Also, a self hosted GitHub Runner is used to run regression test actions in this workflow. In order to install GitHub Runner, go to Settings -> Actions -> Runners -> New self-hosted runner on GitHub. To learn more about GitHub Actions, see [GitHub Actions](https://docs.github.com/en/actions/quickstart).

After installing and configuring a GitHub Runner in your local environment, configure it to start as a service during system startup in order to ensure that the self-hosted runner runs continuously and automatically. You can find more information about systemd services at [Systemd Services](https://linuxhandbook.com/create-systemd-services/).
