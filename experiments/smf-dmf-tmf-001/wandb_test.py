# experiments/smf-dmf-tmf-001/wandb_test.py
import wandb
import os
os.environ['WANDB__SERVICE_WAIT'] = '900'
run = wandb.init(mode="online", project="wandb_test")
run.log({"test":123})
run.finish()