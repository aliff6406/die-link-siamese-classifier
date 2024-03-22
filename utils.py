import os
import csv
from datetime import datetime
from pytz import timezone
from pathlib import Path


def cur_time():
    fmt = '%Y-%m-%d %H:%M:%S %Z%z'
    uk_time = timezone('Europe/London')
    loc_dt = datetime.now(uk_time)
    return loc_dt.strftime(fmt).replace(' ', '_')

def create_if_not_exist(path):
    if not os.path.exists(path):
        Path(path).touch()

def init_run_log(path):
    create_if_not_exist(path)
    with open(path, 'w') as f:
        f.write('epochs,initial_lr,bs,weight_decay,optimizer,momentum,scheduler,step_size,gamma\n')

def init_log(path):
    create_if_not_exist(path)
    with open(path, 'r') as f:
        if len(f.readlines()) <= 0:
            init_log_header(path)

def init_log_header(path):
    with open(path, 'w') as f:
        f.write('epoch,loss,acc,bs,lr\n')

def init_evaluate_log(path):
    create_if_not_exist(path)
    with open(path, 'w') as f:
        f.write('accuracy,precision,recall,f1score,auc,fpr,TPR,FPR\n')

def init_problematic_coin_log(path):
    create_if_not_exist(path)
    with open(path, 'w') as f:
        f.write('coin1,coin2,prediction\n')
        
def write_csv(file, newrow):
    with open(file, mode='a') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(newrow)

def load_losses_accs(path):
    losses = []
    accs = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            losses.append(float(row[1]))
            accs.append(float(row[2]))
    return losses, accs
