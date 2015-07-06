import sys
import os
import shutil
sys.path.append('build/install/python/caffe/proto/')

from caffe_pb2 import SolverParameter, NetParameter
from datetime import date, datetime
from google.protobuf import text_format

net_name = sys.argv[1]
device_id = sys.argv[2]
base_path = sys.argv[3]

testing = False

solver_temp = 'models/jdet/solver.prototxt.example'
#base_path = '/media/datadisk_c/snapshots/jdet/'
db_path_file = 'examples/jdet/db_paths.yaml'

import yaml
db_path = yaml.load(open(db_path_file));
train_db_path = db_path['train_{}'.format(net_name)]
val_db_path = db_path['val_{}'.format(net_name)]

starting_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

solver_name = 'models/jdet/solver_{}.prototxt'.format(net_name)
proto_name = 'models/jdet/{}_train_val.prototxt'.format(net_name)
snapshot_name = '{}_{}/{}_gpu_{}'.format(net_name, date.today().isoformat(), starting_time, device_id)
snapshot_dir = base_path + '{}'.format(snapshot_name)
log_dir = os.path.join(snapshot_dir, 'log')

if not os.path.isdir(snapshot_dir) and not testing:
  os.makedirs(snapshot_dir)

if not os.path.isdir(log_dir) and not testing:
  os.makedirs(log_dir)

snapshot_path = os.path.join(snapshot_dir, net_name)

#load the solver and build the final one
solver = SolverParameter()
text_format.Merge(open(solver_temp).read(), solver)

#modify parameter
solver.net = proto_name
#solver.device_id = int(device_id)
solver.snapshot_prefix = snapshot_path

#modify network file to point to the local database
net_proto = NetParameter()
text_format.Merge(open(proto_name).read(), net_proto)

for i in range(len(net_proto.layer)):
  if net_proto.layer[i].name == 'data':
    if net_proto.layer[i].include[0].phase == 0: #train
      net_proto.layer[i].data_param.source = train_db_path
    else:
      net_proto.layer[i].data_param.source = val_db_path # val

if testing:
  solver.snapshot = 0

open(solver_name, 'w').write(text_format.MessageToString(solver))
open(proto_name, 'w').write(text_format.MessageToString(net_proto))

#output the solver
if not testing:
  shutil.copy(proto_name, os.path.join(snapshot_dir, '{}_proto_{}.prototxt'.format(net_name, starting_time)))
  shutil.copy(solver_name, os.path.join(snapshot_dir, '{}_solver_{}.prototxt'.format(net_name, starting_time)))
else:
  log_dir="/media/datadisk_c/snapshots/tmp"

print log_dir
