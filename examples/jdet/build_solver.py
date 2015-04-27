import sys
import os
import shutil
sys.path.append('python/caffe/proto/')

from caffe_pb2 import SolverParameter
from datetime import date, datetime
from google.protobuf import text_format

net_name = sys.argv[1]
device_id = sys.argv[2]

solver_temp = 'models/jdet/solver.prototxt.example'
base_path = '/media/datadisk_c/snapshots/jdet/'

starting_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

solver_name = 'models/jdet/solver_{}.prototxt'.format(net_name)
proto_name = 'models/jdet/{}_train_val.prototxt'.format(net_name)
snapshot_name = '{}_{}/{}'.format(net_name, date.today().isoformat(), starting_time)
snapshot_dir = base_path + '{}'.format(snapshot_name)
log_dir = os.path.join(snapshot_dir, 'log')

if not os.path.isdir(snapshot_dir):
  os.makedirs(snapshot_dir)

if not os.path.isdir(log_dir):
  os.makedirs(log_dir)

snapshot_path = os.path.join(snapshot_dir, net_name)

#load the solver and build the final one
solver = SolverParameter()
text_format.Merge(open(solver_temp).read(), solver)

#modify parameter
solver.net = proto_name
solver.device_id = int(device_id)
solver.snapshot_prefix = snapshot_path

#output the solver
open(solver_name, 'w').write(text_format.MessageToString(solver))
shutil.copy(proto_name, os.path.join(snapshot_dir, '{}_proto_{}.prototxt'.format(net_name, starting_time)))
shutil.copy(solver_name, os.path.join(snapshot_dir, '{}_solver_{}.prototxt'.format(net_name, starting_time)))

print log_dir
