#!/usr/bin/env python3
"""Compile the three small GPU fixtures locally; never execute a GPU here."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if len(sys.argv)!=2:raise SystemExit('usage: build_supplement_fixtures.py EXISTING_EXPERIMENT_DIR')
BASE=(ROOT/sys.argv[1]).resolve(strict=True)
specs={
    'test_primitives':('test_l3_supplement.cu',ROOT/'core/include',['-DL3_DIRECT_RX=true','-DL3_FAULT_INJECT_ACK_DELAY=true','-DBULK_NO_CACHE=true']),
    'test_dq':('test_dq_publication.cu',ROOT/'core/include',[]),
    'test_capacity':('test_dq_capacity.cu',BASE/'dual_diagnostic_source/core/include',[]),
}
for name,(source,include,extra) in specs.items():
    binary=BASE/name
    if binary.exists():raise RuntimeError('Refusing to overwrite an existing fixture: '+str(binary))
    cmd=['nvcc',str(ROOT/'scripts/multigpu'/source),'-o',str(binary),'-O3','-std=c++17','-rdc=true','-gencode=arch=compute_80,code=sm_80','-I'+str(include),'-I/a100-data/wyh/boost_1_87_0/',*extra]
    (BASE/(name+'_command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
    with (BASE/(name+'_build.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('Build failed: '+name)
    (BASE/(name+'_sha256.txt')).write_text(hashlib.sha256(binary.read_bytes()).hexdigest()+'\n')
    print(name,'BUILD_PASS',flush=True)
