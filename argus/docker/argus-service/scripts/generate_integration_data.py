#!/bin/python
# -*- coding: utf-8 -*- 

import argparse
import json
import os
import subprocess
import shutil

def main():
    parser = argparse.ArgumentParser(description='generatate integration data for APP')
    parser.add_argument('--app_bin', type=str, help='APP binary filepath.')
    parser.add_argument('--app_conf', type=str, help='APP conf filepath.')
    parser.add_argument('--out_dir', type=str, help='integration data output dir.')
    args = parser.parse_args()
    gen_integration_data(args.app_bin, args.app_conf, args.out_dir)

def gen_integration_data(bin_path, conf_path, out_dir):
    '''Generate integration data for APP

    Args:
        bin_path(string)    : APP binary file path
        conf_path(string)   : APP conf file path
        out_dir(string)     : output directory 

    Raises:
        Exception
    '''
    out = subprocess.check_output('%s -f %s info' % (bin_path, conf_path), shell=True)
    app_info = json.loads(out)
    #print app_info

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    gen_dashboard(app_info['monitor']['dashboards'], out_dir)
    
    
def gen_dashboard(dashboard_map, out_dir):
    monitor_dir = os.path.join(out_dir, "monitor")
    dashboards_dir = os.path.join(monitor_dir, "dashboards")
    if os.path.exists(dashboards_dir):
        shutil.rmtree(dashboards_dir)
    os.makedirs(dashboards_dir)
    for name, dashboard_str in dashboard_map.iteritems():
        d = json.loads(dashboard_str)
        fn = os.path.join(dashboards_dir, '%s.json' % (name))
        with open(fn, "w") as f:
            json.dump(d, f)

if __name__ == '__main__':
    main()