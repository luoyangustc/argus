#!/bin/python

import argparse
import json
import os
import jinja2

def main():
    parser = argparse.ArgumentParser(description='generatate app configure by app meta info')
    parser.add_argument('--app_build_conf_file', type=str, help='APP build config file path.')
    parser.add_argument('--app_tpl_dir', type=str, help='APP template directory.')
    parser.add_argument('--app_out_dir', type=str, help='APP output directory.')
    args = parser.parse_args()
    generate_app(args.app_build_conf_file, os.path.join(args.app_tpl_dir, 'app_suite_test.go.j2'), os.path.join(args.app_out_dir, 'app_suite_test.go'))


def generate_app(app_build_conf_file, tpl_filepath, out_filepath):
    '''Parse app meta file, and generate app conf

    Args:
        app meta(dict)              : app meta info
        tpl_filepath(string)        : app conf template file path
        out_filepath(string)        : generated file path
    
    Raises:
        Exception
    '''

    print 'tpl(%s) => out(%s)' % (tpl_filepath, out_filepath)
    app_build = json.load(open(app_build_conf_file, 'r'))
    tpl = jinja2.Template(open(tpl_filepath, 'r').read())
    content = tpl.render(app=app_build)
    basedir = os.path.dirname(out_filepath)
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    with open(out_filepath, 'w') as f:
        f.write(content)

    
if __name__ == '__main__':
    main()
