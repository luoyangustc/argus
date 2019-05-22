import aisdk.common.render
import subprocess
import os
from aisdk.common.download_model import download_model_by_app, get_cfg_by_app, extract_model_by_app, read_env_file
from .flavor import Flavor
FNULL = open(os.devnull, 'w')


def gpu_index_env():
    GPU_INDEX = int(os.getenv("GPU_INDEX", "0"))
    return "NV_GPU={}".format(GPU_INDEX)


class RunCmdException(Exception):
    pass


def is_jenkins():
    return os.getenv("JENKINS") == "true"


def docker_tty_cmd():
    return "-t" if is_jenkins() else "-it"


def fix_jenkins_file_permission():
    if not is_jenkins():
        return
    run("mkdir -p res res_build")
    run("sudo chmod -R 777 res res_build")


def run(cmd, quiet=False):
    if quiet:
        p = subprocess.Popen(
            cmd,
            stdout=FNULL,
            stderr=FNULL,
            shell=True,
            executable='/bin/bash')
    else:
        print('- ', cmd)
        p = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
    p.wait()
    if p.returncode != 0:
        raise RunCmdException()


def docker_login_ava_public():
    env = read_env_file()
    run('docker login reg.qiniu.com -u {} -p {}'.format(
        env['AVA_PUBLIC_USERNAME'], env['AVA_PUBLIC_PASSWORD']),
        quiet=True)


def docker_login_ava_test():
    env = read_env_file()
    run('docker login reg.qiniu.com -u {} -p {}'.format(
        env['AVA_TEST_USERNAME'], env['AVA_TEST_PASSWORD']),
        quiet=True)


def docker_login_ava_prd():
    env = read_env_file()
    run('docker login reg.qiniu.com -u {} -p {}'.format(
        env['AVA_PRD_USERNAME'], env['AVA_PRD_PASSWORD']),
        quiet=True)


def test_common():
    fix_jenkins_file_permission()
    image = "tmp_test_common_env"
    docker_login_ava_public()
    run('docker build -t {image} -f baseimage/test_common_env/Dockerfile .'.
        format(image=image))
    run('''
docker run \
{docker_tty_cmd} \
--rm \
-v {root_dir}:/src \
--name tmp_test_common_env {image} \
/src/hack/runtest-common.sh
'''.format(root_dir=os.getcwd(), image=image, docker_tty_cmd=docker_tty_cmd()))


def test_app(app_name):
    fix_jenkins_file_permission()
    if app_name == 'common':
        test_common()
        return
    copy_src = False
    aisdk.common.render.render_docker_file(app_name, copy_src=copy_src)
    run('rm -rf res_build/model')
    run('mkdir -p res_build/model/')

    download_model_by_app(app_name)
    extract_model_by_app(app_name)

    cfg = get_cfg_by_app(app_name)
    src = cfg['build']['model_tar']
    if copy_src:
        run('cp -rf {src} {dist}'.format(
            src=src, dist=src.replace('res', 'res_build')))

    docker_login_ava_prd()
    image = 'tmp-{app_name}-local-test'.format(app_name=app_name)
    run('docker build -t {image} -f ./res/tmp_dockerfile .'.format(
        image=image))
    run('''
{gpu_index_env} nvidia-docker run \
{docker_tty_cmd} \
--rm \
--net {net} \
-e USE_DEVICE=GPU \
-e APP={app_name} \
-v {root_dir}:/src \
--name tmp-ai-sdk-test-{app_name} {image} \
/src/hack/runtest.sh
'''.format(
        app_name=app_name,
        image=image,
        net='host' if is_jenkins() else 'bridge',
        root_dir=os.getcwd(),
        docker_tty_cmd=docker_tty_cmd(),
        gpu_index_env=gpu_index_env()))


def run_app(app_name, not_run_app=False, flavor=''):
    if not flavor:
        flavor = 'DEV'
    Flavor(flavor)
    fix_jenkins_file_permission()
    copy_src = True
    aisdk.common.render.render_docker_file(app_name, copy_src=copy_src)
    run('rm -rf res_build/model')
    run('mkdir -p res_build/model/')

    download_model_by_app(app_name)
    extract_model_by_app(app_name)

    cfg = get_cfg_by_app(app_name)
    src = cfg['build']['model_tar'].replace('.tar', '')
    if copy_src:
        run('cp -rf {src} {dist}'.format(
            src=src, dist=src.replace('res', 'res_build')))

    docker_login_ava_prd()
    image = 'tmp_{app_name}-local-test'.format(app_name=app_name)
    if is_jenkins():
        image = 'tmp_{app_name}-jenkins-test'.format(app_name=app_name)
    run('docker build -t {image} -f ./res/tmp_dockerfile .'.format(
        image=image))
    if not not_run_app:
        run('''
        {gpu_index_env} nvidia-docker run \
        {docker_tty_cmd} \
        --rm \
        --net host \
        -e USE_DEVICE=GPU \
        -e PORT_HTTP=9100 \
        -e AISDK_FLAVOR={flavor} \
        --name tmp-ai-sdk-test {image}
        '''.format(
            image=image,
            flavor=flavor,
            docker_tty_cmd=docker_tty_cmd(),
            gpu_index_env=gpu_index_env()))


def build_core():
    run("docker build -t tmp-build-eval-core -f golang/src/qiniu.com/ai-sdk/eval_core/Dockerfile ."
        )
    run("docker run {docker_tty_cmd} --rm -v `pwd`/res_build:/out -v `cd ../../src/qiniu.com/argus && pwd `:/src/qiniu.com/argus -v `pwd`:/src/qiniu.com/ai-sdk tmp-build-eval-core /src/qiniu.com/ai-sdk/golang/src/qiniu.com/ai-sdk/eval_core/build.sh"
        .format(docker_tty_cmd=docker_tty_cmd()))


def lint():
    if is_jenkins():
        fix_jenkins_file_permission()
        image = "tmp_test_common_env"
        docker_login_ava_public()
        run('docker build -t {image} -f baseimage/test_common_env/Dockerfile .'
            .format(image=image))
        run('''
    docker run \
    {docker_tty_cmd} \
    --rm \
    --net host \
    -v {root_dir}:/src \
    --name tmp_test_common_env {image} \
    /src/hack/lint.sh
    '''.format(
            root_dir=os.getcwd(), image=image,
            docker_tty_cmd=docker_tty_cmd()))
    else:
        run('''cd python && \
            python2 -c "import pylint; assert pylint.__version__=='1.9.3'" && \
            python2 -c "import yapf; assert yapf.__version__=='0.24.0'" && \
            python2 -m pylint aisdk --rcfile=../pylintrc && \
            python2 -m yapf -i -r .
        ''')
