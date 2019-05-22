import jinja2
import json
import codecs


def render_docker_file(app_name,
                       out_dockerfile_path='./res/tmp_dockerfile',
                       copy_src=True):
    docker_file_path = 'python/aisdk/app/{}/Dockerfile'.format(app_name)
    content = codecs.open(docker_file_path, 'r', 'utf8').read()
    content = content.replace('#{%', '{%')
    cfg = json.loads(
        open('python/aisdk/app/{}/config.json'.format(app_name)).read())
    while True:
        template = jinja2.Template(content)
        out = template.render(
            cfg=cfg, app_name=cfg['app_name'], copy_src=copy_src)
        if out == content:
            break
        else:
            content = out
    with codecs.open(out_dockerfile_path, 'w', 'utf8') as f:
        f.write(out)


if __name__ == '__main__':
    render_docker_file('pulp')
