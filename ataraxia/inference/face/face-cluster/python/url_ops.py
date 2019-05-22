import urllib


def download_url(url, save=False):
    """Copy the contents of a file from a given URL
    to a local file.
    """
    web_file = urllib.urlopen(url)
    content = web_file.read()
    web_file.close()

    if save:
        local_file = open(url.split('/')[-1], 'w')
        local_file.write()
        local_file.close()

    return content
