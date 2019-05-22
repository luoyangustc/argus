import aisdk.common.config
from aisdk.common.flavor import Flavor, GPU_SHARE
from aisdk.common.logger import log

app_name = 'terror_wangan_mixup'
flavor = Flavor(app_name=app_name)
cfg = aisdk.common.config.load_config(app_name, flavor)
if flavor.flavor == GPU_SHARE:
    log.info("flavor GPU_SHARE, change batch_size %s -> %s", cfg['batch_size'],
             4)
    cfg['batch_size'] = 4
