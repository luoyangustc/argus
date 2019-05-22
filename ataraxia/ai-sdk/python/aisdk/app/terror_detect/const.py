import aisdk.common.config
from aisdk.common.flavor import Flavor

app_name = 'terror_detect'
flavor = Flavor(app_name=app_name)
cfg = aisdk.common.config.load_config(app_name, flavor)
# if flavor.flavor == GPU_SHARE:
#     flavor.forward_num = 2
