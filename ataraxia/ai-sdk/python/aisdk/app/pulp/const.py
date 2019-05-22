import aisdk.common.config
from aisdk.common.flavor import Flavor

app_name = 'pulp'
flavor = Flavor(app_name=app_name)
cfg = aisdk.common.config.load_config(app_name, flavor)
