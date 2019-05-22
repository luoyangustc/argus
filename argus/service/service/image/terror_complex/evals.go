package terror_complex

import (
	"qiniu.com/argus/service/service/biz"
)

//----------------------------------------------------------------------------//
var EVAL_TERROR_DET_CONFIG = biz.EvalModelConfig{
	Image: "reg.qiniu.com/avaprd/aisdk-terror_detect_complex:20190214-v209-CENSORv3.3.3",
	Type:  biz.EvalRunTypeSDK,
}

//----------------------------------------------------------------------------//
var EVAL_TERROR_MIXUP_CONFIG = biz.EvalModelConfig{
	Image: "reg.qiniu.com/avaprd/aisdk-terror_mixup_complex:20190214-v209-CENSORv3.3.3",
	Type:  biz.EvalRunTypeSDK,
}
