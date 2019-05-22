package image_sync

import (
	"context"
	"github.com/go-kit/kit/endpoint"
	"qiniu.com/argus/service/middleware"
	sbiz "qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/image_sync"
	"qiniu.com/argus/service/service/biz"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/image/ocrvat"
	"qiniu.com/argus/service/transport"
)

const (
	VERSION = "1.0.0"
)

// eval 统一名称
const (
	eodName = "ocr-detect"
	eorName = "ocr-recog"
	eopName = "ocr-post"
)

var (
	_DOC_OCRVAT = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{"增值税发票识别"},
		Request: `POST /v1/ocrvat  Http/1.1
Content-Type:application/json
{
	"data":{
		"uri": "http://pbqb5ctvq.bkt.clouddn.com/YBZZS_00212060.jpg"
	}
}`,
		Response: `200 ok
Content-Type:application/json
{
    "code":0,
    "message":"",
    "result":{
        "uri":"",
        "bboxes":[
            [[636.22986,1298.9965],[636.18,1259.7034],[1042.18,1257.2982],[1042.2822,1296.5103]],
            [[2200.1765,315.89276],[2199.8635,267.2329],[2507.742,266.9532],[2508.1038,315.53653]],
            [[1986.4376,188.0904],[1985.9543,119.5574],[2355.685,115.03136],[2357.2246,183.43137]],
            [[1840.6908,1129.464],[1839.4836,1086.4576],[2106.2126,1079.1105],[2107.4607,1123.0555]],
            [[2353.4075,192.70236],[2352.1582,153.04263],[2552.5781,151.98686],[2553.8438,190.61514]],
            [[636.377,1414.9198],[637.32605,1378.6147],[1253.352,1375.5996],[1252.4856,1412.7965]],
            [[1256.4283,1408.749],[1256.3353,1380.6053],[1514.0957,1378.9154],[1514.2125,1407.022]],
            [[2139.249,722.8691],[2139.0383,689.0018],[2217.232,688.7626],[2217.4512,722.61633]],
            [[2137.5967,771.69794],[2137.3862,737.8194],[2217.548,737.55426],[2217.7676,771.4189]],
            [[2139.863,821.52576],[2139.646,786.6398],[2217.8643,786.3616],[2218.0903,821.2336]],
            [[487.24158,299.04495],[487.07233,218.92404],[705.2455,218.48547],[705.47205,298.5178]],
            [[843.813,729.0614],[843.7365,689.9566],[1069.3715,689.2698],[1069.4768,728.3298]],
            [[843.9091,778.20734],[843.83264,739.0899],[1049.7288,738.4119],[1049.8317,777.48846]],
            [[844.9984,828.36896],[843.9287,788.2391],[1068.634,782.4329],[1069.7311,822.51575]],
            [[637.44543,1472.4033],[638.3945,1436.0844],[1500.5052,1432.216],[1499.6671,1468.3761]],
            [[257.4534,733.9798],[257.45242,692.74677],[524.19653,691.9335],[524.2337,733.11084]],
            [[257.45456,782.2669],[257.45355,740.0149],[523.245,739.141],[523.28296,781.33624]],
            [[258.45282,829.5592],[258.45166,787.2942],[524.2818,786.35675],[524.3199,828.5649]],
            [[887.43274,1215.9641],[887.3147,1159.6487],[1316.8448,1157.3186],[1317.042,1213.5111]],
            [[1195.084,722.76733],[1194.9946,694.0415],[1237.1412,693.9121],[1237.2346,722.6318]],
            [[1195.2336,770.82263],[1195.1442,742.0877],[1237.2975,741.94806],[1237.3909,770.67676]],
            [[1195.5309,818.7424],[1195.4392,789.2974],[1238.29,789.14514],[1238.3859,818.5837]],
            [[1404.0435,725.2452],[1403.918,692.25037],[1513.2097,691.91547],[1513.3469,724.8919]],
            [[1398.3132,773.27094],[1398.1884,740.2647],[1512.425,739.88684],[1512.5623,772.87384]],
            [[1398.4948,821.29285],[1398.37,788.2762],[1512.6245,787.8703],[1512.7617,820.8678]],
            [[1540.8955,723.80365],[1540.7596,691.83105],[1753.0283,691.1806],[1753.1865,723.1186]],
            [[1540.1112,770.77893],[1540.955,737.79376],[1753.2655,739.0901],[1752.4468,772.0401]],
            [[1541.3036,819.7617],[1541.1632,786.7693],[1753.4979,786.0162],[1753.6611,818.9729]],
            [[636.2602,541.12286],[636.20917,501.0232],[1492.7369,499.2377],[1492.9005,539.1633]],
            [[2425.3677,722.93915],[2425.111,687.1321],[2561.5098,686.7155],[2561.7822,722.4976]],
            [[2412.056,770.7426],[2411.8079,735.9167],[2562.8542,735.4178],[2563.1194,770.21674]],
            [[2412.403,819.5123],[2412.155,784.6752],[2562.2517,784.142],[2562.5166,818.9524]],
            [[689.02795,1358.1205],[689.9671,1319.8239],[1312.4679,1317.9543],[1311.613,1356.1302]],
            [[637.0953,417.8485],[637.0467,379.7847],[1037.9175,379.195],[1038.0161,417.18152]],
            [[1954.2542,724.4641],[1954.0638,690.56464],[2099.9243,690.1177],[2100.1306,723.99194]],
            [[1939.8285,773.3839],[1939.6342,738.4736],[2100.2158,737.9424],[2100.4282,772.8241]],
            [[1953.8228,822.2206],[1953.6268,787.30145],[2100.5134,786.779],[2100.726,821.67194]],
            [[1209.6729,331.23962],[1208.562,83.8552],[1590.0471,83.347305],[1591.4661,330.25153]],
            [[673.8756,478.89853],[673.81964,438.8229],[1292.6425,437.72754],[1292.7798,477.67743]],
            [[1626.1829,1532.7478],[1623.9894,1483.5657],[1765.7894,1479.5505],[1767.9978,1527.6926]],
            [[637.3344,605.3018],[637.28326,565.1854],[1308.8474,563.5674],[1308.9869,603.5473]],
            [[2325.6948,1127.8928],[2323.4404,1084],[2562.523,1075.8107],[2564.8064,1119.6478]],
            [[2102.1006,1208.0287],[2100.8591,1165.0575],[2400.159,1157.4287],[2401.4392,1200.3325]],
            [[1237.3478,347.14008],[1236.8358,235.37212],[1553.6014,234.70648],[1554.2291,346.29425]],
            [[388.41345,337.28613],[388.39264,291.67783],[465.59818,291.59875],[465.63058,337.18927]],
            [[450.1772,1539.3624],[448.1513,1491.9053],[589.6696,1485.8558],[591.71466,1533.2773]],
            [[164.18272,501.6468],[209.03485,500.4882],[220.03546,1343.5237],[175.0637,1344.8799]],
            [[1657.5211,428.11096],[1656.3406,385.2602],[2467.704,380.06863],[2468.99,422.74127]],
            [[1656.7659,476.95932],[1655.5853,434.09476],[2464.172,429.6978],[2465.4578,472.385]],
            [[1655.0144,522.834],[1655.7932,478.95532],[2468.4302,479.32965],[2467.7683,522.03406]],
            [[1656.223,571.7092],[1656.0243,528.816],[2466.8306,527.0024],[2467.1428,569.71844]],
            [[2322.5852,244.30064],[2322.2935,201.6452],[2546.1494,201.51663],[2546.4724,244.1233]],
            [[496.96536,202.88577],[496.8301,139.83011],[920.6704,136.1522],[921.8816,199.06929]],
            [[1049.14,158.8999],[1047.8408,80.079124],[1750.8055,76.17568],[1752.2784,154.71338]]
        ],
        "res":{
            "_BeiZhu":"",
            "_DaLeiMingCheng":"",
            "_DaiKaiBiaoShi":"",
            "_DaiKaiJiGuanDiZhiJiDianHua":"",
            "_DaiKaiJiGuanGaiZhang":"",
            "_DaiKaiJiGuanHaoMa":"",
            "_DaiKaiJiGuanMingCheng":"",
            "_DanZhengMingCheng":"四川增值税专用发票",
            "_FaPiaoDaiMa_DaYin":"5100153130",
            "_FaPiaoDaiMa_YinShua":"5100153130",
            "_FaPiaoHaoMa_DaYin":"00212060",
            "_FaPiaoHaoMa_YinShua":"00212050",
            "_FaPiaoJianZhiZhang":"",
            "_FaPiaoLianCi":"抵扣联",
            "_FaPiaoYinZhiPiHanJiYinZhiGongSi":"",
            "_FuHeRen":"",
            "_GaiZhangDanWeiMingCheng":"",
            "_GaiZhangDanWeiShuiHao":"",
            "_GouMaiFangDiZhiJiDianHua":"四川省绵阳科创园区园艺东街8号0816-2536680",
            "_GouMaiFangKaiHuHangJiZhangHao":"中国银行绵阳涪城支行123912372612",
            "_GouMaiFangMingCheng":"四川光发科技有限公司",
            "_GouMaiFangNaShuiShiBieHao":"91510700MA62474D2N",
            "_HeJiJinE_BuHanShui":"￥20256.41",
            "_HeJiShuiE":"￥3443.59",
            "_JiQiBianHao":"gU",
            "_JiaShuiHeJi_DaXie":"ⓧ贰万叁仟柒佰圆整",
            "_JiaShuiHeJi_XiaoXie":"￥23700.00",
            "_JiaoYanMa":"",
            "_KaiPiaoRen":"皮广元",
            "_KaiPiaoRiQi":"2016年6月27日",
            "_MiMa":"18639+31/054*+4404>9523-79668>2>0>5*4>08*14+2637-6-/4//3<<+-*/+*2659*+8145<1530+6147+/2998/548-495884>+2",
            "_ShouKuanRen":"皮广元",
            "_WanShuiPingZhengHao":"",
            "_XiaoLeiMingCheng":"一般增值税发票",
            "_XiaoShouDanWeiGaiZhangLeiXing":"",
            "_XiaoShouFangDiZhiJiDianHua":"成都市高新区高新大道创业路14-16号 028-87492600",
            "_XiaoShouFangKaiHuHangJiZhangHao":"中国银行股份有限公司成都晋阳分理处125263950141",
            "_XiaoShouFangMingCheng":"成都金睿通信有限公司",
            "_XiaoShouFangNaShuiRenShiBieHao":"9151010057461436xK",
            "_XiaoShouMingXi":[
                ["光纤跳线散件","FC/APC-3.0","套","10000","0.3846153846","3846.15","17%","653.85"],
                ["光纤跳线散件","FC/PC-3.0","套","30000","0.3418803419","10256.41","17%","1743.59"],
                ["光纤跳线散件","FC/APC-0.9","套","20000","0.3076923077","6153.85","17%","1046.15"]
            ]
        }
    }
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "uri", Type: "string", Desc: "图片资源地址"},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "0:表示处理成功；不为0:表示出错"},
			{Name: "message", Type: "string", Desc: "描述结果或出错信息"},
			{Name: "bboxes", Type: "string", Desc: "文本框"},
			{Name: "res", Type: "float32", Desc: "结构化字段信息"},
		},
		ErrorMessage: []scenario.APIDocError{
			{Code: 4000801, Desc: "未检测到场景信息"},
		},
	}
)

func Import(serviceID string) func(interface{}) {
	return func(is interface{}) {
		Init(is.(scenario.ImageServer), serviceID)
	}
}

func Init(is scenario.ImageServer, serviceID string) {
	var eodSet sbiz.ServiceEvalSetter
	var eorSet sbiz.ServiceEvalSetter
	var eopSet sbiz.ServiceEvalSetter

	var eod ocrvat.EvalOcrSariVatDetectService
	var eor ocrvat.EvalOcrSariVatRecogService
	var eop ocrvat.EvalOcrSariVatPostProcessService

	set := is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "ocrvat", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				{
					eod1, _ := eodSet.Gen()
					eod = eod1.(ocrvat.EvalOcrSariVatDetectService)

					eor2, _ := eorSet.Gen()
					eor = eor2.(ocrvat.EvalOcrSariVatRecogService)

					eop3, _ := eopSet.Gen()
					eop = eop3.(ocrvat.EvalOcrSariVatPostProcessService)
				}
				s, _ := ocrvat.NewOcrSariVatService(eod, eor, eop)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return ocrvat.OcrSariVatEndpoints{} },
		}).
		UpdateRouter(func(
			sf func() middleware.Service,
			imageParser func() pimage.IImageParse,
			path func(string) *scenario.ServiceRoute,
		) error {
			endp := func() ocrvat.OcrSariVatEndpoints {
				svc := sf()
				endp, ok := svc.(ocrvat.OcrSariVatEndpoints)
				if !ok {
					svc, _ = middleware.MakeMiddleware(svc, ocrvat.OcrSariVatEndpoints{}, nil, nil)
					endp = svc.(ocrvat.OcrSariVatEndpoints)
				}
				return endp
			}

			type Req struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
			}

			path("/v1/ocrvat").Doc(_DOC_OCRVAT).Route().
				Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req0 interface{}) (interface{}, error) {
					req1, _ := req0.(Req)
					var req2 ocrvat.OcrSariVatReq
					var err error
					req2.Data.IMG, err = imageParser().ParseImage(ctx, req1.Data.URI)
					if err != nil {
						return nil, err
					}
					return endp().OcrSariVatEP(ctx, req2)
				},
				Req{}))
			return nil
		})

	eodSet = set.NewEval(
		sbiz.ServiceEvalInfo{Name: eodName, Version: "1.0.0"},
		EvalOcrSariVatDetectClient,
		func() middleware.ServiceEndpoints { return ocrvat.EvalOcrSariVatDetectEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-eval-ataraxia-ocr-sari-invoice-vat:201808051555--201808051608-v335-dev",
		Args: &biz.ModelConfigArgs{
			BatchSize:  1,
			ImageWidth: 225,
		},
		Type: biz.EvalRunTypeServing,
	})

	eorSet = set.NewEval(
		sbiz.ServiceEvalInfo{Name: eorName, Version: "1.0.0"},
		EvalOcrSariVatRecogClient,
		func() middleware.ServiceEndpoints { return ocrvat.EvalOcrSariVatRecogEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-eval-ataraxia-ocr-sari-crann:201808051919--201808052143-v336-dev",
		Model: "ava-ocr-sari-crann/ocr-sari-crann-20180607.tar",
		Args: &biz.ModelConfigArgs{
			BatchSize:  1,
			ImageWidth: 225,
		},
		Type: biz.EvalRunTypeServing,
	})

	eopSet = set.NewEval(
		sbiz.ServiceEvalInfo{Name: eopName, Version: "1.0.0"},
		EvalOcrSariVatPostProcessClient,
		func() middleware.ServiceEndpoints { return ocrvat.EvalOcrSariVatPostProcessEndpoints{} },
	).SetModel(biz.EvalModelConfig{
		Image: "hub2.qiniu.com/1381102897/ava-eval-ataraxia-ocr-sari-invoice-vat:201808051555--201808051608-v335-dev",
		Args: &biz.ModelConfigArgs{
			BatchSize:  1,
			ImageWidth: 225,
		},
		Type: biz.EvalRunTypeServing,
	})

	// 添加一种1卡的部署方式
	set.AddEvalsDeployModeOnGPU("", [][]sbiz.ServiceEvalDeployProcess{
		{
			{Name: eorName, Num: 1}, // ctpn 独占 GPU0
		},
	})
}

type DetectReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Type string `json:"type"`
	} `json:"params"`
}

type RecogReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Bboxes [][4][2]float32 `json:"bboxes"`
	} `json:"params"`
}

type PostProcessReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Type  string   `json:"type"`
		Texts []string `json:"texts"`
	} `json:"params"`
}

type EvalOcrSariVatPostProcessResp struct {
	Code    int                    `json:"code"`
	Message string                 `json:"message"`
	Result  map[string]interface{} `json:"result"`
}

func EvalOcrSariVatDetectClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ocrvat.EvalOcrSariVatDetectResp{})
	return ocrvat.EvalOcrSariVatDetectEndpoints{
		EvalOcrSariVatDetectEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ocrvat.EvalOcrSariVatDetectReq)
			var req2 DetectReq
			req2.Data.URI = string(req1.Data.IMG.URI)
			req2.Params.Type = req1.Params.Type
			return end(ctx, req2)
		},
	}
}

func EvalOcrSariVatRecogClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ocrvat.EvalOcrSariVatRecogResp{})
	return ocrvat.EvalOcrSariVatRecogEndpoints{
		EvalOcrSariVatRecogEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ocrvat.EvalOcrSariVatRecogReq)
			var req2 RecogReq
			req2.Data.URI = string(req1.Data.IMG.URI)
			req2.Params.Bboxes = req1.Params.Bboxes
			return end(ctx, req2)
		},
	}
}

func EvalOcrSariVatPostProcessClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ocrvat.EvalOcrSariVatPostProcessResp{})
	return ocrvat.EvalOcrSariVatPostProcessEndpoints{
		EvalOcrSariVatPostProcessEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ocrvat.EvalOcrSariVatPostProcessReq)
			var req2 PostProcessReq
			req2.Data.URI = string(req1.Data.IMG.URI)
			req2.Params.Type = req1.Params.Type
			req2.Params.Texts = req1.Params.Texts
			return end(ctx, req2)
		},
	}
}
