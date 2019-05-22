package utility

import (
	"context"
	"encoding/json"
	"testing"
)

type mockOcrSariIdcard struct{}

func (mot mockOcrSariIdcard) SariIdcardPreProcessEval(
	ctx context.Context, req _EvalOcrSariIdcardPreProcessReq, env _EvalEnv,
) (ret _EvalOcrSariIdcardPreProcessResp, err error) {
	var str string
	if req.Params.Type == "predetect" {
		str = `
		{
			"code": 0,
			"message": "",
			"result": {	
				"alignedImg":"http://p9zv90cqq.bkt.clouddn.com/alignedimg/cut_001.jpg",
				"bboxes":[
					[[120,225],[120,270],[440,270],[440,225]],
					[[120,305],[120,350],[440,350],[440,305]],
					[[120,265],[120,310],[440,310],[440,265]],
					[[35,115],[35,155],[365,155],[365,115]],
					[[225,365],[225,415],[690,415],[690,365]],
					[[120,165],[120,210],[370,210],[370,165]],
					[[135,50],[135,100],[212,100],[212,50]]
				],
				"class":0,
				"names":["住址1","住址3","住址2","性民","公民身份号码","出生","姓名"],
				"regions":[
					[[120,225],[120,270],[440,270],[440,225]],
					[[120,305],[120,350],[440,350],[440,305]],
					[[120,265],[120,310],[440,310],[440,265]],
					[[35,115],[35,155],[365,155],[365,115]],
					[[225,365],[225,415],[690,415],[690,365]],
					[[120,165],[120,210],[370,210],[370,165]],
					[[135,50],[135,100],[212,100],[212,50]]
				]
			}
		}
	`
	} else if req.Params.Type == "prerecog" {
		str = `
		{
			"code": 0,
			"message": "",
			"result": {	
				"bboxes":[
					[[134,227],[419,227],[419,262],[134,262]],
					[[134,268],[235,268],[235,296],[134,296]],
					[[35,115],[35,155],[365,155],[365,115]],
					[[634,369],[635,402],[226,406],[225,373]],
					[[120,165],[120,210],[370,210],[370,165]],
					[[115,50],[115,100],[232,100],[232,50]]
				]
			}
		}
	`
	} else {
		str = `
		{
			"code": 0,
			"message": "",
			"result": {	
				"res":{
					"住址": "河南省项城市芙蓉巷东四胡同2号",
					"公民身份号码": "412702199705127504",
					"出生": "1997年5月12日",
					"姓名": "张杰",
					"性别": "女",
					"民族": "汉"
				}
			}
		}
	`
	}

	err = json.Unmarshal([]byte(str), &ret)
	return
}

func (mot mockOcrSariIdcard) SariIdcardDetectEval(
	ctx context.Context, req _EvalOcrSariIdcardDetectReq, env _EvalEnv,
) (ret _EvalOcrSariIdcardDetectResp, err error) {
	str := `
		{
			"code": 0,
			"message": "",
			"result": {	
				"bboxes":[
					[121,231,413,229,413,260,121,263],
					[72,173,346,176,345,205,72,202],
					[45,62,215,60,215,93,46,95],
					[243,374,619,371,620,404,244,407],
					[131,270,239,270,239,299,131,298],
					[46,374,205,373,206,403,46,404],
					[47,122,162,120,162,150,47,151],
					[206,121,302,121,302,148,206,148],
					[44,232,119,233,118,259,44,258]
				]
			}
		}
	`
	err = json.Unmarshal([]byte(str), &ret)
	return
}

func (mot mockOcrSariIdcard) SariIdcardRecognizeEval(
	ctx context.Context, req _EvalOcrSariIdcardRecogReq, env _EvalEnv,
) (ret _EvalOcrSariIdcardRecogResp, err error) {
	str := `
		{
			"code": 0,
			"message": "",
			"result": {	
				"bboxes":[
					[[134,227],[419,227],[419,262],[134,262]],
					[[134,268],[235,268],[235,296],[134,296]],
					[[35,115],[35,155],[365,155],[365,115]],
					[[634,369],[635,402],[226,406],[225,373]],
					[[120,165],[120,210],[370,210],[370,165]],
					[[115,50],[115,100],[232,100],[232,50]]
				],
				"text":[
					"河南省项城市芙蓉巷东四",
					"胡同2号",
					"性别‘女一民族汉",
					"412702199705127504",
					"1997年5月12日",
					"张杰"
				]
			}
		}
	`
	err = json.Unmarshal([]byte(str), &ret)
	return
}

func TestOcrSariIdcard(t *testing.T) {
	service, ctx := getMockContext(t)
	service.iOcrSariIdcard = mockOcrSariIdcard{}
	ctx.Exec(`
		post http://argus.ava.ai/v1/ocr/idcard
		auth |authstub -uid 1 -utype 4|
		json '{
			"data": {
				"uri": "http://p9zv90cqq.bkt.clouddn.com/001.jpg"  
			}   
		}'
		ret 200
		header Content-Type $(mime) 
		equal $(mime) 'application/json'
		echo $(resp.body)
		json '{
			"code": $(code),
			"message": $(msg),
			"result":{
				"uri": $(uri),
				"bboxes": $(bboxes),
				"type": $(type),
				"res": $(res)
			}	
		}'
		equal $(code) 0
		equal $(type) 0
	`)
}
