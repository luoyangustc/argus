package idcensor

import (
	"context"
	"encoding/json"
	"testing"

	"qiniu.com/argus/utility"

	"github.com/qiniu/http/restrpc.v1"
	"qiniupkg.com/qiniutest/httptest.v1"
	"qiniupkg.com/x/mockhttp.v7"
	"qiniupkg.com/x/xlog.v7"
)

type tLog struct {
	t *testing.T
}

func (t *tLog) Write(p []byte) (n int, err error) {
	t.t.Log(string(p))
	return len(p), nil
}

func getMockContext(t *testing.T) (*Server, httptest.Context) {
	xlog.SetOutputLevel(0)
	xlog.SetOutput(&tLog{t: t})

	srv := NewServer(Config{})

	router := restrpc.Router{
		PatternPrefix: "v1",
	}

	transport := mockhttp.NewTransport()
	transport.ListenAndServe("aiproject.idcensor.com", router.Register(srv))

	ctx := httptest.New(t)
	ctx.SetTransport(transport)
	return srv, ctx
}

type mockFaceSim struct {
}

func (f mockFaceSim) Eval(ctx context.Context, req *utility.FaceSimReq) (resp utility.FaceSimResp, err error) {

	ret := `
	{
		"code": 0,
		"message": "success",
		"result": {
			"faces":[{
					  "score": 0.987,
					  "pts": [[225,195], [351,195], [351,389], [225,389]]
					  },
					  {
					  "score": 0.997,
					  "pts": [[225,195], [351,195], [351,389], [225,389]]
					  }], 
			"similarity": 0.87,
			"same": true 
		}   
	}
	`
	json.Unmarshal([]byte(ret), &resp)
	return
}

type mockIdCardOcr struct {
}

func (d mockIdCardOcr) Eval(ctx context.Context, req *_EvalIDcardReq) (resp _EvalIDcardResp, err error) {
	ret := `
	{
		"code": 0,
		"message": "",
		"result": {
			"name": "田淼淼",
			"people": "汉",
			"sex": "女",
			"address": "陕西省高碑店市庄发镇",
			"id_number": "1356648999203243269"    
		}
	}
	`
	json.Unmarshal([]byte(ret), &resp)
	return
}

type mockIdDatabase struct {
}

func (c mockIdDatabase) Eval(ctx context.Context, id string) (rep _IdDatabaeResp, err error) {
	rep.Face = "base64 face data"
	return rep, nil
}

func TestServer(t *testing.T) {

	service, ctx := getMockContext(t)
	service.iDCardOcr = mockIdCardOcr{}
	service.iDdatabase = mockIdDatabase{}
	service.iFaceSim = mockFaceSim{}

	ctx.Exec(`
		post http://aiproject.idcensor.com/v1/id/censor
		auth |authstub -uid 1 -utype 4|
		json '{
			"data": {
				"uri":"http://test.image1.jpg"  
			}
		}'
		ret 200
		header Content-Type $(mime) 
		equal $(mime) 'application/json'
		echo $(resp.body)
		json '{
			"code": $(code),
			"result":{
					   	"similarity":$(sim),
					   	"same":$(same)
			}	
		}'
		equal $(code) 0
		equal $(sim) 0.87
		equal $(same) true
	`)
}
