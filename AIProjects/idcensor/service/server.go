package idcensor

import (
	"context"
	"net/http"
	"strings"
	"time"

	"qiniu.com/argus/argus/com/util"
	"qiniu.com/argus/utility"
	"qiniu.com/auth/authstub.v1"
	"qiniupkg.com/http/httputil.v2"

	"github.com/qiniu/rpc.v1"
	"qiniu.com/auth/qiniumac.v1"
)

type Config struct {
	HanghuiAK      string `json:"hanghui_ak"`
	HanghuiSK      string `json:"hanghui_sk"`
	AK             string `json:"access_key"`
	SK             string `json:"secret_key"`
	ArgusHost      string `json:"argus_host"`
	ServingHost    string `json:"serving_host"`
	OfficialDbHost string `json:"offical_database_host"`
}

type Server struct {
	iFaceSim
	iDCardOcr
	iDdatabase
}

type IdCensorReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

type IdCensorResp struct {
	Code    int                 `json:"code"`
	Message string              `json:"message"`
	Result  *IdCensorRespResult `json:"result,omitempty"`
}

type IdCensorRespResult struct {
	Id         string  `json:"id"`
	Similarity float32 `json:"similarity"`
	Same       bool    `json:"same"`
}

func NewServer(cf Config) *Server {
	return &Server{
		iFaceSim:   newFaceSim(cf.ArgusHost, newQiniuAuthClient(cf.AK, cf.SK, 10*time.Second)),
		iDCardOcr:  newIdCardOcr(cf.ServingHost, newQiniuAuthClient(cf.AK, cf.SK, 10*time.Second)),
		iDdatabase: newIdDatabase(cf.OfficialDbHost, New(&AccessKey{cf.HanghuiAK, []byte(cf.HanghuiSK)}), 35*time.Second),
	}
}

func newQiniuAuthClient(ak, sk string, timeout time.Duration) *rpc.Client {
	return &rpc.Client{
		Client: &http.Client{
			Timeout: timeout,
			Transport: qiniumac.NewTransport(
				&qiniumac.Mac{AccessKey: ak, SecretKey: []byte(sk)},
				http.DefaultTransport,
			),
		},
	}
}

func (s Server) PostIdCensor(ctx context.Context, args *IdCensorReq, env *authstub.Env) {

	var (
		ctex, xl = util.CtxAndLog(ctx, env.W, env.Req)
		resp     IdCensorResp
	)

	xl.Infof("args:%v", args)
	if strings.TrimSpace(args.Data.URI) == "" {
		resp.Code = 3
		resp.Message = "invalid uri"
		httputil.Reply(env.W, http.StatusNotAcceptable, resp)
		return
	}

	cardInfo, err := s.iDCardOcr.Eval(ctex, &_EvalIDcardReq{
		Data: struct {
			URI string `json:"uri"`
		}{
			URI: args.Data.URI,
		},
	})
	if err != nil {
		xl.Errorf("query iDCardOcr error:%v", err)
		resp.Code = 1
		resp.Message = "no valid id info obtained"
		httputil.Reply(env.W, http.StatusInternalServerError, resp)
		return
	}

	xl.Infof("id card ocr result:%v", cardInfo)
	if len(strings.TrimSpace(cardInfo.Result.IdNumber)) < 15 { //身份证最少15位
		resp.Code = 1
		resp.Message = "no valid id info obtained"
		resp.Result = &IdCensorRespResult{
			Id: cardInfo.Result.IdNumber,
		}
		httputil.Reply(env.W, http.StatusInternalServerError, resp)
		return
	}

	card, err := s.iDdatabase.Eval(ctex, cardInfo.Result.IdNumber)
	if err != nil {
		xl.Errorf("query official idcard database error:%v", err)
		resp.Code = 2
		resp.Message = "not found in id database"
		resp.Result = &IdCensorRespResult{
			Id: cardInfo.Result.IdNumber,
		}
		httputil.Reply(env.W, http.StatusNotFound, resp)
		return
	}
	if strings.TrimSpace(card.Face) == "" {
		xl.Errorf("query official idcard database get invalid face:%v", card)
		resp.Code = 3
		resp.Message = "query official idcard database get invalid face"
		resp.Result = &IdCensorRespResult{
			Id: cardInfo.Result.IdNumber,
		}
		httputil.Reply(env.W, http.StatusInternalServerError, resp)
		return
	}

	cardUrl := "data:application/octet-stream;base64," + card.Face
	simResp, err := s.iFaceSim.Eval(ctx, &utility.FaceSimReq{
		Data: []struct {
			URI string `json:"uri"`
		}{
			{
				URI: args.Data.URI,
			},
			{
				URI: cardUrl,
			},
		},
	})

	if err != nil {
		xl.Errorf("query facex sim error:%v", err)
		resp.Code = 3
		resp.Message = err.Error()
		resp.Result = &IdCensorRespResult{
			Id: cardInfo.Result.IdNumber,
		}
		httputil.Reply(env.W, http.StatusInternalServerError, resp)
		return
	}

	xl.Infof("similiarity resp :%v", simResp)

	resp.Message = "success"
	resp.Result = &IdCensorRespResult{
		Id:         cardInfo.Result.IdNumber,
		Similarity: simResp.Result.Similarity,
		Same:       simResp.Result.Same,
	}
	httputil.Reply(env.W, http.StatusOK, resp)
	return
}
