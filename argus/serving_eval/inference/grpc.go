package inference

import (
	"context"
	"encoding/json"
	"errors"
	"io/ioutil"
	"time"

	"github.com/qiniu/xlog.v1"
	"google.golang.org/grpc"
	"qiniu.com/argus/atserving/model"
	pb "qiniu.com/argus/serving_eval/inference/proto_go"
)

const INFERENCE_GRPC_ADDR = ":9009" // 对外暴露的推理API grpc协议

const waitSubprocessTimeout = 60 * time.Second

type grpcInstanceer struct {
}

func NewGrpc() Creator { return grpcInstanceer{} }

func (zmqI grpcInstanceer) Create(ctx context.Context, params *CreateParams) (Instance, error) {
	xl := xlog.FromContextSafe(ctx)
	s := &grpcInstance{}
	conn, err := grpc.Dial("127.0.0.1"+INFERENCE_GRPC_ADDR, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	// defer conn.Close()
	s.c = pb.NewInferenceClient(conn)
	xl.Info("wait subprocess...")
	start := time.Now()
	for {
		_, err := s.c.Ping(ctx, &pb.PingMsg{})
		if err == nil {
			return s, nil
		}
		if time.Since(start) > waitSubprocessTimeout {
			return nil, errors.New("wait subprocess timeout")
		}
		time.Sleep(time.Second / 10)
	}
}

type grpcInstance struct {
	c pb.InferenceClient
}

func (ii grpcInstance) Preprocess(
	ctx context.Context, req model.EvalRequestInner,
) (model.EvalRequestInner, error) {
	return req, nil
}

func (i *grpcInstance) inference(
	ctx context.Context, request *pb.InferenceRequest,
) (Response, error) {
	xl := xlog.FromContextSafe(ctx)
	resp, err := i.c.Inference(ctx, request)
	if err != nil {
		return Response{}, err
	}
	code, message := foramtCodeMessage(int(resp.Code), resp.Message)
	r := Response{Code: code, Message: message}
	if len(resp.Body) > 0 {
		r.Body = resp.Body
	} else if resp.Result != "" {
		err := json.Unmarshal([]byte(resp.Result), &r.Result)
		if err != nil {
			xl.Errorf("json.Unmarshal error:%v, result:%v", err, resp.Result)
			return r, err
		}
	}
	return r, nil
}

func (instance *grpcInstance) Inference(
	ctx context.Context, reqs []model.EvalRequestInner,
) ([]Response, error) {
	xl := xlog.FromContextSafe(ctx)
	resps := make([]Response, len(reqs))
	for i, req := range reqs {
		request := &pb.InferenceRequest{
			Reqid: xl.ReqId(),
			Data:  &pb.InferenceRequest_RequestData{},
		}
		{
			switch v := req.Data.URI.(type) {
			case model.BYTES:
				request.Data.Body = v
			case model.STRING:
				bs, _ := ioutil.ReadFile(v.String())
				request.Data.Body = bs
			}
			if req.Data.Attribute != nil {
				bs, _ := json.Marshal(req.Data.Attribute)
				request.Data.Attribute = string(bs)
			}
		}
		if req.Params != nil {
			bs, _ := json.Marshal(req.Params)
			request.Params = string(bs)
		}
		resp, err := instance.inference(ctx, request)
		if err != nil {
			return nil, err
		}
		resps[i] = resp
	}
	return resps, nil
}

func (ii grpcInstance) InferenceGroup(
	ctx context.Context, reqs []model.GroupEvalRequestInner,
) ([]Response, error) {
	return nil, nil
}

func (ii grpcInstance) PreprocessGroup(
	ctx context.Context, req model.GroupEvalRequestInner,
) (model.GroupEvalRequestInner, error) {
	return req, nil
}
