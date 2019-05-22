// +build zmq

package inference

import (
	"context"
	"encoding/json"
	"errors"
	"io/ioutil"
	"math/rand"
	"sync"
	"time"

	"github.com/gogo/protobuf/proto"

	zmq "github.com/pebbe/zmq4"
	"github.com/qiniu/xlog.v1"
	"google.golang.org/grpc"
	"qiniu.com/argus/atserving/model"
	lib "qiniu.com/argus/serving_eval/inference/proto_go" // TODO:pb
	pb "qiniu.com/argus/serving_eval/inference/proto_go"
)

const INFERENCE_ZMQ_ADDR = "tcp://127.0.0.1:9655" // 对外暴露的推理API zmq协议
const zmqThreadNum = 200

type zmqInstanceer struct {
}

func NewZmq() Creator { return zmqInstanceer{} }

func (zmqI zmqInstanceer) Create(ctx context.Context, params *CreateParams) (Instance, error) {
	xl := xlog.FromContextSafe(ctx)
	ist := &zmqInstance{}
	for i := 0; i < zmqThreadNum; i++ {
		socket, err := zmq.NewSocket(zmq.REQ)
		if err != nil {
			xl.Panicln(err)
		}
		err = socket.Connect(INFERENCE_ZMQ_ADDR)
		if err != nil {
			xl.Panicln(err)
		}
		ist.socksts = append(ist.socksts, socket)
		ist.muxs = append(ist.muxs, sync.Mutex{})
	}

	{
		xl.Info("wait subprocess...")
		conn, err := grpc.Dial("127.0.0.1"+INFERENCE_GRPC_ADDR, grpc.WithInsecure())
		if err != nil {
			return nil, err
		}
		// defer conn.Close()
		c := pb.NewInferenceClient(conn)
		start := time.Now()
		for {
			_, err := c.Ping(ctx, &pb.PingMsg{})
			if err == nil {
				return ist, nil
			}
			if time.Since(start) > waitSubprocessTimeout {
				return nil, errors.New("wait subprocess timeout")
			}
			time.Sleep(time.Second / 10)
		}
	}
	return ist, nil
}

type zmqInstance struct {
	socksts []*zmq.Socket
	muxs    []sync.Mutex
}

func (ii zmqInstance) Preprocess(
	ctx context.Context, req model.EvalRequestInner,
) (model.EvalRequestInner, error) {
	return req, nil
}

func (i *zmqInstance) inference(
	ctx context.Context, request *lib.InferenceRequest,
) (Response, error) {
	xl := xlog.FromContextSafe(ctx)
	n := rand.Intn(zmqThreadNum)
	i.muxs[n].Lock()
	defer i.muxs[n].Unlock()
	buf, err := proto.Marshal(request)
	if err != nil {
		xl.Panic(err)
	}
	_, err = i.socksts[n].SendBytes(buf, 0)
	if err != nil {
		xl.Warn(err)
		return Response{}, err
	}
	buf, err = i.socksts[n].RecvBytes(0)
	if err != nil {
		return Response{}, err
	}
	var resp lib.InferenceResponse
	err = proto.Unmarshal(buf, &resp)
	if err != nil {
		xl.Panic(err)
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

func (instance *zmqInstance) Inference(
	ctx context.Context, reqs []model.EvalRequestInner,
) ([]Response, error) {
	xl := xlog.FromContextSafe(ctx)
	resps := make([]Response, len(reqs))
	for i, req := range reqs {
		request := &lib.InferenceRequest{
			Reqid: xl.ReqId(),
			Data:  &lib.InferenceRequest_RequestData{},
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

func (ii zmqInstance) InferenceGroup(
	ctx context.Context, reqs []model.GroupEvalRequestInner,
) ([]Response, error) {
	return nil, nil
}

func (ii zmqInstance) PreprocessGroup(
	ctx context.Context, req model.GroupEvalRequestInner,
) (model.GroupEvalRequestInner, error) {
	return req, nil
}
