package main

import (
	"context"
	"math/rand"
	"net"
	"sync"

	"github.com/gogo/protobuf/proto"
	zmq "github.com/pebbe/zmq4"
	xlog "github.com/qiniu/xlog.v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
	pb "qiniu.com/ai-sdk/proto"
)

const zmqThreadNum = 200

type server struct {
	socksts []*zmq.Socket
	muxs    []sync.Mutex
}

func newGrpcServer(ctx context.Context) *server {
	s := &server{}

	xl := xlog.FromContextSafe(ctx)
	frontend, err := zmq.NewSocket(zmq.ROUTER)
	if err != nil {
		xl.Panicln(err)
	}
	err = frontend.Bind(INFERENCE_ZMQ_ADDR)
	if err != nil {
		xl.Panicln(err)
	}
	backend, err := zmq.NewSocket(zmq.DEALER)
	if err != nil {
		xl.Panicln(err)
	}
	err = backend.Bind(INFERENCE_ZMQ_IN)
	if err != nil {
		xl.Panicln(err)
	}
	go func() {
		for {
			err = zmq.Proxy(frontend, backend, nil)
			if err != nil {
				xl.Warn("zmq proxy", err)
			}
		}
	}()
	for i := 0; i < zmqThreadNum; i++ {
		socket, err := zmq.NewSocket(zmq.REQ)
		if err != nil {
			xl.Panicln(err)
		}
		err = socket.Connect(INFERENCE_ZMQ_ADDR)
		if err != nil {
			xl.Panicln(err)
		}
		s.socksts = append(s.socksts, socket)
		s.muxs = append(s.muxs, sync.Mutex{})
	}
	return s
}

func (s *server) Inference(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	xl := xlog.FromContextSafe(ctx)
	n := rand.Intn(zmqThreadNum)
	s.muxs[n].Lock()
	defer s.muxs[n].Unlock()
	buf, err := proto.Marshal(req)
	if err != nil {
		xl.Panic(err)
	}
	_, err = s.socksts[n].SendBytes(buf, 0)
	if err != nil {
		xl.Warn(err)
		return nil, err
	}
	buf, err = s.socksts[n].RecvBytes(0)
	if err != nil {
		xl.Warn(err)
		return nil, err
	}
	var r pb.InferenceResponse
	err = proto.Unmarshal(buf, &r)
	if err != nil {
		xl.Panic(err)
	}
	return &r, nil
}

func (s *server) Ping(ctx context.Context, req *pb.PingMsg) (*pb.PingMsg, error) {
	globalSubprocessStatus.waitStart()
	return &pb.PingMsg{}, nil
}

func runGrpcServer(ctx context.Context, port string) {
	lis, err := net.Listen("tcp", port)
	if err != nil {
		xl.Fatalf("failed to listen: %v", err)
	}
	globalSubprocessStatus.waitStart()
	s := grpc.NewServer(grpc.MaxRecvMsgSize(16 * 1024 * 1024))
	pb.RegisterInferenceServer(s, newGrpcServer(ctx))
	reflection.Register(s)
	if err := s.Serve(lis); err != nil {
		xl.Fatalf("failed to serve: %v", err)
	}
}
