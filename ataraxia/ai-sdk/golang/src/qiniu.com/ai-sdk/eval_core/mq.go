package main

import (
	"context"
	"sync"

	"github.com/gogo/protobuf/proto"
	"github.com/pborman/uuid"
	zmq "github.com/pebbe/zmq4"
	pb "qiniu.com/ai-sdk/proto"
)

var networkIn = make(chan *pb.ForwardMsg, 100)

var m = sync.Map{}

const forwardInDialer = "tcp://127.0.0.1:9302"

func evalRepLoop() {
	reqer, err := zmq.NewSocket(zmq.REP)
	ce(err)
	ce(reqer.Connect(forwardInDialer))
	for {
		buf, err := reqer.RecvBytes(0)
		if err != nil {
			xl.Warn("reqer recv", err)
			continue
		}
		var in pb.ForwardMsgs
		ce(proto.Unmarshal(buf, &in))
		waitDones := make([]chan *pb.ForwardMsg, len(in.Msgs))
		for i := range in.Msgs {
			u := uuid.NewRandom().String()
			in.Msgs[i].Uuid = u
			xl.Debugf("new req %#v", u)
			waitDone := make(chan *pb.ForwardMsg, 1)
			m.Store(u, waitDone)
			waitDones[i] = waitDone
			networkIn <- in.Msgs[i]
		}
		r := &pb.ForwardMsgs{
			Msgs: make([]*pb.ForwardMsg, len(in.Msgs)),
		}
		for i := range in.Msgs {
			r.Msgs[i] = <-waitDones[i]
		}
		buf, err = proto.Marshal(r)
		ce(err)
		_, err = reqer.SendBytes(buf, 0)
		if err != nil {
			xl.Warn("reqer send", err)
			continue
		}
	}
}

func runBatchMq(ctx context.Context) {
	go func() {
		subscriber, err := zmq.NewSocket(zmq.PUSH)
		ce(err)
		defer subscriber.Close()
		ce(subscriber.Bind(FORWARD_IN))
		for {
			b := <-networkIn
			buf, err := proto.Marshal(b)
			ce(err)
			xl.Debug("send 9201", len(buf))
			n, err := subscriber.SendBytes(buf, 0)
			if err != nil {
				xl.Warn("send 9201", err, n)
			}
		}
	}()

	go func() {
		subscriber, err := zmq.NewSocket(zmq.PULL)
		ce(err)
		defer subscriber.Close()
		ce(subscriber.Bind(FORWARD_OUT))
		for {
			buf, err := subscriber.RecvBytes(0)
			xl.Debug("recv", len(buf))
			if err != nil {
				xl.Warn("9202 recv err", err)
			} else {
				processNetworkOut(buf)
			}
		}
	}()

	for i := 0; i < 100; i++ {
		go evalRepLoop()
	}
	frontend, err := zmq.NewSocket(zmq.ROUTER)
	if err != nil {
		xl.Panicln(err)
	}
	err = frontend.Bind(INFERENCE_FORWARD_IN)
	if err != nil {
		xl.Panicln(err)
	}
	backend, err := zmq.NewSocket(zmq.DEALER)
	if err != nil {
		xl.Panicln(err)
	}
	err = backend.Bind(forwardInDialer)
	if err != nil {
		xl.Panicln(err)
	}
	for {
		err = zmq.Proxy(frontend, backend, nil)
		if err != nil {
			xl.Warn("zmq proxy", err)
		}
	}
}

func processNetworkOut(buf []byte) {
	xl.Debug("processNetworkOut", len(buf))
	var msgs pb.ForwardMsgs
	err := proto.Unmarshal(buf, &msgs)
	ce(err)
	for _, msg := range msgs.Msgs {
		uuid := msg.Uuid
		if uuid == "" {
			panic("empty uuid")
		}
		waitDone, ok := m.Load(uuid)
		if ok {
			waitDone.(chan *pb.ForwardMsg) <- msg
		} else {
			xl.Warn("uuid not in map")
		}
		m.Delete(uuid)
	}
}
