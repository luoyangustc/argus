package main

import (
	"testing"

	pb "qiniu.com/ai-sdk/proto"
)

func BenchmarkProto(b *testing.B) {
	buf := make([]byte, 50*1024*1024)
	msg := pb.ForwardMsg{}
	msg.NetworkInputBuf = make([]byte, 1080*720*3*4)
	msg.Desc = "xxx"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n, err := msg.MarshalTo(buf)
		if err != nil {
			panic(err)
		}
		b.SetBytes(int64(n))
	}
}

func BenchmarkMemcopy(b *testing.B) {
	buf := make([]byte, 50*1024*1024)
	msg := pb.ForwardMsg{}
	msg.NetworkInputBuf = make([]byte, 1080*720*3*4)
	msg.Desc = "xxx"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n := copy(buf, msg.NetworkInputBuf)
		b.SetBytes(int64(n))
	}
}
