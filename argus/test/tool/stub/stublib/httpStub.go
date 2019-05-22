package stublib

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"time"
)

type mock struct {
	Request  []byte
	Response string
	Num      int
	Code     int
}

type Livemock struct {
	Request  [][]byte
	Response string
	Num      int
	Code     int
}

func (a *Livemock) LiveCallback(w http.ResponseWriter, r *http.Request) {
	//a.Request = make([]byte, 1024*128)
	//r.Body.Read(a.Request)
	//var err error
	defer r.Body.Close()
	request, _ := ioutil.ReadAll(r.Body)
	a.Request = append(a.Request, request)
	// a.Request = string(req)
	fmt.Println("Received Request:")
	fmt.Println(string(request))
	a.Num += 1
	io.WriteString(w, a.Response)
	w.WriteHeader(a.Code)
}

func (a *mock) Callback(w http.ResponseWriter, r *http.Request) {
	//a.Request = make([]byte, 1024*128)
	//r.Body.Read(a.Request)
	//var err error
	defer r.Body.Close()
	a.Request, _ = ioutil.ReadAll(r.Body)
	// a.Request = string(req)
	fmt.Println("Received Request:")
	fmt.Println(string(a.Request))
	a.Num += 1
	io.WriteString(w, a.Response)
	w.WriteHeader(a.Code)
}

func (a *mock) SetCode(code int) {
	a.Code = code
}

func (a *mock) Reset() {
	a.Num = 0
	a.Request = nil
	a.Response = ""
}

func (a *mock) Recived() bool {
	return a.Num > 0
}

func (a *mock) Wait(t int) bool {
	t *= 1000
	step := 10
	if t < step {
		time.Sleep(time.Microsecond * time.Duration(t))
		return a.Recived()
	} else {
		for s := 0; s < t; s += step {
			time.Sleep(time.Microsecond * time.Duration(step))
			if a.Recived() {
				return true
			}
		}
	}
	return false
}

func (a *mock) SetRes(v string) {
	a.Response = v
}

func (a *Livemock) SetRes(v string) {
	a.Response = v
}

func NewServer(port string) *http.Server {
	return &http.Server{
		Addr:           ":" + port,
		Handler:        http.DefaultServeMux,
		ReadTimeout:    1 * time.Second,
		WriteTimeout:   1 * time.Second,
		MaxHeaderBytes: 1 << 20,
	}
}

type Stub struct {
	*mock
	*http.Server
}

type LiveStub struct {
	*Livemock
	*http.Server
}

func NewLiveStub(port string) *LiveStub {
	stub := &LiveStub{new(Livemock), NewServer(port)}
	http.HandleFunc("/", stub.LiveCallback)
	go func() {
		if err := stub.ListenAndServe(); err != nil {
			panic(err)
		}
	}()
	return stub
}

func NewStub(port string) *Stub {
	stub := &Stub{new(mock), NewServer(port)}
	http.HandleFunc("/", stub.Callback)
	go func() {
		if err := stub.ListenAndServe(); err != nil {
			panic(err)
		}
	}()
	return stub
}

// func main() {
// 	stub := NewStub("8080")
// 	stub.SetRes("abc")
// 	//time.Sleep(10 * time.Second)
// 	//stub.SetRes("hij")
// 	//time.Sleep(10 * time.Second)
// 	if err := stub.Shutdown(nil); err != nil {
// 		panic(err)
// 	}
// 	//fmt.Println("done!")
// }
