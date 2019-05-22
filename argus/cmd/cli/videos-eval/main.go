package main

import (
	"bufio"
	"container/list"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	xlog "qiniupkg.com/x/xlog.v7"

	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"

	"qbox.us/cc/config"

	ahttp "qiniu.com/argus/argus/com/http"
)

type Body struct {
	Data struct {
		URI       string      `json:"uri"`
		Attribute interface{} `json:"attribute"`
	} `json:"data"`
	Params interface{} `json:"params"`
	Ops    []struct {
		OP      string      `json:"op"`
		HookURL string      `json:"hookURL"`
		Params  interface{} `json:"params"`
	} `json:"ops"`
}

type Config struct {
	HTTPHost string `json:"http_host"`

	PoolMax  int `json:"pool_max"`
	PoolSize int `json:"pool_size"`

	Request struct {
		URL  string `json:"url"`
		UID  uint32 `json:"uid"`
		Body Body   `json:"body"`
	} `json:"request"`
	URIList string `json:"uri_list"`
}

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	config.Init("f", "videos-eval", "videos-eval.conf")
	var conf Config
	if err := config.Load(&conf); err != nil {
		log.Fatal("Failed to load configure file!")
	}

	log.SetOutputLevel(log.Ldebug)
	log.Debugf("load conf %#v", conf)

	tasks := NewTasks(conf.URIList)
	{
		count, counts := tasks.Init(func() []string {
			ops := make([]string, 0)
			for _, op := range conf.Request.Body.Ops {
				ops = append(ops, op.OP)
			}
			return ops
		}())
		log.Infof("%d %v", count, counts)
	}
	// return
	run := &Run{
		Config:    conf,
		Tasks:     tasks,
		handles:   make(chan bool, conf.PoolMax),
		callbacks: make(chan Callback),
	}
	{
		for i := 0; i < conf.PoolSize; i++ {
			run.handles <- true
		}
	}
	service := Service{Run: run}

	mux := servestk.New(restrpc.NewServeMux())
	router := restrpc.Router{
		PatternPrefix: "",
		Mux:           mux,
	}
	go func() {
		if err := router.ListenAndServe(conf.HTTPHost, service); err != nil {
			log.Errorf("start error: %v", err)
		}
	}()
	run.run()
}

////////////////////////////////////////////////////////////////////////////////

type Tasks struct {
	// uriv map[string]*list.Element
	uris    *list.List
	current *list.Element

	filename string
	file     *os.File
	scanner  *bufio.Scanner
}

func NewTasks(filename string) *Tasks {
	file, _ := os.Open(filename)
	return &Tasks{
		uris:     list.New(),
		filename: filename,
		file:     file,
		scanner:  bufio.NewScanner(file),
	}
}

func (ts *Tasks) Init(ops []string) (int, []int) {

	var (
		lock = new(sync.Mutex)

		ms     = make(map[string]map[string]bool)
		counts = make([]int, len(ops))

		wait = new(sync.WaitGroup)
	)

	for i, op := range ops {
		m := make(map[string]bool)
		ms[op] = m
		wait.Add(1)
		go func(op string, i int) {
			defer wait.Done()

			var count = 0
			defer func() {
				lock.Lock()
				defer lock.Unlock()
				counts[i] = count
			}()

			file, err := os.Open(ts.filename + "__" + op)
			if os.IsNotExist(err) {
				return
			}
			defer file.Close()

			scanner := bufio.NewScanner(file)
			scanner.Buffer([]byte{}, bufio.MaxScanTokenSize*64)
			for scanner.Scan() {
				if uri := strings.Split(scanner.Text(), "\t")[0]; len(uri) > 0 {
					func() {
						lock.Lock()
						defer lock.Unlock()

						m[uri] = true
						count++
					}()
				}
			}
			if err := scanner.Err(); err != nil {
				xlog.New(op).Fatalf("reading %s: %v", ts.filename+"__"+op, err)
			}
		}(op, i)
	}

	var (
		end   int64 = 0
		count       = 0
		wait2       = new(sync.WaitGroup)
	)

	wait2.Add(1)
	go func() {
		defer wait2.Done()
		var (
			BUF = 500
		)

		for {
			for ts.scanner.Scan() {
				uri := ts.scanner.Text()
				if len(uri) == 0 {
					count++
					continue
				}
				count++

				ts.uris.PushBack(struct {
					URI string
					OPs map[string]bool
				}{
					URI: uri,
					OPs: func() map[string]bool {
						m := make(map[string]bool)
						for _, op := range ops {
							m[op] = true
						}
						return m
					}(),
				})

				if ts.uris.Len() >= BUF {
					break
				}
			}

			b, n := func() (bool, int) {
				lock.Lock()
				defer lock.Unlock()

				rms := make([]*list.Element, 0, ts.uris.Len())
				for e := ts.uris.Front(); e != nil; e = e.Next() {
					v := e.Value.(struct {
						URI string
						OPs map[string]bool
					})
					for _, op := range ops {
						if _, ok := ms[op][v.URI]; ok {
							delete(v.OPs, op)
							delete(ms[op], v.URI)
						}
					}
					if len(v.OPs) > 0 {
						e.Value = v
					} else {
						rms = append(rms, e)
					}
				}
				for _, rm := range rms {
					ts.uris.Remove(rm)
				}

				left := 0
				for _, m := range ms {
					if len(m) > left {
						left = len(m)
					}
				}
				if left == 0 && atomic.LoadInt64(&end) > 0 {
					return true, len(rms)
				}

				return false, len(rms)
			}()

			if b {
				break
			}
			if n == 0 {
				time.Sleep(time.Second)
				BUF += 500
			}
			xlog.New("SCANNER").Infof("LIST: %d, RM: %d  BUF: %d", ts.uris.Len(), n, BUF)

		}
	}()

	wait.Wait()
	atomic.StoreInt64(&end, 1)

	wait2.Wait()

	ts.current = ts.uris.Front()
	return count, counts
}

func (ts *Tasks) Next() (string, []string, bool) {
	if ts.current != nil {
		v := ts.current.Value.(struct {
			URI string
			OPs map[string]bool
		})
		ts.current = ts.current.Next()
		return v.URI,
			func() []string {
				var ops = make([]string, 0)
				for op := range v.OPs {
					ops = append(ops, op)
				}
				return ops
			}(),
			true
	}

	if ok := ts.scanner.Scan(); !ok {
		return "", []string{}, ok
	}
	return ts.scanner.Text(), []string{}, true
}

func (ts *Tasks) Close() error {
	if ts.file != nil {
		ts.file.Close()
	}
	return nil
}

////////////////////////////////////////////////////////////////////////////////

type Run struct {
	Config
	*Tasks

	handles   chan bool
	callbacks chan Callback
}

func (r *Run) run() {

	var (
		xl = xlog.New("RUN")

		uris  = make(map[string]string)
		uric  = make(map[string]int)
		urisL = new(sync.Mutex)

		done int64 = 0
		wait       = new(sync.WaitGroup)
	)

	defer r.Tasks.Close()
	// file, _ := os.Open(r.URIList)
	// defer file.Close()

	wait.Add(1)
	go func() {
		defer wait.Done()

		ws := make(map[string]*bufio.Writer)
		for _, op := range r.Request.Body.Ops {

			file, _ := os.OpenFile(r.URIList+"__"+op.OP, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0755)
			defer file.Close()

			w := bufio.NewWriter(file)
			defer w.Flush()

			ws[op.OP] = w
		}

		var (
		// c = make(map[string]int)
		// n = len(r.Request.Body.Ops)
		)

		for result := range r.callbacks {

			xl.Infof("%#v", result)

			var (
				uri = func() string {
					urisL.Lock()
					defer urisL.Unlock()
					return uris[result.ID]
				}()

				w     = ws[result.OP]
				bs, _ = json.Marshal(result)
			)

			if len(uri) == 0 {
				continue
			}

			w.Write([]byte(uri))
			w.Write([]byte("\t"))
			w.Write(bs)
			w.Write([]byte("\n"))
			w.Flush()

			func() {
				urisL.Lock()
				defer urisL.Unlock()

				uric[uri] -= 1
				if uric[uri] <= 0 {
					delete(uric, uri)
					delete(uris, result.ID)
					r.handles <- true
				}
			}()

			if len(uris) == 0 && atomic.LoadInt64(&done) == 1 {
				break
			}
		}

	}()

	var (
		client = ahttp.NewQiniuStubRPCClient(r.Request.UID, 4, time.Second)
	)

	// scanner := bufio.NewScanner(file)
	// for scanner.Scan() {
	for {
		uri, ops, ok := r.Tasks.Next()
		if !ok {
			break
		}

		<-r.handles

		var (
			id = strconv.FormatInt(time.Now().UnixNano(), 10)
			// uri    = scanner.Text()
			params = func() Body {
				var p = Body{
					Data:   r.Request.Body.Data,
					Params: r.Request.Body.Params,
					Ops: make([]struct {
						OP      string      `json:"op"`
						HookURL string      `json:"hookURL"`
						Params  interface{} `json:"params"`
					}, 0),
				}
				if ops != nil && len(ops) > 0 {
					for _, op := range ops {
						for _, _op := range r.Request.Body.Ops {
							if op == _op.OP {
								p.Ops = append(p.Ops, _op)
							}
						}
					}
				} else {
					p.Ops = r.Request.Body.Ops
				}
				return p
			}()
		)
		params.Data.URI = uri

		for i := 0; i < 10; i++ {
			var ret = struct {
				Job string `json:"job"`
			}{}
			err := client.CallWithJson(context.Background(),
				&ret, "POST",
				fmt.Sprintf("%s%s", r.Request.URL, id),
				params,
			)
			if err == nil {
				xl.Infof("POST: %s %v", r.Request.URL, ret)
				break
			}
			xl.Warnf("post failed. %v", err)
			time.Sleep(time.Second * 5)
		}

		xl.Infof("%s %s %v", uri, id, params)

		func() {
			urisL.Lock()
			defer urisL.Unlock()
			uris[id] = uri
			uric[uri] = len(params.Ops)
		}()
	}
	// if err := scanner.Err(); err != nil {
	// 	xl.Errorf("reading standard input: %v", err)
	// }

	atomic.StoreInt64(&done, 1)
	wait.Wait()
}
