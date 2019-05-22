package supervisor

import (
	"net/http"
	"sync"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/misc/pool"

	. "github.com/qiniu/http/audit/proto"
)

// ----------------------------------------------------------

type Supervisor struct {
	sessions pool.Pool
	mutex    sync.Mutex
}

func New() *Supervisor {

	return new(Supervisor)
}

func (p *Supervisor) OnStartReq(req *Request) (id interface{}, err error) {

	p.mutex.Lock()
	id = p.sessions.Add(req)
	p.mutex.Unlock()
	return
}

func (p *Supervisor) OnEndReq(id interface{}) {

	p.mutex.Lock()
	p.sessions.Free(id.(*interface{}))
	p.mutex.Unlock()
}

func (p *Supervisor) DumpRequests() (reqs []*Request) {

	ipage := 0
	for {
		p.mutex.Lock()
		err := p.sessions.ForPage(ipage, func(v interface{}) {
			reqs = append(reqs, v.(*Request))
		})
		p.mutex.Unlock()
		if err != nil {
			break
		}
		ipage++
	}
	return
}

// ----------------------------------------------------------

type Config struct {
	BindHost string `json:"bind_host"`
}

type dumpRet struct {
	Reqs []*Request `json:"reqs"`
}

func (p *Supervisor) Dump(w http.ResponseWriter, req *http.Request) {

	httputil.Reply(w, 200, &dumpRet{
		Reqs: p.DumpRequests(),
	})
}

func (p *Supervisor) RegisterHandlers(mux *http.ServeMux) {

	mux.HandleFunc("/debug/dump", p.Dump)
}

func (p *Supervisor) ListenAndServe(cfg *Config) (err error) {

	mux := http.NewServeMux()
	p.RegisterHandlers(mux)
	return http.ListenAndServe(cfg.BindHost, mux)
}

// ----------------------------------------------------------
