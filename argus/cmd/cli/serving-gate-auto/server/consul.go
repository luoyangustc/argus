package server

import (
	"bytes"
	"context"
	"encoding/json"
	"html/template"
	"net/http"
	"strings"

	"github.com/pkg/errors"
	"github.com/qiniu/rpc.v3"
)

type ConsulService struct {
	ChecksCritical int      `json:"ChecksCritical"`
	ChecksPassing  int      `json:"ChecksPassing"`
	ChecksWarning  int      `json:"ChecksWarning"`
	Name           string   `json:"Name"`
	Nodes          []string `json:"Nodes"`
}

func (s *server) readConsulApi() (r []*ConsulService, err error) {
	if s.mock {
		d := `[{"Name":"argus-bcp","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"argus-cap","Nodes":["master-cs-cs49"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"argus-ccp","Nodes":["master-cs-cs49"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"argus-ccp-manual","Nodes":["master-cs-cs49"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"argus-ccp-review","Nodes":["master-cs-cs49"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"argus-censor","Nodes":["fop-cs-csgpu2"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"argus-job-gate","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"argus-jobm-bc","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"argus-jobw-bc","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"argus-jobw-ci","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"argus-jobw-cv","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"argus-jobw-ii","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"argus-jobw-iv","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"argus-policitian","Nodes":["master-cs-cs49"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"aslan","Nodes":["master-cs-cs48","master-cs-cs48","master-cs-cs49"],"ChecksPassing":6,"ChecksWarning":0,"ChecksCritical":0},{"Name":"aslan3","Nodes":["master-cs-cs49"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-argus-gate","Nodes":["fop-cs-csgpu2"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-argus-util","Nodes":["fop-cs-csgpu2"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-argus-vframe","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-argus-video","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-face-group","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-face-group-upg","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-face-group-w","Nodes":["fop-cs-csgpu2","master-cs-cs48","master-cs-cs48"],"ChecksPassing":6,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-facex-detect","Nodes":["fop-cs-csgpu2"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-facex-feature-v2","Nodes":["fop-cs-csgpu2"],"ChecksPassing":1,"ChecksWarning":0,"ChecksCritical":1},{"Name":"ava-facex-feature-v3","Nodes":["fop-cs-csgpu2"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-facex-search","Nodes":["master-cs-cs49"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-image-group","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-image-group-w","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-ocr-sari-blicen","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-pulp","Nodes":["fop-cs-csgpu2"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-search-bjrun","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-serving-gate","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-terror-classify","Nodes":["fop-cs-csgpu2"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-terror-detect","Nodes":["fop-cs-csgpu2"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-terror-postdet","Nodes":["fop-cs-csgpu2"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ava-terror-predetect","Nodes":["fop-cs-csgpu2"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"boots-cadvisor","Nodes":["fop-cs-csgpu5","master-cs-cs46","master-cs-cs48","master-cs-cs49"],"ChecksPassing":4,"ChecksWarning":0,"ChecksCritical":0},{"Name":"boots-scheduler","Nodes":["master-cs-cs46"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"bucket-inspect","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"concat","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"consul","Nodes":["master-cs-cs46","master-cs-cs48","master-cs-cs49"],"ChecksPassing":3,"ChecksWarning":0,"ChecksCritical":0},{"Name":"dora-avts-stream","Nodes":["master-cs-cs48","master-cs-cs48"],"ChecksPassing":4,"ChecksWarning":0,"ChecksCritical":0},{"Name":"dora-vod","Nodes":["master-cs-cs48","master-cs-cs48"],"ChecksPassing":4,"ChecksWarning":0,"ChecksCritical":0},{"Name":"faketest","Nodes":["master-cs-cs48","master-cs-cs48"],"ChecksPassing":4,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ffmpeg","Nodes":["fop-cs-csgpu2","fop-cs-csgpu2","fop-cs-csgpu2","fop-cs-csgpu2","fop-cs-csgpu2","fop-cs-csgpu2","master-cs-cs48","master-cs-cs48","master-cs-cs48","master-cs-cs48","master-cs-cs48","master-cs-cs48","master-cs-cs48","master-cs-cs48","master-cs-cs48"],"ChecksPassing":30,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ffmpeg-adapt","Nodes":["master-cs-cs48","master-cs-cs48"],"ChecksPassing":4,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ffmpeg-qsv","Nodes":["fop-cs-csgpu5"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"ffmpeg-shunt","Nodes":["master-cs-cs48"],"ChecksPassing":1,"ChecksWarning":0,"ChecksCritical":0},{"Name":"fop-proxy","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"hello","Nodes":["master-cs-cs49"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"image","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"image-censor","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"imageave","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"imageslim","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"labelx-backend","Nodes":["fop-cs-csgpu2","master-cs-cs48"],"ChecksPassing":4,"ChecksWarning":0,"ChecksCritical":0},{"Name":"labelx-server","Nodes":["master-cs-cs49"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"md2html","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"mkdir","Nodes":["master-cs-cs49","master-cs-cs49","master-cs-cs49"],"ChecksPassing":6,"ChecksWarning":0,"ChecksCritical":0},{"Name":"mkzip","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"notify-filter","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"odconv","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"portal-xxxxx","Nodes":["master-cs-cs49"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"qa-test1","Nodes":["master-cs-cs49","master-cs-cs49","master-cs-cs49"],"ChecksPassing":6,"ChecksWarning":0,"ChecksCritical":0},{"Name":"qboxkmq","Nodes":["master-cs-cs49"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"qboxqiniuproxy","Nodes":["master-cs-cs49"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"qboxup","Nodes":["master-cs-cs49"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"qhash","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"qpulp","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"qrcode","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"saveas","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"stress","Nodes":["master-cs-cs49"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"testkane","Nodes":["master-cs-cs49"],"ChecksPassing":1,"ChecksWarning":0,"ChecksCritical":1},{"Name":"video-censor","Nodes":["master-cs-cs48"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0},{"Name":"yifangyun_preview","Nodes":["master-cs-cs49"],"ChecksPassing":2,"ChecksWarning":0,"ChecksCritical":0}]`
		ce(json.Unmarshal([]byte(d), &r))
		return
	}
	c := rpc.Client{Client: http.DefaultClient}
	err = c.CallWithJson(context.Background(), &r, "GET", "http://consul.dev.qiniu.io/v1/internal/ui/services", nil)
	if err != nil {
		return nil, errors.Wrap(err, "call consul")
	}
	return
}

func (s *server) readAppStatus() {
	r, err := s.readConsulApi()
	if err != nil {
		xl.Warn("readAppStatus", err)
		return
	}
	s.lock.Lock()
	defer s.lock.Unlock()
	oldServiceMap := s.serviceMap
	s.serviceMap = make(map[string]*ConsulService)
	for _, v := range r {
		if strings.HasPrefix(v.Name, "ava") || strings.HasPrefix(v.Name, "argus") {
			if oldServiceMap[v.Name] == nil {
				xl.Debugf("add service %#v", v.Name)
			}
			s.serviceMap[v.Name] = v
		}
	}
	for k := range oldServiceMap {
		if s.serviceMap[k] == nil {
			xl.Debugf("delete service %#v", k)
		}
	}
}

func (s *server) appExists(app string) bool {
	s.lock.RLock()
	svr := s.serviceMap[app]
	s.lock.RUnlock()
	return svr != nil
}

func (s *server) appStatusPage() string {
	s.lock.RLock()
	buf, err := json.Marshal(s.serviceMap)
	s.lock.RUnlock()
	ce(err)
	var r interface{}
	ce(json.Unmarshal(buf, &r))
	var b bytes.Buffer
	tpl, err := template.New("").Parse(`
	CS 环境存在的APP：
	{{range .}}<li>{{.Name}} 正常实例数目：{{.ChecksPassing}} {{if gt .ChecksCritical 0. }} 健康检查失败状态实例数目：{{.ChecksCritical}} {{end}} {{if gt .ChecksCritical 0. }} 健康检查警告状态实例数目： {{.ChecksWarning}}  {{end}}</li>{{end}}`)
	ce(err)
	ce(tpl.Execute(&b, r))
	return b.String()
}
