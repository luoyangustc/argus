package video

import (
	"bytes"
	"text/template"

	"github.com/gorilla/mux"
	"qiniu.com/argus/service/middleware"
)

type RouterConfig struct {
	Port   string `json:"port"`
	Prefix string `json:"prefix"`
}

type ServiceRouterConfig struct {
	Prefix   string            `json:"prefix"`
	Redirect map[string]string `json:"redirect"`
}

type Router struct {
	RouterConfig
	*mux.Router

	srs          []*Route
	callbackDocs []*Route
}

func newRouter(conf RouterConfig) *Router {
	return &Router{RouterConfig: conf, Router: mux.NewRouter(), srs: make([]*Route, 0)}
}

func (r *Router) NewRouteSetter(serviceName string, newVS func() interface{}) RouteSetter {
	return &routeSetter{Router: r, name: serviceName, newVideoService: newVS}
}

func (r *Router) Doc() ([]byte, error) {

	buf := bytes.NewBuffer(nil)

	buf.WriteString(`
# 基本参数

## 资源表示方式（URI）

通过统一方式定位、获取资源（图片、二进制数据等）

* HTTP，网络资源，形如：http://host/path、https://host/path
* Stream, 流协议(RTSP、RTMP、HLS)，形如：rtsp://host/path，rtmp://host/path;视频编码格式为H264
* FILE，本地文件，形如：file://path

## 错误返回

| 错误码 | 描述 |
| :--- | :--- |
| 4000100 | 请求参数错误 |
| 4000203 | 资源地址不存在 |
| 4150501 | 资源无法解析或非视频文件 |
| 5000400 | 异步视频分析任务错误 |
| 5000900 | 系统错误 |

# API列表
`)
	for _, sr := range r.srs {
		bs, _ := sr.doc.Marshal()
		buf.Write(bs)
	}
	if len(r.callbackDocs) > 0 {
		buf.WriteString(`
# 回调
`)
		for _, cb := range r.callbackDocs {
			bs, _ := cb.doc.Marshal()
			buf.Write(bs)
		}
	}
	return buf.Bytes(), nil
}

////////////////////////////////////////////////////////////////////////////////
type RouteSetter interface {
	UpdateRouter(func(func() middleware.Service, func() interface{}, func(string) *Route) error) RouteSetter
	Callback() *Route
}

type routeSetter struct {
	*Router
	name   string
	prefix string

	newVideoService func() interface{}
	service         middleware.Service
}

func (rs *routeSetter) New(service middleware.Service, endpoints middleware.ServiceEndpoints) (middleware.Service, error) {
	rs.service = service
	return service, nil
}

func (rs *routeSetter) Register(f func(
	service func() middleware.Service,
	genVS func() interface{},
	path func(string) *Route,
) error) {
	_ = f(rs.Service, rs.newVideoService, rs.Path)
}

func (rs *routeSetter) Service() middleware.Service { return rs.service }
func (rs *routeSetter) Path(path string) *Route {
	route := rs.Router.Path(path)
	if len(rs.prefix) > 0 {
		route = route.PathPrefix(rs.prefix) // TODO 这是误用！
	}
	r := &Route{routeSetter: rs, route: route, path: path}
	rs.Router.srs = append(rs.Router.srs, r)
	return r
}

func (rs *routeSetter) Callback() *Route {
	r := &Route{routeSetter: rs}
	rs.Router.callbackDocs = append(rs.Router.callbackDocs, r)
	return r
}

func (s *routeSetter) UpdateRouter(
	f func(func() middleware.Service, func() interface{}, func(string) *Route) error,
) RouteSetter {
	s.Register(f)
	return s
}

////////////////////////////////////////////////////////////////////////////////
type Route struct {
	*routeSetter
	route *mux.Route
	path  string
	doc   APIDoc
}

func (r *Route) Doc(doc APIDoc) *Route {
	r.doc = doc
	return r
}
func (r *Route) Route() *mux.Route { return r.route }

////////////////////////////////////////////////////////////////////////////////

type APIDocParam struct {
	Name string
	Type string
	Desc string
	Must string
}

type APIDocError struct {
	Code int
	Desc string
}

type APIDoc struct {
	Name          string
	Version       string
	Desc          []string
	Request       string
	Response      string
	RequestParam  []APIDocParam
	ResponseParam []APIDocParam
	ErrorMessage  []APIDocError
	Appendix      []string
}

func (doc APIDoc) Marshal() ([]byte, error) {

	buf := bytes.NewBuffer(nil)

	_ = template.Must(template.New("API").Parse(`## {{ .Name }} (version: {{ .Version }})
{{ with .Desc }}{{ range . }}
> {{ . }}<br>{{ end }}{{ end }}

**Request**

`+
		"```"+
		`
{{ .Request }}
`+
		"```"+
		`

**Response**

`+
		"```"+
		`
{{ .Response }}
`+
		"```"+
		`{{ with .RequestParam }}
**Request Params**

| 参数 | 类型 | 必选 | 描述 |
| :--- | :---- | :--- | :--- |
{{ range . }}| {{ .Name }} | {{ .Type }} | {{ .Must }} |{{ .Desc }} |
{{ end }}{{ end }}{{ with .ResponseParam }}
**Response Params**

| 参数 | 类型 | 描述 |
| :--- | :---- | :--- |
{{ range . }}| {{ .Name }} | {{ .Type }} | {{ .Desc }} |
{{ end }}{{ end }}{{ with .ErrorMessage }}
**Error Message**

| 错误码 | 描述 |
| :--- | :--- |
{{ range . }}| {{ .Code }} | {{ .Desc }} |
{{ end }}{{ end }}{{ with .Appendix }}
*Appendix*

{{ range . }} {{ . }}
{{ end }}{{ end }}
`)).Execute(buf, doc)

	return buf.Bytes(), nil
}

////////////////////////////////////////////////////////////////////////////////

type OpDocParam struct {
	Name string
	Type string
	Desc string
	Must string
}

type OPDoc struct {
	Name          string
	Version       string
	Desc          []string
	Request       string
	Response      string
	RequestParam  []OpDocParam
	ResponseParam []OpDocParam
}

func (doc OPDoc) Marshal() ([]byte, error) {

	buf := bytes.NewBuffer(nil)

	_ = template.Must(template.New("OPS").Parse(`## {{ .Name }} (version: {{ .Version }})
{{ with .Desc }}{{ range . }}
> {{ . }}<br>{{ end }}{{ end }}

**Request**

`+
		"```"+
		`
{{ .Request }}
`+
		"```"+
		`{{ with .RequestParam }}
**Request Params**

| 参数 | 类型 | 必选 | 描述 |
| :--- | :---- | :--- | :--- |
{{ range . }}| {{ .Name }} | {{ .Type }} | {{ .Must }} |{{ .Desc }} |
{{ end }}{{ end }}
**Response**

`+
		"```"+
		`
{{ .Response }}
`+
		"```"+
		`{{ with .ResponseParam }}

**Response Params**

| 参数 | 类型 | 描述 |
| :--- | :---- | :--- |
{{ range . }}| {{ .Name }} | {{ .Type }} | {{ .Desc }} |
{{ end }}{{ end }}
`)).Execute(buf, doc)

	return buf.Bytes(), nil
}
