package image_sync

import (
	"bytes"
	"text/template"

	"github.com/gorilla/mux"

	"qiniu.com/argus/service/middleware"
	pimage "qiniu.com/argus/service/service/image"
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

	srs []*ServiceRoute
}

func newRouter(conf RouterConfig) *Router {
	return &Router{RouterConfig: conf, Router: mux.NewRouter(), srs: make([]*ServiceRoute, 0)}
}

func (r *Router) NewService(serviceName string, imageParser pimage.IImageParse) *ServiceRouter {
	return &ServiceRouter{Router: r, name: serviceName, IImageParse: imageParser}
}

func (r *Router) Doc() ([]byte, error) {
	buf := bytes.NewBuffer(nil)

	buf.WriteString(`
# 基本参数

## 输入图片格式
支持JPG、PNG、BMP、GIF

## 资源表示方式（URI）

通过统一方式定位、获取资源（图片、二进制数据等）

* HTTP，网络资源，形如：http://host/path、https://host/path
* FILE，本地文件，形如：file://path
* Data，Data URI Scheme形态的二进制文件，形如：data:application/octet-stream;base64,xxx。ps: 当前只支持前缀为data:application/octet-stream;base64,的数据

## 错误返回

| 错误码 | 描述 |
| :--- | :--- |
| 4000100 | 请求参数错误 |
| 4000201 | 图片资源地址不支持 |
| 4000203 | 获取图片失败 |
| 4000204 | 获取图片超时 |
| 4150301 | 图片格式不支持 |
| 4000302 | 图片过大，图片长宽超过4999像素、或图片大小超过10M |
| 5000900 | 系统错误 |

# API列表

`)

	for _, sr := range r.srs {
		bs, _ := sr.doc.Marshal()
		buf.Write(bs)
	}
	return buf.Bytes(), nil
}

type ServiceRouter struct {
	*Router
	name   string
	prefix string

	service middleware.Service
	pimage.IImageParse

	// f func(
	// 	service func() middleware.Service,
	// 	imageParser func() pimage.IImageParse,
	// 	path func(string) *ServiceRoute,
	// ) error
}

func (sr *ServiceRouter) New(service middleware.Service, endpoints middleware.ServiceEndpoints) (middleware.Service, error) {
	sr.service = service
	// err := sr.f(service, sr.IImageParse, sr.Path)
	return service, nil
}

func (sr *ServiceRouter) Register(f func(
	service func() middleware.Service,
	imageParser func() pimage.IImageParse,
	path func(string) *ServiceRoute,
) error) {
	_ = f(sr.Service, sr.ImageParse, sr.Path)
}

func (sr *ServiceRouter) Service() middleware.Service    { return sr.service }
func (sr *ServiceRouter) ImageParse() pimage.IImageParse { return sr.IImageParse }
func (sr *ServiceRouter) Path(path string) *ServiceRoute {
	route := sr.Router.Path(path)
	if len(sr.prefix) > 0 {
		route = route.PathPrefix(sr.prefix) // TODO 这是误用！
	}
	r := &ServiceRoute{ServiceRouter: sr, route: route, path: path}
	sr.Router.srs = append(sr.Router.srs, r)
	return r
}

type ServiceRoute struct {
	*ServiceRouter
	route *mux.Route
	path  string
	doc   APIDoc
}

func (sr *ServiceRoute) Doc(doc APIDoc) *ServiceRoute {
	sr.doc = doc
	sr.doc.Path = sr.path
	return sr
}
func (sr *ServiceRoute) Route() *mux.Route { return sr.route }

////////////////////////////////////////////////////////////////////////////////

type APIDocParam struct {
	Name string
	Type string
	Desc string
}

type APIDocError struct {
	Code int
	Desc string
}

type APIDoc struct {
	Path          string
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

	_ = template.Must(template.New("API").Parse(`## {{ .Path }} ({{ .Version }})
{{ with .Desc }}{{ range . }}
> {{ . }}<br>{{ end }}{{ end }}

*Request*

`+
		"```"+
		`
{{ .Request }}
`+
		"```"+
		`

*Response*

`+
		"```"+
		`
{{ .Response }}
`+
		"```"+
		`{{ with .RequestParam }}
*Request Params*

| 参数 | 类型 | 描述 |
| :--- | :---- | :--- |
{{ range . }}| {{ .Name }} | {{ .Type }} | {{ .Desc }} |
{{ end }}{{ end }}{{ with .ResponseParam }}
*Response Params*

| 参数 | 类型 | 描述 |
| :--- | :---- | :--- |
{{ range . }}| {{ .Name }} | {{ .Type }} | {{ .Desc }} |
{{ end }}{{ end }}{{ with .ErrorMessage }}
*Error Message*

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
