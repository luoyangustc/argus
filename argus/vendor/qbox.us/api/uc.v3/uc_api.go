// 相比v1，bucketInfo不会返回BindDomains
package uc

import (
	"encoding/base64"
	"net/http"
	"strconv"
	"strings"

	"github.com/qiniu/rpc.v1"
	"github.com/qiniu/rpc.v1/lb.v2.1"
	"qbox.us/api"
	"qbox.us/api/uc"
	rpc2 "qbox.us/rpc"
)

type BucketInfo uc.BucketInfoWithoutBindDomains

type BucketInfos []struct {
	Name string     `json:"name"`
	Info BucketInfo `json:"info"`
}

type BucketType int

const (
	TYPE_COM BucketType = iota
	TYPE_MEDIA
	TYPE_DL
)

func (v BucketType) Valid() bool {
	switch v {
	case TYPE_COM, TYPE_MEDIA, TYPE_DL:
		return true
	}
	return false
}

// ------------------------------------------------------------------------------------------

type Service struct {
	Conn       *lb.Client
	uc.Service // 功能废弃，兼容保留
}

func New(hosts []string, t http.RoundTripper) *Service {
	cfg := &lb.Config{
		Hosts:    hosts,
		TryTimes: uint32(len(hosts)),
	}
	client := lb.New(cfg, t)
	return &Service{
		Conn: client,
	}
}

// ------------------------------------------------------------------------------------------

func (r Service) BucketInfo(l rpc.Logger, bucket string) (info BucketInfo, err error) {

	params := map[string][]string{
		"bucket": {bucket},
	}
	err = r.Conn.CallWithForm(l, &info, "/v2/bucketInfo", params)
	return
}

func (r Service) BucketInfos(l rpc.Logger, zone string) (infos BucketInfos, err error) {
	return r.BucketInfosWithShared(l, zone, false)
}

func (r Service) BucketInfosWithShared(l rpc.Logger, zone string, shared bool) (infos BucketInfos, err error) {

	params := map[string][]string{}
	if zone != "" {
		params["zone"] = []string{zone}
	}
	if shared {
		params["shared"] = []string{"true"}
	}
	err = r.Conn.CallWithForm(l, &infos, "/v2/bucketInfos", params)
	return
}

func (r Service) GlbBucketInfos(l rpc.Logger, region, global string) (infos BucketInfos, err error) {
	return r.GlbBucketInfosWithShared(l, region, global, false)
}

func (r Service) GlbBucketInfosWithShared(l rpc.Logger, region, global string, shared bool) (infos BucketInfos, err error) {

	params := map[string][]string{}
	if region != "" {
		params["region"] = []string{region}
	}
	if global != "" {
		params["global"] = []string{global}
	}
	if shared {
		params["shared"] = []string{"true"}
	}
	err = r.Conn.CallWithForm(l, &infos, "/v2/bucketInfos", params)
	return
}

const (
	EnabledKey  = 0
	DisabledKey = 1

	// 允许空 Refer 访问
	NoReferAllow = 0
	// 不允许空 Refer 访问
	NoReferDisallow = 1
)

type AppInfo struct {
	Key     string `json:"key,omitempty"`
	Secret  string `json:"secret,omitempty"`
	Key2    string `json:"key2,omitempty"`
	Secret2 string `json:"secret2,omitempty"`
	AppId   uint32 `json:"appId"`
	State   uint16 `json:"state,omitempty"`  // 第1对 Key/Secret 的状态
	State2  uint16 `json:"state2,omitempty"` // 第2对 Key2/Secret2 的状态
}

func (r *Service) AppInfo(l rpc.Logger, app string) (info AppInfo, err error) {

	params := map[string][]string{
		"app": {app},
	}
	err = r.Conn.CallWithForm(l, &info, "/appInfo", params)
	return
}

type HostsInfo struct {
	Global bool                `json:"global"`
	Http   map[string][]string `json:"http"`
	Https  map[string][]string `json:"https"`
}

func (r *Service) BucketHosts(l rpc.Logger, bucket string) (info HostsInfo, err error) {
	err = r.Conn.Call(l, &info, r.Host+"/host/"+bucket)
	return
}

func (r *Service) RefreshBucket(l rpc.Logger, bucket string) (err error) {

	params := map[string][]string{
		"bucket": {bucket},
	}
	err = r.Conn.CallWithForm(l, nil, "/refreshBucket", params)
	return
}

func (r *Service) FopAuth(l rpc.Logger, bucket string, mode int, fop string) (err error) {

	params := map[string][]string{
		"bucket": {bucket},
		"fop":    {fop},
		"mode":   {strconv.Itoa(mode)},
	}
	err = r.Conn.CallWithForm(l, nil, "/fopAuth", params)
	return
}

func (r *Service) AntiLeechMode(l rpc.Logger, bucket string, mode int) (err error) {

	params := map[string][]string{
		"bucket": {bucket},
		"mode":   {strconv.Itoa(mode)},
	}
	err = r.Conn.CallWithForm(l, nil, "/antiLeechMode", params)
	return
}

func (r *Service) ReferAntiLeech(l rpc.Logger, bucket string, mode int, norefer int, pattern string) (err error) {

	params := map[string][]string{
		"bucket":  {bucket},
		"mode":    {strconv.Itoa(mode)},
		"norefer": {strconv.Itoa(norefer)},
		"pattern": {pattern},
	}
	err = r.Conn.CallWithForm(l, nil, "/referAntiLeech", params)
	return
}

func (r *Service) Private(l rpc.Logger, bucket string, private int) (err error) {

	params := map[string][]string{
		"bucket":  {bucket},
		"private": {strconv.Itoa(private)},
	}
	err = r.Conn.CallWithForm(l, nil, "/private", params)
	return
}

func (r *Service) NoIndexPage(l rpc.Logger, bucket string, noIndexPage int) (err error) {

	params := map[string][]string{
		"bucket":      {bucket},
		"noIndexPage": {strconv.Itoa(noIndexPage)},
	}
	err = r.Conn.CallWithForm(l, nil, "/noIndexPage", params)
	return
}

func (r *Service) ImgSFT(l rpc.Logger, bucket string, ft int) (err error) {

	params := map[string][]string{
		"bucket": {bucket},
		"imgSFT": {strconv.Itoa(ft)},
	}
	err = r.Conn.CallWithForm(l, nil, "/imgSFT", params)
	return
}

func (r *Service) MaxAge(l rpc.Logger, bucket string, maxAge int32) (code int, err error) {

	params := map[string][]string{
		"bucket": {bucket},
		"maxAge": {strconv.FormatInt(int64(maxAge), 10)},
	}
	err = r.Conn.CallWithForm(l, nil, "/maxAge", params)
	return
}

func (r *Service) Channel(l rpc.Logger, bucket string, channels []string) (err error) {

	params := map[string][]string{
		"bucket":  {bucket},
		"channel": channels,
	}
	err = r.Conn.CallWithForm(l, nil, "/channel", params)
	return
}

func (r *Service) Type(l rpc.Logger, bucket string, bucketType BucketType) (err error) {

	params := map[string][]string{
		"bucket": {bucket},
		"type":   {strconv.Itoa(int(bucketType))},
	}
	err = r.Conn.CallWithForm(l, nil, "/type", params)
	return
}

func (r *Service) SetPersistFop(l rpc.Logger, bucket string, persist int) (code int, err error) {

	params := map[string][]string{
		"bucket":  {bucket},
		"persist": {strconv.Itoa(persist)},
	}
	err = r.Conn.CallWithForm(l, nil, "/persistFop", params)
	return
}

func (r *Service) TokenAntiLeech(l rpc.Logger, bucket string, mode int) (err error) {

	return r.Conn.Call(l, nil, "/tokenAntiLeech/"+bucket+"/mode/"+strconv.Itoa(mode))
}

func (r *Service) NewMacKey(l rpc.Logger, bucket string, index int) (key string, code int, err error) {

	ret := struct {
		Key string `json:"key"`
	}{}
	err = r.Conn.Call(l, &ret, "/newMacKey/"+bucket+"/index/"+strconv.Itoa(index))
	return ret.Key, code, err
}

func (r *Service) DeleteMacKey(l rpc.Logger, bucket string, index int) (err error) {

	return r.Conn.Call(l, nil, "/deleteMacKey/"+bucket+"/index/"+strconv.Itoa(index))
}

type AccessInfo struct {
	Key    string `json:"key"`
	Secret string `json:"secret"`
}

func (r *Service) NewAccess(l rpc.Logger, app string) (info AccessInfo, err error) {

	params := map[string][]string{
		"app": {app},
	}
	err = r.Conn.CallWithForm(l, &info, "/newAccess", params)
	return
}

func (r *Service) SetKeyState(l rpc.Logger, app string, accessKey string, state int) (err error) {

	params := map[string][]string{
		"app":   {app},
		"key":   {accessKey},
		"state": {strconv.Itoa(state)},
	}
	err = r.Conn.CallWithForm(l, nil, "/setKeyState", params)
	return
}

func (r *Service) DeleteAccess(l rpc.Logger, app string, accessKey string) (err error) {

	params := map[string][]string{
		"app": {app},
		"key": {accessKey},
	}
	err = r.Conn.CallWithForm(l, nil, "/deleteAccess", params)
	return
}

func (r *Service) Image(l rpc.Logger, bucketName string, srcSiteUrls []string, srcHost string, expires int) (err error) {
	url := "/image/" + bucketName
	url += "/from/" + rpc2.EncodeURI(strings.Join(srcSiteUrls, ";"))
	if expires != 0 {
		url += "/expires/" + strconv.Itoa(expires)
	}
	if srcHost != "" {
		url += "/host/" + rpc2.EncodeURI(srcHost)
	}
	return r.Conn.Call(l, nil, url)
}

func (r *Service) ImageWithAKSK(l rpc.Logger, bucketName string, srcSiteUrls []string, srcHost string, expires int, qiniuAK, qiniuSK string) (err error) {
	url := "/image/" + bucketName
	for _, srcSiteUrl := range srcSiteUrls {
		url += "/from/" + rpc2.EncodeURI(srcSiteUrl)
	}
	if expires != 0 {
		url += "/expires/" + strconv.Itoa(expires)
	}
	if srcHost != "" {
		url += "/host/" + rpc2.EncodeURI(srcHost)
	}
	if qiniuAK != "" {
		url += "/qiniuak/" + rpc2.EncodeURI(qiniuAK) + "/qiniusk/" + rpc2.EncodeURI(qiniuSK)
	}
	return r.Conn.Call(l, nil, url)
}

func (r *Service) Unimage(l rpc.Logger, bucketName string) (err error) {
	return r.Conn.Call(l, nil, "/unimage/"+bucketName)
}

func (r *Service) AccessMode(l rpc.Logger, bucketName string, mode int) (err error) {
	return r.Conn.Call(l, nil, "/accessMode/"+bucketName+"/mode/"+strconv.Itoa(mode))
}

func (r *Service) Separator(l rpc.Logger, bucketName string, sep string) (err error) {
	return r.Conn.Call(l, nil, "/separator/"+bucketName+"/sep/"+base64.URLEncoding.EncodeToString([]byte(sep)))
}

func (r *Service) Style(l rpc.Logger, bucketName string, name string, style string) (err error) {
	return r.Conn.Call(l, nil, "/style/"+bucketName+"/name/"+rpc2.EncodeURI(name)+"/style/"+rpc2.EncodeURI(style))
}

func (r *Service) Unstyle(l rpc.Logger, bucketName string, name string) (err error) {
	return r.Conn.Call(l, nil, "/unstyle/"+bucketName+"/name/"+rpc2.EncodeURI(name))
}

func (r *Service) PreferStyleAsKey(l rpc.Logger, bucketName string, prefer bool) (err error) {
	return r.Conn.Call(l, nil, "/preferStyleAsKey/"+bucketName+"/prefer/"+strconv.FormatBool(prefer))
}

// ------------------------------------------------------------------------------------------

/*
Request:
	POST /setImagePreviewStyle?name=<Name>&mode=square&size=<Size> [&q=<Quality>&sharpen=<Sharpen>]
	POST /setImagePreviewStyle?name=<Name>&height=<Size> [&q=<Quality>&sharpen=<Sharpen>]
	POST /setImagePreviewStyle?name=<Name>&width=<Size> [&q=<Quality>&sharpen=<Sharpen>]
	POST /setImagePreviewStyle?name=<Name>&width=<Width>&height=<Height> [&q=<Quality>&sharpen=<Sharpen>]
Style:
	square:<Size>;q:<Quality>;sharpen:<Sharpen>
	<Width>x;q:<Quality>;sharpen:<Sharpen>
	x<Height>;q:<Quality>;sharpen:<Sharpen>
	<Width>x<Height>;q:<Quality>;sharpen:<Sharpen>
*/
func (r *Service) SetImagePreviewStyle(l rpc.Logger, name string, style string) (code int, err error) {

	params := map[string][]string{
		"name": {name},
	}
	ps := strings.Split(style, ";")
	ps0 := ps[0]
	if strings.HasPrefix(ps0, "square:") {
		params["mode"] = []string{"square"}
		params["size"] = []string{ps0[7:]}
	} else {
		pos := strings.Index(ps0, "x")
		if pos == -1 {
			code, err = api.InvalidArgs, api.EInvalidArgs
			return
		}
		width := ps0[:pos]
		height := ps0[pos+1:]
		if width != "" {
			params["width"] = []string{width}
		}
		if height != "" {
			params["height"] = []string{height}
		}
	}
	for i := 1; i < len(ps); i++ {
		pos := strings.Index(ps[i], ":")
		if pos == -1 {
			code, err = api.InvalidArgs, api.EInvalidArgs
			return
		}
		params[ps[i][:pos]] = []string{ps[i][pos+1:]}
	}
	err = r.Conn.CallWithForm(l, nil, "/setImagePreviewStyle", params)
	return
}

// ------------------------------------------------------------------------------------------
