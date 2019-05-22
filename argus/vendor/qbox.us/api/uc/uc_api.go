package uc

import (
	"encoding/base64"
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"qbox.us/api"
	"qbox.us/api/tblmgr.v2"
	"qbox.us/rpc"
)

// ------------------------------------------------------------------------------------------
type BucketInfoWithoutBindDomains struct {
	Source            string             `json:"source" bson:"source"`
	Host              string             `json:"host" bson:"host"`
	Expires           int                `json:"expires" bson:"expires"`
	SourceQiniuAK     string             `json:"source_qiniu_ak" bson:"source_qiniu_ak"`
	SourceQiniuSK     string             `json:"source_qiniu_sk" bson:"source_qiniu_sk"`
	Protected         int                `json:"protected" bson:"protected"`
	Separator         string             `json:"separator" bson:"separator"`
	Styles            map[string]string  `json:"styles" bson:"styles"`
	RefreshTime       int64              `json:"refresh_time" bson:"refresh_time"`
	ReferWhiteList    []string           `json:"refer_wl" bson:"refer_wl"`
	ReferBlackList    []string           `json:"refer_bl" bson:"refer_bl"`
	ReferNoRefer      bool               `json:"no_refer" bson:"no_refer"`
	AntiLeechMode     int                `json:"anti_leech_mode" bson:"anti_leech_mode"` // 0:off,1:wl,2:bl
	Private           int                `json:"private" bson:"private"`
	NoIndexPage       int                `json:"no_index_page" bson:"no_index_page"`
	ImgSFT            int                `json:"imgsft" bson:"imgsft"` //img_storage_fault_tolerant
	PreferStyleAsKey  bool               `json:"prefer_style_as_key" bson:"prefer_style_as_key"`
	MaxAge            int32              `json:"max_age" bson:"max_age"`
	MacKey            string             `json:"mac_key" bson:"mac_key"`
	MacKey2           string             `json:"mac_key2" bson:"mac_key2"`
	TokenAntiLeech    int                `json:"token_anti_leech" bson:"token_anti_leech"`
	Channel           []string           `json:"channel" bson:"channel"` // 没用了
	PersistFop        int                `json:"persist_fop" bson:"persist_fop"`
	Zone              string             `json:"zone" bson:"zone"`
	Region            string             `json:"region" bson:"region"`
	Global            bool               `json:"global" bson:"global"`
	Line              bool               `json:"line" bson:"line"`
	Type              BucketType         `json:"type" bson:"type"`
	NotifyQueue       string             `json:"notify_queue" bson:"notify_queue"`
	NotifyMessage     string             `json:"notify_message" bson:"notify_message"`
	NotifyMessageType string             `json:"notify_message_type" bson:"notify_message_type"`
	Ouid              uint32             `json:"ouid" bson:"ouid"`
	Otbl              string             `json:"otbl" bson:"otbl"`
	Perm              uint32             `json:"perm" bson:"perm"`
	ShareUsers        []tblmgr.ShareUser `json:"share_users" bson:"share_users"`
	BucketRules       []BucketRule       `json:"bucket_rules" bson:"bucket_rules"`

	FopAccessWhiteList []string `json:"fop_accs_wlist" bson:"fop_accs_wlist,omitempty"`
}

type BucketRule struct {
	Name            string    `json:"name" bson:"name"`
	Prefix          string    `json:"prefix" bson:"prefix"`
	DeleteAfterDays int       `json:"delete_after_days" bson:"delete_after_days"`
	ToLineAfterDays int       `json:"to_line_after_days" bson:"to_line_after_days"`
	Ctime           time.Time `json:"ctime" bson:"ctime"`
}

type BucketInfo struct {
	BucketInfoWithoutBindDomains
	BindDomains []string `json:"bind_domains" bson:"bind_domains"`
}

type HostsInfo struct {
	Ttl    int                 `json:"ttl"` // 单位为秒
	Global bool                `json:"global,omitempty"`
	Http   map[string][]string `json:"http"`
	Https  map[string][]string `json:"https"`
}

type HostsInfoV2 struct {
	Ttl int                   `json:"ttl"` // 单位为秒
	Io  map[string]DomainList `json:"io"`
	Up  map[string]DomainList `json:"up"`
}

type DomainList struct {
	Main   []string `json:"main"`
	Backup []string `json:"backup,omitempty"`
	Info   string   `json:"info,omitempty"`
}

const (
	TooManyKeys = 700 // UC: 太多AccessKey
	NotFound    = 701 // UC: 没有发现此AccessKey
)

var (
	ETooManyKeys = api.RegisterError(TooManyKeys, "too many keys")
	ENotFound    = api.RegisterError(NotFound, "not found")
	PatternRegex = regexp.MustCompile(`^((((\*)|([-0-9a-z]+))(\.[-0-9a-z]+)+(:\d{1,5})?)|\*)$`)
)

// ------------------------------------------------------------------------------------------

type Service struct {
	Host string
	Conn rpc.Client
}

func New(host string, t http.RoundTripper) *Service {
	client := &http.Client{Transport: t}
	return &Service{host, rpc.Client{client}}
}

// ------------------------------------------------------------------------------------------

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

func (r *Service) AppInfo(app string) (info AppInfo, code int, err error) {

	params := map[string][]string{
		"app": {app},
	}
	code, err = r.Conn.CallWithForm(&info, r.Host+"/appInfo", params)
	return
}

func (r *Service) BucketInfo(bucket string) (info BucketInfo, code int, err error) {

	params := map[string][]string{
		"bucket": {bucket},
	}
	code, err = r.Conn.CallWithForm(&info, r.Host+"/bucketInfo", params)
	return
}

func (r *Service) BucketHosts(bucket string) (info HostsInfo, code int, err error) {
	code, err = r.Conn.Call(&info, r.Host+"/host/"+bucket)
	return
}

func (r *Service) RefreshBucket(bucket string) (code int, err error) {

	params := map[string][]string{
		"bucket": {bucket},
	}
	code, err = r.Conn.CallWithForm(nil, r.Host+"/refreshBucket", params)
	return
}

func (r *Service) FopAuth(bucket string, mode int, fop string) (code int, err error) {

	params := map[string][]string{
		"bucket": {bucket},
		"fop":    {fop},
		"mode":   {strconv.Itoa(mode)},
	}
	code, err = r.Conn.CallWithForm(nil, r.Host+"/fopAuth", params)
	return
}

func (r *Service) AntiLeechMode(bucket string, mode int) (code int, err error) {

	params := map[string][]string{
		"bucket": {bucket},
		"mode":   {strconv.Itoa(mode)},
	}
	code, err = r.Conn.CallWithForm(nil, r.Host+"/antiLeechMode", params)
	return
}

func (r *Service) ReferAntiLeech(bucket string, mode int, norefer int, pattern string) (code int, err error) {

	params := map[string][]string{
		"bucket":  {bucket},
		"mode":    {strconv.Itoa(mode)},
		"norefer": {strconv.Itoa(norefer)},
		"pattern": {pattern},
	}
	code, err = r.Conn.CallWithForm(nil, r.Host+"/referAntiLeech", params)
	return
}

func (r *Service) Private(bucket string, private int) (code int, err error) {

	params := map[string][]string{
		"bucket":  {bucket},
		"private": {strconv.Itoa(private)},
	}
	code, err = r.Conn.CallWithForm(nil, r.Host+"/private", params)
	return
}

func (r *Service) NoIndexPage(bucket string, noIndexPage int) (code int, err error) {

	params := map[string][]string{
		"bucket":      {bucket},
		"noIndexPage": {strconv.Itoa(noIndexPage)},
	}
	code, err = r.Conn.CallWithForm(nil, r.Host+"/noIndexPage", params)
	return
}

func (r *Service) ImgSFT(bucket string, ft int) (code int, err error) {

	params := map[string][]string{
		"bucket": {bucket},
		"imgSFT": {strconv.Itoa(ft)},
	}
	code, err = r.Conn.CallWithForm(nil, r.Host+"/imgSFT", params)
	return
}

func (r *Service) MaxAge(bucket string, maxAge int32) (code int, err error) {

	params := map[string][]string{
		"bucket": {bucket},
		"maxAge": {strconv.FormatInt(int64(maxAge), 10)},
	}
	code, err = r.Conn.CallWithForm(nil, r.Host+"/maxAge", params)
	return
}

func (r *Service) Channel(bucket string, channels []string) (code int, err error) {

	params := map[string][]string{
		"bucket":  {bucket},
		"channel": channels,
	}
	code, err = r.Conn.CallWithForm(nil, r.Host+"/channel", params)
	return
}

func (r *Service) Type(bucket string, bucketType BucketType) (code int, err error) {

	params := map[string][]string{
		"bucket": {bucket},
		"type":   {strconv.Itoa(int(bucketType))},
	}
	code, err = r.Conn.CallWithForm(nil, r.Host+"/type", params)
	return
}

func (r *Service) SetPersistFop(bucket string, persist int) (code int, err error) {

	params := map[string][]string{
		"bucket":  {bucket},
		"persist": {strconv.Itoa(persist)},
	}
	code, err = r.Conn.CallWithForm(nil, r.Host+"/persistFop", params)
	return
}

func (r *Service) TokenAntiLeech(bucket string, mode int) (code int, err error) {

	return r.Conn.Call(nil, r.Host+"/tokenAntiLeech/"+bucket+"/mode/"+strconv.Itoa(mode))
}

func (r *Service) NewMacKey(bucket string, index int) (key string, code int, err error) {

	ret := struct {
		Key string `json:"key"`
	}{}
	code, err = r.Conn.Call(&ret, r.Host+"/newMacKey/"+bucket+"/index/"+strconv.Itoa(index))
	return ret.Key, code, err
}

func (r *Service) DeleteMacKey(bucket string, index int) (code int, err error) {

	return r.Conn.Call(nil, r.Host+"/deleteMacKey/"+bucket+"/index/"+strconv.Itoa(index))
}

type AccessInfo struct {
	Key    string `json:"key"`
	Secret string `json:"secret"`
}

func (r *Service) NewAccess(app string) (info AccessInfo, code int, err error) {

	params := map[string][]string{
		"app": {app},
	}
	code, err = r.Conn.CallWithForm(&info, r.Host+"/newAccess", params)
	return
}

func (r *Service) SetKeyState(app string, accessKey string, state int) (code int, err error) {

	params := map[string][]string{
		"app":   {app},
		"key":   {accessKey},
		"state": {strconv.Itoa(state)},
	}
	code, err = r.Conn.CallWithForm(nil, r.Host+"/setKeyState", params)
	return
}

func (r *Service) DeleteAccess(app string, accessKey string) (code int, err error) {

	params := map[string][]string{
		"app": {app},
		"key": {accessKey},
	}
	code, err = r.Conn.CallWithForm(nil, r.Host+"/deleteAccess", params)
	return
}

func (r *Service) Image(bucketName string, srcSiteUrls []string, srcHost string, expires int) (code int, err error) {
	url := r.Host + "/image/" + bucketName
	url += "/from/" + rpc.EncodeURI(strings.Join(srcSiteUrls, ";"))
	if expires != 0 {
		url += "/expires/" + strconv.Itoa(expires)
	}
	if srcHost != "" {
		url += "/host/" + rpc.EncodeURI(srcHost)
	}
	return r.Conn.Call(nil, url)
}

func (r *Service) ImageWithAKSK(bucketName string, srcSiteUrls []string, srcHost string, expires int, qiniuAK, qiniuSK string) (code int, err error) {
	url := r.Host + "/image/" + bucketName
	for _, srcSiteUrl := range srcSiteUrls {
		url += "/from/" + rpc.EncodeURI(srcSiteUrl)
	}
	if expires != 0 {
		url += "/expires/" + strconv.Itoa(expires)
	}
	if srcHost != "" {
		url += "/host/" + rpc.EncodeURI(srcHost)
	}
	if qiniuAK != "" {
		url += "/qiniuak/" + rpc.EncodeURI(qiniuAK) + "/qiniusk/" + rpc.EncodeURI(qiniuSK)
	}
	return r.Conn.Call(nil, url)
}

func (r *Service) Unimage(bucketName string) (code int, err error) {
	return r.Conn.Call(nil, r.Host+"/unimage/"+bucketName)
}

func (r *Service) AccessMode(bucketName string, mode int) (code int, err error) {
	return r.Conn.Call(nil, r.Host+"/accessMode/"+bucketName+"/mode/"+strconv.Itoa(mode))
}

func (r *Service) Separator(bucketName string, sep string) (code int, err error) {
	return r.Conn.Call(nil, r.Host+"/separator/"+bucketName+"/sep/"+base64.URLEncoding.EncodeToString([]byte(sep)))
}

func (r *Service) Style(bucketName string, name string, style string) (code int, err error) {
	return r.Conn.Call(nil, r.Host+"/style/"+bucketName+"/name/"+rpc.EncodeURI(name)+"/style/"+rpc.EncodeURI(style))
}

func (r *Service) Unstyle(bucketName string, name string) (code int, err error) {
	return r.Conn.Call(nil, r.Host+"/unstyle/"+bucketName+"/name/"+rpc.EncodeURI(name))
}

func (r *Service) PreferStyleAsKey(bucketName string, prefer bool) (code int, err error) {
	return r.Conn.Call(nil, r.Host+"/preferStyleAsKey/"+bucketName+"/prefer/"+strconv.FormatBool(prefer))
}

func (r *Service) AddBucketRule(bucketName string, name, prefix string, DeleteAfterDays, ToLineAfterDays int) (code int, err error) {

	params := map[string][]string{
		"bucket":             {bucketName},
		"name":               {name},
		"prefix":             {prefix},
		"delete_after_days":  {fmt.Sprint(DeleteAfterDays)},
		"to_line_after_days": {fmt.Sprint(ToLineAfterDays)},
	}
	code, err = r.Conn.CallWithForm(nil, r.Host+"/rules/add", params)
	return
}

func (r *Service) UpdateBucketRule(bucketName, name, prefix string, DeleteAfterDays, ToLineAfterDays int) (code int, err error) {

	params := map[string][]string{
		"bucket":             {bucketName},
		"name":               {name},
		"prefix":             {prefix},
		"delete_after_days":  {fmt.Sprint(DeleteAfterDays)},
		"to_line_after_days": {fmt.Sprint(ToLineAfterDays)},
	}
	code, err = r.Conn.CallWithForm(nil, r.Host+"/rules/update", params)
	return
}

func (r *Service) DeleteBucketRule(bucketName string, name string) (code int, err error) {

	params := map[string][]string{
		"bucket": {bucketName},
		"name":   {name},
	}
	code, err = r.Conn.CallWithForm(nil, r.Host+"/rules/delete", params)
	return
}

func (r *Service) GetBucketRule(bucketName string) (rules []BucketRule, code int, err error) {

	params := map[string][]string{
		"bucket": {bucketName},
	}
	code, err = r.Conn.CallWithForm(&rules, r.Host+"/rules/get", params)
	return
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
func (r *Service) SetImagePreviewStyle(name string, style string) (code int, err error) {

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
	code, err = r.Conn.CallWithForm(nil, r.Host+"/setImagePreviewStyle", params)
	return
}

// ------------------------------------------------------------------------------------------
