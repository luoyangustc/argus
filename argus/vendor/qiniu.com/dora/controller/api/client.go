package api

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"qiniu.com/auth/qiniumac.v1"
	"qiniu.com/dora/controller/api/rpc"
)

const (
	AUTOSCALING_TYPE_CPU   = "CPU"
	AUTOSCALING_TYPE_MEM   = "MEMORY"
	AUTOSCALING_TYPE_REQ   = "REQUESTS"
	AUTOSCALING_TYPE_TIMER = "TIMER"
)

type Config struct {
	AccessKey string
	SecretKey string
	Host      string
	Transport http.RoundTripper
}

type Client struct {
	Config
	rpc.Client
	mac       *qiniumac.Mac
	ApiPrefix string
}

func New(cfg Config) *Client {
	p := &Client{
		Config:    cfg,
		ApiPrefix: "/v1",
	}

	p.mac = &qiniumac.Mac{cfg.AccessKey, []byte(cfg.SecretKey)}
	p.Client = rpc.Client{qiniumac.NewClient(p.mac, cfg.Transport)}
	return p
}

//-----------------------------------------------------------------------------
// Create an ufop
// POST /v1/ufops

type UfopArgs struct {
	Name string `json:"name"`
	Mode string `json:"mode"`
	Desc string `json:"desc,omitempty"` // OPTIONAL
}

func (c *Client) CreateUfop(ctx context.Context, params UfopArgs) (err error) {
	url := fmt.Sprintf("%s%s/ufops", c.Host, c.ApiPrefix)
	err = transErr(c.CallWithJson(ctx, nil, "POST", url, params))
	return
}

//-----------------------------------------------------------------------------
// Delete an ufop
// DELETE /v1/ufops/<ufop>

func (c *Client) DeleteUfop(ctx context.Context, ufop string) (err error) {
	url := fmt.Sprintf("%s%s/ufops/%s", c.Host, c.ApiPrefix, ufop)
	err = transErr(c.Call(ctx, nil, "DELETE", url))
	return
}

//-----------------------------------------------------------------------------
// Get ufop list
// GET /v1/ufops

type UfopInfo struct {
	Desc string `json:"desc"`
	Mode string `json:"mode"`
	Name string `json:"name"`
}

func (c *Client) GetUfops(ctx context.Context) (ufops []UfopInfo, err error) {
	url := fmt.Sprintf("%s%s/ufops", c.Host, c.ApiPrefix)
	err = transErr(c.Call(ctx, &ufops, "GET", url))
	return
}

//-----------------------------------------------------------------------------
// Get brief ufop info
// GET /v1/ufops/<ufop>/info/brief

type UfopBrief struct {
	Mode    string    `json:"mode"`
	Release []Release `json:"releases"`
}

func (c *Client) GetUfopInfoBrief(ctx context.Context, ufop string) (brief UfopBrief, err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/info/brief", c.Host, c.ApiPrefix, ufop)
	err = transErr(c.Call(ctx, &brief, "GET", url))
	return
}

//-----------------------------------------------------------------------------
// Get detailed ufop info
// GET /v1/ufops/<ufop>/info/detail

type UfopDetail struct {
	Mode          string          `json:"mode"`
	ReleaseDetail []ReleaseDetail `json:"releases"`
}

func (c *Client) GetUfopInfoDetail(ctx context.Context, ufop string) (detail UfopDetail, err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/info/detail", c.Host, c.ApiPrefix, ufop)
	err = transErr(c.Call(ctx, &detail, "GET", url))
	return
}

//-----------------------------------------------------------------------------
// List Flavors
// GET /v1/flavors

type Flavor struct {
	Name string `json:"name"`
	Cpu  uint   `json:"cpu"`
	Mem  uint   `json:"memory"`
	Disk string `json:"disk"`
	Gpu  string `json:"gpu,omitempty"` // OPTIONAL format: <model>:core  e.g.: K80:1
}

func (c *Client) ListFlavors(ctx context.Context) (flavors []Flavor, err error) {
	url := fmt.Sprintf("%s%s/flavors", c.Host, c.ApiPrefix)
	err = transErr(c.Call(ctx, &flavors, "GET", url))
	return
}

//-----------------------------------------------------------------------------
// List Regions
// GET /v1/regions

type Region struct {
	Name string `json:"name"`
	Desc string `json:"desc,omitempty"` // OPTIONAL
}

func (c *Client) ListRegions(ctx context.Context) (regions []Region, err error) {
	url := fmt.Sprintf("%s%s/regions", c.Host, c.ApiPrefix)
	err = transErr(c.Call(ctx, &regions, "GET", url))
	return
}

//-----------------------------------------------------------------------------
// Create Release
// POST /v1/ufops/<ufop>/releases

type HealthCheck struct {
	Path    string `json:"path"`
	Timeout uint   `json:"timeout"` // OPTIONAL unit: second, default 3s
}

type EnvVariable struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

type ReleaseArgs struct {
	Verstr  string `json:"verstr"`         // user-defined version string
	Desc    string `json:"desc,omitempty"` // OPTIONAL
	Image   string `json:"image"`
	Flavor  string `json:"flavor"`
	MntPath string `json:"mount_path"` // OPTIONAL
	Port    uint   `json:"port"`       // OPTIONAL

	HealthCk     HealthCheck              `json:"health_check,omitempty"` // OPTIONAL
	Env          map[string][]EnvVariable `json:"env,omitempty"`
	LogFilePaths []string                 `json:"log_file_paths,omitemtpy"` // OPTIONAL
}

func (c *Client) CreateRelease(ctx context.Context, ufop string, args ReleaseArgs) (err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/releases", c.Host, c.ApiPrefix, ufop)
	err = transErr(c.CallWithJson(ctx, nil, "POST", url, args))
	return
}

//-----------------------------------------------------------------------------
// Get brief ufop release
// GET /v1/ufops/<ufop>/releases/<verstr>/brief

type Release struct {
	Verstr  string `json:"verstr"`         // user-defined version string
	Desc    string `json:"desc,omitempty"` // OPTIONAL
	Image   string `json:"image"`
	Flavor  string `json:"flavor"`
	Network string `json:"network"`

	HealthCk     HealthCheck              `json:"health_check,omitempty"`   // OPTIONAL
	Env          map[string][]EnvVariable `json:"env,omitempty"`            // OPTIONAL
	LogFilePaths []string                 `json:"log_file_paths,omitemtpy"` // OPTIONAL
	Ctime        time.Time                `json:"ctime"`                    // auto-generated by Controller, UnixNano
}

func (c *Client) GetReleaseInfoBrief(ctx context.Context, ufop string, version string) (brief Release, err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/releases/%s/brief", c.Host, c.ApiPrefix, ufop, version)
	err = transErr(c.Call(ctx, &brief, "GET", url))
	return
}

//-----------------------------------------------------------------------------
// Get detailed ufop release
// GET /v1/ufops/<ufop>/releases/<verstr>/detail

type Runtime struct {
	Region string `json:"region"`
	Expect uint   `json:"expect"` // expected number of instances
	Actual uint   `json:"actual"` // actual number of Running instances
}

type ReleaseDetail struct {
	Release
	Instances []Runtime `json:"instances"`
}

func (c *Client) GetReleaseInfoDetail(ctx context.Context, ufop string, version string) (detail ReleaseDetail, err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/releases/%s/detail", c.Host, c.ApiPrefix, ufop, version)
	err = transErr(c.Call(ctx, &detail, "GET", url))
	return
}

//-----------------------------------------------------------------------------
// resize instance num
// POST /v1/ufops/<ufop>/deployments

type DeploymentArgs struct {
	Verstr string `json:"verstr"`
	Region string `json:"region"`
	Expect uint   `json:"expect"`
}

type DeploymentRet struct {
	DeploymentID string `json:"deployment_id"`
}

func (c *Client) CreateDeployment(ctx context.Context, app string, args DeploymentArgs) (deployment DeploymentRet, err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/deployments", c.Host, c.ApiPrefix, app)
	err = transErr(c.CallWithJson(ctx, &deployment, "POST", url, args))
	return
}

//-----------------------------------------------------------------------------
// get all instances state
// GET /v1/ufops/<ufop>/deployments

type Deployment struct {
	Id      string    `json:"deployment_id"`
	Verstr  string    `json:"verstr"`
	Region  string    `json:"region"`
	Origin  uint      `json:"origin"`  // auto-filled by Controller, origin number of instances
	Expect  uint      `json:"expect"`  // expected number of instances
	Ctime   time.Time `json:"ctime"`   // auto-generated by Controller, UnixNano
	Status  string    `json:"status"`  // auto-updated by Controller
	Message string    `json:"message"` // auto-updated by Controller
}

func (c *Client) ListDeployments(ctx context.Context, ufop string) (deployments []Deployment, err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/deployments", c.Host, c.ApiPrefix, ufop)
	err = transErr(c.Call(ctx, &deployments, "GET", url))
	return
}

//-----------------------------------------------------------------------------
// get an instance state
// GET /v1/ufops/<ufop>/deployments/<deployment_id>

func (c *Client) GetDeployment(ctx context.Context, ufop string, dep_id string) (deployment Deployment, err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/deployments/%s", c.Host, c.ApiPrefix, ufop, dep_id)
	err = transErr(c.Call(ctx, &deployment, "GET", url))
	return
}

//-----------------------------------------------------------------------------
// get an image token
// GET /v1/images/token?image_name=<string>

type ImageToken struct {
	ImageName       string `json:"image_name"`
	RegistryAddress string `json:"registry_address"`
	Token           string `json:"token"`
}

type ImageTokenArgs struct {
	ImageName string
	Scope     string
}

func (c *Client) GetImageToken(ctx context.Context, args ImageTokenArgs) (imageToken ImageToken, err error) {
	url := fmt.Sprintf("%s%s/images/token?image_name=%s&scope=%s", c.Host, c.ApiPrefix, url.QueryEscape(args.ImageName), url.QueryEscape(args.Scope))
	err = transErr(c.Call(ctx, &imageToken, "GET", url))
	return
}

type Image struct {
	Name  string    `json:"name"`
	Tag   string    `json:"tag"`
	Ctime time.Time `json:"ctime"`
}

func (c *Client) ListImages(ctx context.Context) (images []Image, err error) {
	url := fmt.Sprintf("%s%s/images", c.Host, c.ApiPrefix)
	err = transErr(c.Call(ctx, &images, "GET", url))
	return
}

//-----------------------------------------------------------------------------
// Create exec

func (c *Client) CreateExec(ctx context.Context, ufop, region, instId string) (execId string, err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/regions/%s/instances/%s/exec", c.Host, c.ApiPrefix, ufop, region, instId)
	err = transErr(c.Call(ctx, &execId, "POST", url))
	return
}

//-----------------------------------------------------------------------------
// Resize exec tty

func (c *Client) ResizeExecTTY(ctx context.Context, ufop, region, instId, execId string, h, w int) (err error) {
	args := struct {
		H int `json:"h"`
		W int `json:"w"`
	}{
		H: h,
		W: w,
	}
	url := fmt.Sprintf("%s%s/ufops/%s/regions/%s/instances/%s/exec/%s/resize", c.Host, c.ApiPrefix, ufop, region, instId, execId)
	err = transErr(c.CallWithJson(ctx, nil, "POST", url, args))
	return
}

type LogSearchArgs struct {
	From       string `json:"from"`
	To         string `json:"to"`
	InstanceID string `json:"instance_id"`
	Keyword    string `json:"keyword"`
	Size       string `json:"size"`
	Type       string `json:"type"`
}

type Record struct {
	InstanceID string `json:"instance_id"`
	Message    string `json:"message"`
	Timestamp  string `json:"timestamp"`
	Type       string `json:"type"`
}

type LogSearchRet struct {
	Records []Record `json:"records"`
}

//-----------------------------------------------------------------------------
// Log search

func (c *Client) GetLogsSearch(ctx context.Context, ufop, release, region string, args LogSearchArgs) (logSearch LogSearchRet, err error) {
	v := url.Values{}
	v.Add("from", args.From)
	v.Add("to", args.To)
	v.Add("instance_id", args.InstanceID)
	v.Add("keyword", args.Keyword)
	v.Add("size", args.Size)
	v.Add("type", args.Type)
	_url := fmt.Sprintf("%s%s/ufops/%s/releases/%s/regions/%s/logs/search?%s", c.Host, c.ApiPrefix, ufop, release, region, v.Encode())
	err = transErr(c.Call(ctx, &logSearch, "GET", _url))
	return
}

func transErr(originErr error) (err error) {
	//if ErrorInfo, ok := originErr.(*rpc.ErrorInfo); ok {
	//return appsRPC.NewHTTPError(ErrorInfo.Code, ErrorInfo.Err, ErrorInfo.Desc)
	//}
	return originErr
}

//-----------------------------------------------------------------------------
// Apply an ufop
// POST /v1/ufops/<ufop>/apply

func (c *Client) ApplyUfop(ctx context.Context, ufop string) (err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/apply", c.Host, c.ApiPrefix, ufop)
	err = transErr(c.Call(ctx, nil, "POST", url))
	return
}

//-----------------------------------------------------------------------------
// Unapply an ufop
// POST /v1/ufops/<ufop>/unapply

func (c *Client) UnApplyUfop(ctx context.Context, ufop string) (err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/unapply", c.Host, c.ApiPrefix, ufop)
	err = transErr(c.Call(ctx, nil, "POST", url))
	return
}

//-----------------------------------------------------------------------------
// /v1/ufops/<ufop>/instances?region=<string>&release_ver=<string>

type UfopsInstancesListArgs struct {
	Region     string `json:"region"`
	ReleaseVer string `json:"release_ver"`
}

type UfopsInstancesListRet struct {
	Region    string          `json:"region"`
	Verstr    string          `json:"verstr"`
	Instances []UfopsInstance `json:"instances"`
}

type UfopsInstance struct {
	Ctime  time.Time `json:"ctime"`
	ID     string    `json:"id"`
	Status string    `json:"status"`
}

func (c *Client) GetUfopInstances(ctx context.Context, ufop string, args UfopsInstancesListArgs) (instances []UfopsInstancesListRet, err error) {
	v := url.Values{}
	v.Add("region", args.Region)
	v.Add("release_ver", args.ReleaseVer)
	url := fmt.Sprintf("%s%s/ufops/%s/instances?%s", c.Host, c.ApiPrefix, ufop, v.Encode())
	err = transErr(c.Call(ctx, &instances, "GET", url))
	return
}

//-----------------------------------------------------------------------------
// GET /v1/ufops/<ufop>/monitors?metric=<string>&release_ver=<string>&region=<string>&instance_id=<string>&from=<uint>&to=<uint>&step=<uint>

type GetMonitorsArg struct {
	Metric     string `json:"metric"`
	From       string `json:"from"`
	To         string `json:"to"`
	Step       string `json:"step"`
	InstanceID string `json:"instance_id"`
	Region     string `json:"region"`
	Release    string `json:"release_ver"`
}

type PrometheusQueryData struct {
	Metric map[string]string `json:"metric"`
	Values [][2]interface{}  `json:"values"`
}

func (c *Client) GetUfopMonitors(ctx context.Context, ufop string, args GetMonitorsArg) (ret []PrometheusQueryData, err error) {
	v := url.Values{}

	v.Add("metric", args.Metric)
	v.Add("from", args.From)
	v.Add("to", args.To)
	v.Add("step", args.Step)
	v.Add("instance_id", args.InstanceID)
	v.Add("region", args.Region)
	v.Add("release_ver", args.Release)

	url := fmt.Sprintf("%s%s/ufops/%s/monitors?%s", c.Host, c.ApiPrefix, ufop, v.Encode())
	err = transErr(c.Call(ctx, &ret, "GET", url))
	return
}

//-----------------------------------------------------------------------------
// POST /v1/ufops

type UfopChangeDescArgs struct {
	Desc string `json:"desc,omitempty"` // OPTIONAL
}

func (c *Client) ChangeUfopDesc(ctx context.Context, ufop string, params UfopChangeDescArgs) (err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/desc", c.Host, c.ApiPrefix, ufop)
	err = transErr(c.CallWithJson(ctx, nil, "POST", url, params))
	return
}

// POST /v1/ufops/<ufop>/autoscalings/cpu

type AutoscalCpuArgs struct {
	Region       string `json:"region"`         // effective region
	Release      string `json:"release"`        // bind scaling to one release version
	MinNum       uint   `json:"min_num"`        // OPTIONAL minimal number of instances
	MaxNum       uint   `json:"max_num"`        // OPTIONAL maximum number of instances
	Step         uint   `json:"step"`           // OPTIONAL 1-5 default 1. scale up/down step
	ScaleUpCpu   uint   `json:"scale_up_cpu"`   // 50-100 scale up condition
	ScaleDownCpu uint   `json:"scale_down_cpu"` // 0-50 scale down condition
}

func (c *Client) SetUfopAutoscalingCpu(ctx context.Context, ufop string, args AutoscalCpuArgs) (err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/autoscalings/cpu", c.Host, c.ApiPrefix, ufop)
	err = transErr(c.CallWithJson(ctx, nil, "POST", url, args))
	return
}

//-----------------------------------------------------------------------------
// POST /v1/ufops/<ufop>/autoscalings/memory

type AutoscalMemArgs struct {
	Region       string `json:"region"`         // effective region
	Release      string `json:"release"`        // bind scaling to one release version
	MinNum       uint   `json:"min_num"`        // OPTIONAL minimal number of instances
	MaxNum       uint   `json:"max_num"`        // OPTIONAL maximum number of instances
	Step         uint   `json:"step"`           // OPTIONAL 1-5 default 1. scale up/down step
	ScaleUpMem   uint   `json:"scale_up_mem"`   // 50-100 scale up condition
	ScaleDownMem uint   `json:"scale_down_mem"` // 0-50 scale down condition
}

func (c *Client) SetUfopAutoscalingMem(ctx context.Context, ufop string, args AutoscalMemArgs) (err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/autoscalings/memory", c.Host, c.ApiPrefix, ufop)
	err = transErr(c.CallWithJson(ctx, nil, "POST", url, args))
	return
}

//-----------------------------------------------------------------------------
// POST /v1/ufops/<ufop>/autoscalings/requests

type AutoscalReqsArgs struct {
	Region        string `json:"region"`          // effective region
	Release       string `json:"release"`         // bind scaling to one release version
	MinNum        uint   `json:"min_num"`         // OPTIONAL minimal number of instances
	MaxNum        uint   `json:"max_num"`         // OPTIONAL maximum number of instances
	Step          uint   `json:"step"`            // OPTIONAL 1-5 default 1. scale up/down step
	ScaleUpReqs   uint   `json:"scale_up_reqs"`   // scale up condition
	ScaleDownReqs uint   `json:"scale_down_reqs"` // scale down condition
}

func (c *Client) SetUfopAutoscalingRequests(ctx context.Context, ufop string, args AutoscalReqsArgs) (err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/autoscalings/requests", c.Host, c.ApiPrefix, ufop)
	err = transErr(c.CallWithJson(ctx, nil, "POST", url, args))
	return
}

//-----------------------------------------------------------------------------
// POST /v1/ufops/<ufop>/autoscalings/timer

type AutoscalTimeTriggerArgs struct {
	From   string `json:"from"`   // start time e.g. 19:30
	To     string `json:"to"`     // end time e.g. 23:00
	Expect uint   `json:"expect"` // expected final number of instances
}

type AutoscalTimerArgs struct {
	Region      string                    `json:"region"`       // effective region
	Release     string                    `json:"release"`      // bind scaling to one release version
	Default     uint                      `json:"default"`      // default number of instances during unspecified time secitons
	TimeTrigger []AutoscalTimeTriggerArgs `json:"time_trigger"` // triggers (based on time) for scale up and down
}

func (c *Client) SetUfopAutoscalingTimer(ctx context.Context, ufop string, args AutoscalTimerArgs) (err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/autoscalings/timer", c.Host, c.ApiPrefix, ufop)
	err = transErr(c.CallWithJson(ctx, nil, "POST", url, args))
	return
}

//-----------------------------------------------------------------------------
// POST /v1/ufops/<ufop>/autoscalings/<type>/switch?action=<string enable|disable>&region=<string>

type AutoscalActionArgs struct {
	Action string `json:"action"` // "enable" or "disable"
	Region string `json:"region"`
}

func (c *Client) SwitchUfopAutoscalings(ctx context.Context, ufop string, asType string, args AutoscalActionArgs) (err error) {
	v := url.Values{}
	v.Add("action", args.Action)
	v.Add("region", args.Region)
	url := fmt.Sprintf("%s%s/ufops/%s/autoscalings/%s/switch?%s", c.Host, c.ApiPrefix, ufop, asType, v.Encode())
	err = transErr(c.Call(ctx, nil, "POST", url))
	return
}

//-----------------------------------------------------------------------------
// GET /v1/ufops/<ufop>/autoscalings

type AutoscalCpu struct {
	Enabled      bool      `json:"enabled"`
	MinNum       uint      `json:"min_num"`
	MaxNum       uint      `json:"max_num"`
	Step         uint      `json:"step"`
	ScaleUpCpu   uint      `json:"scale_up_cpu"`
	ScaleDownCpu uint      `json:"scale_down_cpu"`
	Ctime        time.Time `json:"ctime"`
	Mtime        time.Time `json:"mtime"`
}

type AutoscalMem struct {
	Enabled      bool      `json:"enabled"`
	MinNum       uint      `json:"min_num"`
	MaxNum       uint      `json:"max_num"`
	Step         uint      `json:"step"`
	ScaleUpMem   uint      `json:"scale_up_mem"`
	ScaleDownMem uint      `json:"scale_down_mem"`
	Ctime        time.Time `json:"ctime"`
	Mtime        time.Time `json:"mtime"`
}

type AutoscalReqs struct {
	Enabled       bool      `json:"enabled"`
	MinNum        uint      `json:"min_num"`
	MaxNum        uint      `json:"max_num"`
	Step          uint      `json:"step"`
	ScaleUpReqs   uint      `json:"scale_up_reqs"`
	ScaleDownReqs uint      `json:"scale_down_reqs"`
	Ctime         time.Time `json:"ctime"`
	Mtime         time.Time `json:"mtime"`
}

type AsTimeTrigger struct {
	From   string `json:"from"`
	To     string `json:"to"`
	Expect uint   `json:"expect"`
}

type AutoscalTimer struct {
	Enabled     bool            `json:"enabled"`
	Default     uint            `json:"default"`
	TimeTrigger []AsTimeTrigger `json:"time_trigger"`
	Ctime       time.Time       `json:"ctime"`
	Mtime       time.Time       `json:"mtime"`
}

type Autoscaling struct {
	Ufop    string         `json:"ufop"`
	Region  string         `json:"region"`
	Release string         `json:"release"`
	Type    string         `json:"type"`
	ByCpu   *AutoscalCpu   `json:"by_cpu,omitempty"`
	ByMem   *AutoscalMem   `json:"by_mem,omitempty"`
	ByReqs  *AutoscalReqs  `json:"by_reqs,omitempty"`
	ByTimer *AutoscalTimer `json:"by_timer,omitempty"`
}

func (c *Client) GetUfopAutoscalings(ctx context.Context, ufop string) (autoscalings []Autoscaling, err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/autoscalings", c.Host, c.ApiPrefix, ufop)
	err = transErr(c.Call(ctx, &autoscalings, "GET", url))
	return
}

//-----------------------------------------------------------------------------
// DELETE /v1/ufops/<ufop>/autoscalings/<type>?region=<string>

func (c *Client) DeleteUfopAutoscalings(ctx context.Context, ufop string, asType string, region string) (err error) {
	v := url.Values{}
	v.Add("region", region)
	url := fmt.Sprintf("%s%s/ufops/%s/autoscalings/%s?%s", c.Host, c.ApiPrefix, ufop, asType, v.Encode())
	err = transErr(c.Call(ctx, nil, "DELETE", url))
	return
}

//-----------------------------------------------------------------------------
// GET /v1/ufops/<ufop>/autoscalings/logs?region=<string>&page_num=<int>&page_size=<int>from=<int>&to=<int>

type UfopsAutoscalingsLogsArgs struct {
	Region   string `json:"region"`
	PageNum  string `json:"page_num"`
	PageSize string `json:"page_size"`
	From     string `json:"from"` // optional
	To       string `json:"to"`   // optional
}

type ScalingLog struct {
	Region      string    `json:"region"`
	Verstr      string    `json:"verstr"`
	TriggerType string    `json:"trigger_type"`
	Actual      uint      `json:"actual"`
	Expect      uint      `json:"expect"`
	Metric      string    `json:"metric"`
	Status      string    `json:"status"`
	Ctime       time.Time `json:"ctime"`
}

type UfopsAutoscalingsLogsRet struct {
	Total int          `json:"total"`
	Logs  []ScalingLog `json:"logs"`
}

func (c *Client) GetUfopAutoscalingsLogs(ctx context.Context, ufop string, args UfopsAutoscalingsLogsArgs) (ret UfopsAutoscalingsLogsRet, err error) {
	v := url.Values{}
	v.Add("region", args.Region)
	v.Add("page_num", args.PageNum)
	v.Add("page_size", args.PageSize)
	v.Add("from", args.From)
	v.Add("to", args.To)
	url := fmt.Sprintf("%s%s/ufops/%s/autoscalings/logs?%s", c.Host, c.ApiPrefix, ufop, v.Encode())
	err = transErr(c.Call(ctx, &ret, "GET", url))
	return
}

//-----------------------------------------------------------------------------
// switch official status
// POST /v1/ufops/<ufop>/official

type SwitchOfficialArgs struct {
	Official bool `json:"official"`
}

func (c *Client) SwitchOfficalStatus(ctx context.Context, ufopName string, params SwitchOfficialArgs) (err error) {
	url := fmt.Sprintf("%s%s/ufops/%s/official", c.Host, c.ApiPrefix, ufopName)
	err = transErr(c.CallWithJson(ctx, nil, "POST", url, params))
	return
}
