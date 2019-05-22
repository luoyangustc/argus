package proto

//##############################ArgusVideo BEGIN##############################

type ArgusVideoRequest struct {
	Data   UriData         `json:"data"`
	Params ParamsArgsVideo `json:"params,omitempty"`
	Ops    []OpsArgsVideo  `json:"ops"`
}

type ArgusLiveRequest struct {
	Data   UriData        `json:"data"`
	Params ParamsArgsLive `json:"params,omitempty"`
	Ops    []OpsArgsVideo `json:"ops"`
}

type UriData struct {
	Uri string `json:"uri"`
}

type ParamsArgsVideo struct {
	Async bool `json:"async,omitempty"`
	// Segment SegmentParams `json:"segment,omitempty"`
	Vframe  VframeParams `json:"vframe,omitempty"`
	HookURL string       `json:"hookURl,omitempty"`
	Save    SaveParams   `json:"save,omitempty"`
}

type ParamsArgsLive struct {
	Async bool `json:"async,omitempty"`
	// Segment SegmentParams `json:"segment,omitempty"`
	Live    LiveParams   `json:"live,omitempty"`
	Vframe  VframeParams `json:"vframe,omitempty"`
	HookURL string       `json:"hookURl,omitempty"`
	Save    SaveParams   `json:"save,omitempty"`
}

type LiveParams struct {
	Timeout    float32 `json:"timeout,omitempty"`
	DownStream string  `json:"downstream,omitempty"`
}

type SaveParams struct {
	Bucket string `json:"bucket,omitempty"`
	Zone   int    `json:"zone,omitempty"`
	Prefix string `json:"prefix,omitempty"`
}

type VframeParams struct {
	Mode     int `json:"mode"`
	Interval int `json:"interval,omitempty"`
}

type OpsArgsVideo struct {
	Op         string    `json:"op"`
	HookURL    string    `json:"hookURL,omitempty"`
	CutHookURL string    `json:"cut_hook_url,omitempty"`
	Params     ParamsOps `json:"params,omitempty"`
}

type ParamsOps struct {
	Labels              []LabelsParams  `json:"labels,omitempty"`
	Terminate           TerminateParams `json:"terminate,omitempty"`
	Other               interface{}     `json:"other,omitempty"`
	Ignore_empty_labels bool            `json:"ignore_empty_labels,omitempty"`
}

type OtherParams struct {
	Groups    []string `json:"groups,omitempty"`
	Cluster   string   `json:"cluster,omitempty"`
	Detail    bool     `json:"detail,omitempty"`
	All       bool     `json:"all,omitempty"`
	Threshold float32  `json:"threshold,omitempty"`
	Limit     int      `json:"limit,omitempty"`
}

type LabelsParams struct {
	Label  string  `json:"label,omitempty"`
	Select int     `json:"select,omitempty"`
	Score  float64 `json:"score,omitempty"`
}

type TerminateParams struct {
	Mode   int            `json:"mode,omitempty"`
	Labels map[string]int `json:"labels,omitempty"`
}

//type ArgusVideoResponse map[string]OpResult
type ArgusVideoJob struct {
	Id      string                    `json:"id"`
	Vid     string                    `json:"vid"`
	Request ArgusVideoRequest         `json:"request"`
	Status  string                    `json:"status"`
	Result  map[string]VideoOpsResult `json:"result"`
}

type ArgusVideoCall struct {
	Id     string                    `json:"id"`
	Result map[string]VideoOpsResult `json:"result"`
}

type VideoOpsResult struct {
	Code    int           `json:"code"`
	Message string        `json:"message"`
	Result  VideoOpResult `json:"result"`
}

type VideoOpResult struct {
	Labels   []Label   `json:"labels"`
	Segments []Segment `json:"segments"`
}

type Label struct {
	Label string  `json:"label,op"`
	Score float64 `json:"score"`
}

type Segment struct {
	Offset_begin int     `json:"offset_begin"`
	Offset_end   int     `json:"offset_end"`
	Labels       []Label `json:"labels"`
	Cuts         []Cut   `json:"cuts"`
}

type Cut struct {
	Offset int         `json:"offset"`
	Uri    string      `json:"uri"`
	Result interface{} `json:"result"`
}

//##############################ArgusVideo END##############################

func NewArgusVideoRequest(uri string, op string) (req *ArgusVideoRequest) {
	request := &ArgusVideoRequest{
		Data: UriData{
			Uri: uri,
		},
		Ops: []OpsArgsVideo{
			{
				Op: op,
			},
		},
	}
	request.Ops[0].Params.Terminate.Labels = make(map[string]int)
	return request
}

func NewArgusLiveRequest(uri string, op string) (req *ArgusLiveRequest) {
	request := &ArgusLiveRequest{
		Data: UriData{
			Uri: uri,
		},
		Ops: []OpsArgsVideo{
			{
				Op: op,
			},
		},
	}
	request.Ops[0].Params.Terminate.Labels = make(map[string]int)
	return request
}

func (a *ArgusVideoRequest) SetVframe(mode int, interval int) {
	a.Params.Vframe.Mode = mode
	a.Params.Vframe.Interval = interval
}

func (a *ArgusVideoRequest) SetDetail(pos int, detail bool) {
	var other OtherParams
	other.Detail = detail
	a.Ops[pos].Params.Other = other
}

func (a *ArgusVideoRequest) SetDetailString(pos int, detail string) {
	var other struct {
		Detail string `json:"detail,omitempty"`
	}
	other.Detail = detail
	a.Ops[pos].Params.Other = other
}

func (a *ArgusVideoRequest) SetCluster(pos int, cluster string) {
	var other OtherParams
	other.Cluster = cluster
	a.Ops[pos].Params.Other = other
}

func (a *ArgusVideoRequest) SetGroup(pos int, group string) {
	var other OtherParams
	other.Groups = append(other.Groups, group)
	a.Ops[pos].Params.Other = other
}

func (a *ArgusVideoRequest) AddLabel(label string, sselect int, score float64, pos int) {
	a.Ops[pos].Params.Labels = append(a.Ops[pos].Params.Labels, LabelsParams{Label: label, Select: sselect, Score: score})
}

func (a *ArgusVideoRequest) SetTerminateMode(pos int, mode int) {
	a.Ops[pos].Params.Terminate.Mode = mode
}

func (a *ArgusVideoRequest) AddTerminateLabels(pos int, label string, score int) {
	a.Ops[pos].Params.Terminate.Labels[label] = score
}

func (a *ArgusVideoRequest) SetAsync(f bool) {
	a.Params.Async = f
}

func (a *ArgusVideoRequest) ChangeVideoUrl(uri string) {
	a.Data.Uri = uri
}

func (a *ArgusVideoRequest) SetIgnore(ignore_empty_labels bool, pos int) {
	a.Ops[pos].Params.Ignore_empty_labels = ignore_empty_labels
}

func (a *ArgusVideoRequest) SetLimit(limit int, pos int) {
	var other OtherParams
	other.Limit = limit
	a.Ops[pos].Params.Other = other
}
func (a *ArgusLiveRequest) SetLiveVframe(mode int, interval int) {
	a.Params.Vframe.Mode = mode
	a.Params.Vframe.Interval = interval
}

func (a *ArgusLiveRequest) SetLiveHookUrl(domain string, pos int) {
	a.Params.HookURL = domain
	a.Ops[pos].CutHookURL = domain
}

func (a *ArgusLiveRequest) SetLimitWithGroup(limit int, groups []string, pos int) {
	var other OtherParams
	other.Limit = limit
	other.Groups = groups
	a.Ops[pos].Params.Other = other
}

func (a *ArgusLiveRequest) SetLive(timeout float32) {
	// a.Params.Live.DownStream = downstream
	a.Params.Live.Timeout = timeout
}

//##############################FUNC BEGIN##############################
