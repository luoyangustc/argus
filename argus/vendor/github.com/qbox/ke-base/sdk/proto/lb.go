package proto

import "time"

// Alb ....
type Alb struct {
	//warning: use dnslabel as validator is supported by adding custom validator @wanglei
	Name           string            `json:"name"`
	Region         string            `json:"region"`
	ProjectName    string            `json:"projectName"`
	Description    string            `json:"description"`
	TestDomain     string            `json:"testDomain"`
	UserDomains    string            `json:"userDomains"`
	ChargeMode     string            `json:"chargeMode"`
	BandwidthLimit int               `json:"bandwidthLimit"`
	AlbOptions     map[string]string `json:"albOptions"`
	Createtime     time.Time         `json:"created,omitempty"`
	Updatetime     time.Time         `json:"updated,omitempty"`
	Rules          []BackendRule     `json:"rules"`
	// Status string
}

const (
	// AlbChargeModeNetflow ...
	AlbChargeModeNetflow string = "netflow"
	// AlbChargeModeBandwidth ...
	AlbChargeModeBandwidth string = "bandwidth"

	CnameStatusSuccess string = "success"
	CnameStatusPending string = "pending"
)

// ----------------------------------------------------------------------------------------------------

// Domain ....
type Domain struct {
	Name            string `json:"name"`
	Region          string `json:"region"`
	ProjectName     string `json:"projectName"`
	MustCNameDomain string `json:"mustCNameDomain"`
	CnameStatus     string `json:"cnameStatus"` // "pending" or "success"
	TLSEnable       bool   `json:"tlsenable"`   //  https
	CertID          string `json:"certid"`      //  条件必填 TLSEnable时	证书 id
}

// BackendRule ....
type BackendRule struct {
	ID int64 `json:"id"`
	//warning: use dnslabel as validator is supported by adding custom validator @wanglei
	AlbName     string    `json:"albname"`
	ProjectName string    `json:"projectName"`
	HTTPPath    HTTPPath  `json:"httpPath"`
	Createtime  time.Time `json:"created,omitempty"`
	Updatetime  time.Time `json:"updated,omitempty"`
}

// HTTPPath ...
type HTTPPath struct {
	Path        string `json:"path"` //default "/"
	ServiceName string `json:"serviceName"`
	ServicePort int    `json:"servicePort"`
}

// PatchDomainOption ...
type PatchDomainOption struct {
	Name        string `json:"name"`
	Region      string `json:"region"`
	ProjectName string `json:"projectName"`
	TLSEnable   bool   `json:"tlsenable"` // https
	CertID      string `json:"certid"`    //  条件必填 TLSEnable时	证书 id
}

// PatchAlbOption ...
type PatchAlbOption struct {
	ProjectName    string             `json:"projectName"`
	Name           string             `json:"name"`
	Description    *string            `json:"description,omitempty"`
	UserDomains    *string            `json:"userDomains"`
	ChargeMode     *string            `json:"chargeMode,omitempty"`
	BandwidthLimit *int               `json:"bandwidthLimit,omitempty"`
	AlbOptions     *map[string]string `json:"albOptions,omitempty"`
	Rules          *[]BackendRule     `json:"rules,omitempty"`
}

// PatchRuleOption ...
type PatchRuleOption struct {
	ProjectName string    `json:"projectName"`
	ID          int64     `json:"id"`
	HTTPPath    *HTTPPath `json:"httpPath"`
}

// ----------------------------------------------------------------------------------------------------

type Tlb struct {
	Name           string       `json:"name" `
	Desc           string       `json:"description"`
	Project        string       `json:"projectName"`
	Status         string       `json:"status"`
	IP             string       `json:"ip"`
	IPType         string       `json:"ipType"`
	ChargeMode     string       `json:"chargeMode"`
	BandwidthLimit int64        `json:"bandwidthLimit"`
	Policy         string       `json:"policy"`
	CTime          time.Time    `json:"creationTime"`
	UTime          time.Time    `json:"updateTime"`
	Services       []TlbService `json:"services"`
	Rules          []TlbRule    `json:"rules"`
}

type TlbService struct {
	ServiceName string `json:"serviceName"`
}

type TlbRule struct {
	LBPort      int32  `json:"lbPort"`
	ServicePort int32  `json:"servicePort"`
	Protocol    string `json:"protocol"`
}

type CreateTlbArgs struct {
	Name           string       `json:"name"`
	Desc           string       `json:"description"`
	IPType         string       `json:"ipType"`
	ChargeMode     string       `json:"chargeMode"`
	BandwidthLimit int64        `json:"bandwidthLimit"`
	Policy         string       `json:"policy"`
	Services       []TlbService `json:"services"`
	Rules          []TlbRule    `json:"rules"`
}

type UpdateTlbArgs struct {
	Desc           string       `json:"description"`
	BandwidthLimit int64        `json:"bandwidthLimit"`
	Policy         string       `json:"policy"`
	Services       []TlbService `json:"services"`
	Rules          []TlbRule    `json:"rules"`
}

type ListTlbArgs struct {
	ServiceName string `json:"service"`
}
