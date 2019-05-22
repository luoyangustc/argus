package proto

import (
	"fmt"
	"net/url"
	"time"
)

type User struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

type Project struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

type Cert struct {
	ID         string    `json:"id"`
	Name       string    `json:"name"`
	CommonName string    `json:"commonName"`
	NotBefore  time.Time `json:"notBefore"`
	NotAfter   time.Time `json:"notAfter"`
	CreateTime time.Time `json:"createTime"`
}

// UserInfo user attribute
type UserInfo struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Email string `json:"email,omitempty"`
}

// ProjectInfo attribute
type ProjectInfo struct {
	Name        string  `json:"name"`
	Description string  `json:"description"`
	Type        *string `json:"type,omitempty"`
}

// RegionInfo region attribute
type RegionInfo struct {
	ID          string `json:"id"`
	Description string `json:"description"`
}

// UpdateProjectInfo what update can update
type UpdateProjectInfo struct {
	Description string `json:"description"`
}

// UserRoleInfo user role on one project
type UserRoleInfo struct {
	User  UserInfo `json:"user"`
	Roles []string `json:"roles"`
}

type CreateProjectOpt ProjectInfo

type ValidatePassword struct {
	Password string `json:"password"`
}

// ----------------------------------------------------------------------------------------------------

// ListOption ...
type ListOption struct {
	ProjectName string  `json:"projectName"`
	Region      *string `json:"region,omitempty"`
	Name        *string `json:"name,omitempty"`
	WithRules   bool    `json:"withRules"`             // for list alb
	ServiceName *string `json:"serviceName,omitempty"` // for list albrule
	Domain      *string `json:"domain,omitempty"`      // for list alb
	ID          *int64  `json:"id,omitempty"`          // for list albrule
}

// DelOption ...
type DelOption struct {
	ProjectName string  `json:"projectName"`
	Region      *string `json:"region,omitempty"`
	Name        *string `json:"name,omitempty"`
	ID          *int64  `json:"id,omitempty"` // for delete albrule
}

func (l *ListOption) ToQuery() string {
	q := url.Values{}
	if l.Name != nil {
		q.Set("name", *l.Name)
	}
	if l.Region != nil {
		q.Set("region", *l.Region)
	}
	if l.Domain != nil {
		q.Set("domain", *l.Domain)
	}
	if l.WithRules {
		q.Set("withRules", "true")
	}
	if l.ID != nil {
		q.Set("id", fmt.Sprintf("%d", *l.ID))
	}
	if l.ServiceName != nil {
		q.Set("serviceName", *l.ServiceName)
	}
	q.Set("projectName", l.ProjectName)
	return q.Encode()
}

func (l *DelOption) ToQuery() string {
	q := url.Values{}
	if l.Name != nil {
		q.Set("name", *l.Name)
	}
	if l.Region != nil {
		q.Set("region", *l.Region)
	}
	if l.ID != nil {
		q.Set("id", fmt.Sprintf("%d", *l.ID))
	}
	return q.Encode()
}

//---------------------------------------------------------------------------------
