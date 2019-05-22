package proto

import (
	"path/filepath"
	"strings"

	"qiniu.com/argus/censor_private/util"
)

type Suggestion string
type MimeType string
type Scene string
type SetStatus string
type SetType string
type Role string

const (
	SuggestionAll    Suggestion = "_ALL_" // all只在业务逻辑中使用，实际没有该值的Suggestion
	SuggestionPass   Suggestion = "pass"
	SuggestionReview Suggestion = "review"
	SuggestionBlock  Suggestion = "block"
)

func (s Suggestion) IsValid() bool {
	switch s {
	case SuggestionPass, SuggestionReview, SuggestionBlock:
		return true
	default:
		return false
	}
}

func MergeSuggestion(s1 Suggestion, s2 Suggestion) Suggestion {
	switch s1 {
	case SuggestionReview:
		if s2 == SuggestionBlock {
			return s2
		}
	case SuggestionPass:
		return s2
	}
	return s1
}

const (
	MimeTypeImage   MimeType = "image"
	MimeTypeVideo   MimeType = "video"
	MimeTypeUnknown MimeType = "unknown"
	MimeTypeOther   MimeType = "other"
)

var (
	ValidMimeTypes = []MimeType{MimeTypeImage, MimeTypeVideo}
)

func (t MimeType) IsValid() bool {
	switch t {
	case MimeTypeImage, MimeTypeVideo:
		return true
	default:
		return false
	}
}

func (t MimeType) IsContained(mimeTypes []MimeType) bool {
	for _, v := range mimeTypes {
		if v == t {
			return true
		}
	}
	return false
}

func GetMimeTypeWithExt(uri string) MimeType {
	ext := strings.ToLower(filepath.Ext(uri))
	if len(ext) == 0 {
		return MimeTypeUnknown
	}
	if util.ArrayContains(util.ImageExts, ext) {
		return MimeTypeImage
	}
	if util.ArrayContains(util.VideoExts, ext) {
		return MimeTypeVideo
	}

	return MimeTypeOther
}

const (
	ScenePulp       Scene = "pulp"
	SceneTerror     Scene = "terror"
	ScenePolitician Scene = "politician"
)

var (
	ValidScenes = []Scene{ScenePulp, SceneTerror, ScenePolitician}
)

func (s Scene) IsValid() bool {
	switch s {
	case ScenePulp, SceneTerror, ScenePolitician:
		return true
	default:
		return false
	}
}

func (s Scene) IsContained(scenes []Scene) bool {
	for _, v := range scenes {
		if v == s {
			return true
		}
	}
	return false
}

const (
	SetStatusRunning   SetStatus = "running"
	SetStatusStopped   SetStatus = "stopped"
	SetStatusDeleted   SetStatus = "deleted"
	SetStatusCompleted SetStatus = "completed"
)

func (s SetStatus) IsValid() bool {
	switch s {
	case SetStatusRunning, SetStatusStopped, SetStatusDeleted:
		return true
	default:
		return false
	}
}

const (
	SetTypeMonitorActive  SetType = "monitor_active"
	SetTypeMonitorPassive SetType = "monitor_passive"
	SetTypeTask           SetType = "task"
)

var AllValidMonitorType = []SetType{SetTypeMonitorActive, SetTypeMonitorPassive, SetTypeTask}

func (s SetType) IsValid() bool {
	for _, item := range AllValidMonitorType {
		if s == item {
			return true
		}
	}
	return false
}

const (
	RoleAdmin     Role = "admin"
	RoleCensor    Role = "censor"
	RoleManageSet Role = "manage_set"
)

func (r Role) IsValid() bool {
	switch r {
	case RoleAdmin, RoleCensor, RoleManageSet:
		return true
	default:
		return false
	}
}
