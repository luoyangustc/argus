package feature_group

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

var ErrUpgradeVersionInvalid = errors.New("feature version invalid")
var ErrUpgradeVersionTooSmall = errors.New("feature version should greater than its current version")
var ErrUpgradeAlreadyInProgress = errors.New("feature upgrade already in-progress")
var ErrUpgradeNotFound = errors.New("no feature upgrade found")
var ErrUpgradeBadStatus = fmt.Errorf("bad status, should %v", UpgradeStatusEnum)

type UpgradeStatus string

const (
	UpgradeStatusWaiting   UpgradeStatus = "WAITING"
	UpgradeStatusUpgrading UpgradeStatus = "UPGRADING"
	UpgradeStatusFinished  UpgradeStatus = "FINISHED"
)

var UpgradeStatusEnum = []UpgradeStatus{UpgradeStatusWaiting, UpgradeStatusUpgrading, UpgradeStatusFinished}

func UpgradeStatusFromString(s string) (UpgradeStatus, error) {
	for _, v := range UpgradeStatusEnum {
		if strings.EqualFold(s, string(v)) {
			return v, nil
		}
	}
	return UpgradeStatus(""), ErrUpgradeBadStatus
}

type Upgrade struct {
	ID     string         `json:"id"`
	UID    uint32         `json:"-"`
	UType  uint32         `json:"-"`
	From   FeatureVersion `json:"from"`
	To     FeatureVersion `json:"to"`
	Status UpgradeStatus  `json:"status"`
	Error  string         `json:"error"`

	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type UpgradeInfoInMem struct {
	Upgrades []Upgrade
	*sync.Mutex
}

type CheckResult struct {
	ID          string         `json:"id"`
	Version     FeatureVersion `json:"feature_version"`
	Available   int            `json:"available"`
	Unavailable int            `json:"unavailable"`
}

func FeatureVersionCompare(fv1, fv2 FeatureVersion) int { // TODO
	if fv1 < fv2 {
		return -1
	} else if fv1 > fv2 {
		return 1
	} else {
		return 0
	}
}

/////////////////////////////////////////////////////////////////////////////

type ByteOrder int

const (
	LittleEndian ByteOrder = iota
	BigEndian
)
