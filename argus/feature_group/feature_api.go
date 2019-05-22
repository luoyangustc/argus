package feature_group

import (
	"sync"

	"github.com/pkg/errors"
)

var ErrFeatureAPINotFound = errors.New("feature api not found")

const EmptyFeatureVersion = FeatureVersion("")

type FeatureVersion string

type FeatureAPI interface{}

type FeatureAPIs interface {
	Get(FeatureVersion) (FeatureAPI, error)
	Default() (FeatureVersion, FeatureAPI, error)
	Current() (FeatureVersion, FeatureAPI, error)
	SetCurrent(FeatureVersion) error
	Reset(FeatureVersion, FeatureAPI)
}

type featureAPIs struct {
	featureAPIs map[FeatureVersion]FeatureAPI
	dftVersion  FeatureVersion
	curVersion  FeatureVersion
	*sync.RWMutex
}

var _ FeatureAPIs = (*featureAPIs)(nil)

func NewFeatureAPIs(apis map[FeatureVersion]FeatureAPI, dftVersion, curVersion FeatureVersion) FeatureAPIs {
	_featureAPIs := &featureAPIs{
		featureAPIs: make(map[FeatureVersion]FeatureAPI),
		dftVersion:  dftVersion,
		curVersion:  curVersion,
		RWMutex:     new(sync.RWMutex),
	}

	for k, v := range apis {
		_featureAPIs.featureAPIs[k] = v
	}

	return _featureAPIs
}

func (f *featureAPIs) Get(version FeatureVersion) (FeatureAPI, error) {
	f.RLock()
	defer f.RUnlock()

	if api, ok := f.featureAPIs[version]; ok {
		return api, nil
	}
	return nil, ErrFeatureAPINotFound
}

func (f *featureAPIs) Default() (FeatureVersion, FeatureAPI, error) {
	f.RLock()
	defer f.RUnlock()

	if api, ok := f.featureAPIs[f.dftVersion]; ok {
		return f.dftVersion, api, nil
	}
	return EmptyFeatureVersion, nil, ErrFeatureAPINotFound
}

func (f *featureAPIs) Current() (FeatureVersion, FeatureAPI, error) {
	f.RLock()
	defer f.RUnlock()

	if api, ok := f.featureAPIs[f.curVersion]; ok {
		return f.curVersion, api, nil
	}
	return EmptyFeatureVersion, nil, ErrFeatureAPINotFound
}

func (f *featureAPIs) SetCurrent(version FeatureVersion) error {
	f.Lock()
	defer f.Unlock()

	if _, ok := f.featureAPIs[version]; !ok {
		return ErrFeatureAPINotFound
	}

	f.curVersion = version
	return nil
}

func (f *featureAPIs) Reset(version FeatureVersion, api FeatureAPI) {
	f.Lock()
	defer f.Unlock()

	if api == nil {
		delete(f.featureAPIs, version)
	} else {
		f.featureAPIs[version] = api
	}
}
