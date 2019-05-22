package proto

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"regexp"
	"sync"
	"time"

	httputil "github.com/qiniu/http/httputil.v1"
	"qiniu.com/argus/dbstorage/util"
)

type ImageProcess int
type EditMode int

const (
	BASE64_PREFIX = "data:application/octet-stream;base64,"
)

const (
	REPLACE EditMode = iota
	APPEND
)

const (
	HANDLED_LAST_TIME ImageProcess = iota
	HANDLING_LAST_TIME
	NOT_HANDLED
)

var (
	ErrInvalidArgument      = httputil.NewError(http.StatusBadRequest, "invalid arguments")
	ErrEmptyGroupName       = httputil.NewError(http.StatusBadRequest, "group_name is empty")
	ErrInvalidRejectBadFace = httputil.NewError(http.StatusBadRequest, "reject_bad_face is invalid, should be true or false")
	ErrInvalidMode          = httputil.NewError(http.StatusBadRequest, "mode is invalid, should be SINGLE or LARGEST")
	ErrInvalidFileType      = httputil.NewError(http.StatusBadRequest, "only support csv and json file, file name must have extension .csv or .json")
	ErrInvalidFile          = httputil.NewError(http.StatusBadRequest, "file is not uploaded, invalid or does not have content")
	ErrTaskNotExist         = httputil.NewError(http.StatusBadRequest, "task does not exist")
	ErrTaskAlreadyStarted   = httputil.NewError(http.StatusBadRequest, "task is already started")
	ErrTaskAlreadyCompleted = httputil.NewError(http.StatusBadRequest, "task is already completed")
	ErrTaskNotStarted       = httputil.NewError(http.StatusBadRequest, "task is not running")
	ErrTaskStarted          = httputil.NewError(http.StatusBadRequest, "cannot delete, task is running")
	ErrTaskStopping         = httputil.NewError(http.StatusBadRequest, "task is stopping")
	ErrCreateTaskFail       = httputil.NewError(http.StatusInternalServerError, "fail to create task")
	ErrStartTaskFail        = httputil.NewError(http.StatusInternalServerError, "fail to start task")
	ErrDeleteTaskFail       = httputil.NewError(http.StatusInternalServerError, "fail to delete task")
	ErrTaskLost             = httputil.NewError(http.StatusInternalServerError, "cannot find task job")

	ErrFectchImage = httputil.NewError(101, "cannot fetch url content")
	ErrOpenImage   = httputil.NewError(102, "cannot open image")
	ErrDupImage    = httputil.NewError(103, "duplicated image")
	ErrIdExist     = httputil.NewError(104, "id already exists")
	ErrNoFace      = httputil.NewError(201, "no face detected")
	ErrMultiFace   = httputil.NewError(202, "multiple face detected")
	ErrSmallFace   = httputil.NewError(203, "face too small")
	ErrBigPoseFace = httputil.NewError(204, "face pose is too large")
	ErrBlurFace    = httputil.NewError(205, "face is blurred")
	ErrCoverFace   = httputil.NewError(206, "face is covered")
	ErrNotUpFace   = httputil.NewError(207, "face orientation is not up")

	RegGroupExist    = regexp.MustCompile(`group already exist`)
	RegGroupNotExist = regexp.MustCompile(`not exist`)
	RegFeatureExist  = regexp.MustCompile(`id already exists|feature is already exist`)
	RegNoFace        = regexp.MustCompile(`not face detected|No face found`)
	RegMultiFace     = regexp.MustCompile(`multiple face detected|Multiple faces found`)
	RegSmallFace     = regexp.MustCompile(`face size <|detected too small face`)
	RegBlurFace      = regexp.MustCompile(`blur`)
	RegCoverFace     = regexp.MustCompile(`covered`)
	RegPoseFace      = regexp.MustCompile(`pose`)
	RegNotUpFace     = regexp.MustCompile(`face orientation is not up`)
)

type ImageId string
type ImageURI string
type ImageTag string
type ImageDesc string

type BaseServiceConfig struct {
	Host    string        `json:"host"`
	Timeout time.Duration `json:"timeout"`
}

type TaskServiceConfig struct {
	FeatureGroupService BaseServiceConfig `json:"feature_group_service"`
	ServingService      BaseServiceConfig `json:"serving_service"`
	ThreadNum           int               `json:"thread_num"`
	MaxParallelTaskNum  int               `json:"max_parallel_task_num"`
	IsPrivate           bool              `json:"is_private"`
}

type TaskSource struct {
	Index       int
	Content     []byte
	Id          ImageId
	URI         ImageURI
	Tag         ImageTag
	Desc        ImageDesc
	Process     ImageProcess
	PreCheckErr error
}

type SafeArray struct {
	Array []int
	sync.Mutex
}

func NewSafeArray(len int) *SafeArray {
	return &SafeArray{Array: make([]int, len)}
}

type SafeCounter struct {
	Counter int
	sync.Mutex
}

func NewSafeCounter() *SafeCounter {
	return &SafeCounter{Counter: 0}
}

type SafeMap struct {
	Map map[string]struct{}
	sync.Mutex
}

func NewSafeMap() *SafeMap {
	return &SafeMap{Map: make(map[string]struct{})}
}

type SafeFile struct {
	File     *os.File
	Path     string
	Mode     EditMode
	NeedLock bool
	Mutex    sync.Mutex
}

func NewSafeFile(path string, mode EditMode) (sf *SafeFile, err error) {
	var f *os.File
	if mode == APPEND {
		f, err = os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0644)
	}
	if err != nil {
		return nil, err
	}
	return &SafeFile{
		File: f,
		Path: path,
		Mode: mode,
	}, nil
}

func (f *SafeFile) Write(msg string) (err error) {
	f.Mutex.Lock()
	defer f.Mutex.Unlock()

	if f.Mode == APPEND {
		content := fmt.Sprintf("%s\n", msg)
		_, err = f.File.Write([]byte(content))
	} else {
		err = ioutil.WriteFile(f.Path, []byte(msg), 0644)
	}
	return
}

func (f *SafeFile) ReadAll() ([]byte, error) {
	f.Mutex.Lock()
	defer f.Mutex.Unlock()

	exist, err := util.PathExists(f.Path)
	if err != nil {
		return nil, err
	}
	if exist {
		return ioutil.ReadFile(f.Path)
	}
	return nil, nil
}

func (f *SafeFile) ReadLines() ([]string, error) {
	if f.NeedLock {
		f.Mutex.Lock()
		defer f.Mutex.Unlock()
	}
	file, err := os.Open(f.Path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	res := []string{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		res = append(res, scanner.Text())
	}
	return res, nil

}

func (f *SafeFile) Close() {
	if f.File != nil {
		f.File.Close()
	}
}
