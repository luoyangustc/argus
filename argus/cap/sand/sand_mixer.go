package sand

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"regexp"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/cap/model"
	// "qiniu.com/argus/cap/utils"
)

const (
	SandHeadStr = "QINIU_SAND_"
)

type SandInfoFromFile struct {
	URL   string            `json:"url"`
	Label []model.LabelInfo `json:"label"`
}

// Sand 沙子模型
type Sand struct {
	SandID string `json:"sand_id"` // 对应TaskID
	URI    string `json:"uri"`
}

// SandMixer 沙子工具
type ISandMixer interface {
	AddSand(ctx context.Context, sands ...model.TaskModel) error
	// 沙子使用特殊的taskID规则
	IsSand(ctx context.Context, taskID string) bool

	// 检测沙子结果
	Check(ctx context.Context, result *model.TaskResult) bool
	//根据label类型获取task
	QuerySandsByType(ctx context.Context, num int, typeName string) []*model.TaskModel

	AddSandFileByURL(sandFileUrl string) error
}

// NewSandMixer NewSandMixer
func NewSandMixer(pattern string) ISandMixer {
	return &_SandMixer{
		pattern:      pattern,
		sandMap:      make(map[string]*model.TaskModel),
		sandGroupMap: make(map[string][]string),
		// curNum:       0,
	}
}

////////////////////////////////////////////////////////////////

type _SandMixer struct {
	pattern      string
	sandMap      map[string]*model.TaskModel
	sandGroupMap map[string][]string
	// curNum       int
}

func (mixer *_SandMixer) AddSand(ctx context.Context, sands ...model.TaskModel,
) error {

	xl := xlog.FromContextSafe(ctx)
	xl.Infof("AddSand num: %d", len(sands))

	curlen := len(mixer.sandMap)
	for _, sand := range sands {
		sand0 := sand
		sand0.JobID = "QINIU_SAND"
		if sand0.TaskID == "" {
			sand0.TaskID = fmt.Sprintf("%s%d", SandHeadStr, curlen)
		}
		mixer.sandMap[sand0.TaskID] = &sand0
		mixer.add2SandGroup(ctx, &sand0)
		// xl.Infof("AddSand: %v", sand0)
		curlen++
	}

	return nil
}

func (mixer *_SandMixer) add2SandGroup(ctx context.Context, sand *model.TaskModel) {
	// xl := xlog.FromContextSafe(ctx)

	for _, label := range sand.Labels {
		if _, ok := mixer.sandGroupMap[label.Name]; !ok {
			mixer.sandGroupMap[label.Name] = []string{}
		}
		// xl.Infof("Add to SandGroup ,label name:%s ,taskid:%s", label.Name, sand.TaskID)
		mixer.sandGroupMap[label.Name] = append(mixer.sandGroupMap[label.Name], sand.TaskID)
	}
}

func (mixer *_SandMixer) IsSand(ctx context.Context, taskID string) bool {

	xl := xlog.FromContextSafe(ctx)
	xl.Infof("pattern: %s, taskID: %s", mixer.pattern, taskID)

	if mixer.pattern == "" {
		xl.Infof("MatchString: %v", false)
		// no pattern => no Match
		return false
	}

	isM, err := regexp.MatchString(mixer.pattern, taskID)
	if err != nil {
		xl.Errorf("MatchString err, %v", err)
		return false
	}
	xl.Infof("MatchString: %v", isM)
	return isM
}

func (mixer *_SandMixer) getRandN(n int, max int) []int {
	if n < 10 {
		rand.Seed(time.Now().UnixNano())
		ret := []int{}
		for ; n > 0; n-- {
			ret = append(ret, rand.Intn(max))
		}
		return ret
	}

	var ret []int
	for {
		if n <= 0 {
			break
		}
		pick := n
		if pick > max {
			pick = max
		}
		randN, err := RandomSampleN(0, max, pick)
		if err != nil {
			break
		}
		ret = append(ret, randN...)
		n -= pick
	}
	return ret
}

func (mixer *_SandMixer) QuerySandsByType(ctx context.Context, num int, typeName string) []*model.TaskModel {
	xl := xlog.FromContextSafe(ctx)

	xl.Infof("QuerySandsByType, num:%d, typeName:%s ", num, typeName)
	ret := []*model.TaskModel{}

	if _, ok := mixer.sandGroupMap[typeName]; ok {
		slen := len(mixer.sandGroupMap[typeName])
		randIds := mixer.getRandN(num, slen)
		for _, id := range randIds {
			if task, ok := mixer.sandMap[mixer.sandGroupMap[typeName][id]]; ok {
				ret = append(ret, task)
			}
		}
	} else {
		xl.Infof("can not find sand list of %s", typeName)
		return ret
	}

	xl.Infof("GetSands, %v", ret)
	return ret
}

func (mixer *_SandMixer) Check(ctx context.Context, result *model.TaskResult) bool {

	xl := xlog.FromContextSafe(ctx)
	xl.Infof("Check, %v", result)

	if result == nil {
		return false
	}

	sand, ok := mixer.sandMap[result.TaskID]
	if !ok {
		xl.Errorf("GetSand err, %s not found", result.TaskID)
		return false
	}
	xl.Infof("GetSand, %v", sand)

	// 只有沙子库中和result中都有的label不一样时候才为False
	isOK := true
	for _, slabel := range sand.Labels {
		isOK = true
		for _, retlabel := range result.Labels {
			if retlabel.Type == slabel.Type &&
				retlabel.Name == slabel.Name {
				// 某个data值匹配失败
				if !compareDatas(slabel.Name, slabel.Data, retlabel.Data) {
					isOK = false
					break
				}
			}
		}
		if !isOK { // 某个label一直未匹配上
			break
		}
	}

	xl.Infof("Check isOK = %v", isOK)
	return isOK
}

func (mixer *_SandMixer) AddSandFileByURL(sandFileUrl string) error {
	ctx := context.Background()
	xl := xlog.FromContextSafe(ctx)

	resp, err := http.Get(sandFileUrl)
	if err != nil {
		xl.Errorf("Open err: %v", err)
		return err
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)

	var sandArr []model.TaskModel

	sandNum := 0
	for scanner.Scan() {
		sandInfoFromFile := SandInfoFromFile{}
		err := json.Unmarshal(scanner.Bytes(), &sandInfoFromFile)
		if err != nil {
			xl.Errorf("Open err: %v", err)
			return err
		}
		sand := model.TaskModel{
			URI:    sandInfoFromFile.URL,
			Labels: sandInfoFromFile.Label,
		}
		sandArr = append(sandArr, sand)
		sandNum++
		if len(sandArr) >= 100 {
			err := mixer.AddSand(ctx, sandArr...)
			if err != nil {
				xl.Errorf("addSandToCap err: %v", err)
				return err
			}
			sandArr = []model.TaskModel{}
		}
	}

	// send last
	if len(sandArr) > 0 {
		err := mixer.AddSand(ctx, sandArr...)
		if err != nil {
			xl.Errorf("addSandToCap err: %v", err)
			return err
		}
	}

	xl.Infof("addSandToCap OK: %d", sandNum)
	return nil
}

func RandomSampleN(start, end, n int) ([]int, error) {
	len := end - start

	if start < 0 || end < start || end-start < n || n < 0 {
		err := errors.New("Wrong RandomSampleN parameter")
		log.Println(err)
		return nil, err
	}
	rand.Seed(time.Now().Unix())
	var arr []int
	for i := 0; i < len; i++ {
		arr = append(arr, start+i)
	}

	for i := 0; i < n; i++ {
		j := rand.Intn(len-i) + i
		arr[i], arr[j] = arr[j], arr[i]
	}

	return arr[:n], nil
}

//=================================================================================//
func compareDatas(name string, data1 interface{}, data2 interface{}) bool {
	switch name {
	case "pulp", "terror":
		sDatas := make([]model.LabelData, 0)
		err := parseLabelData(data1, &sDatas)
		if err != nil {
			return false
		}
		rDatas := make([]model.LabelData, 0)
		err = parseLabelData(data2, &rDatas)
		if err != nil {
			return false
		}

		for _, v1 := range sDatas {
			res := false
			for _, v2 := range rDatas {
				if v1.Class == v2.Class {
					res = true
					break
				}
			}
			if !res {
				return false
			}
		}

	case "politician":
		sDatas := make([]model.LabelPoliticianData, 0)
		err := parseLabelData(data1, &sDatas)
		if err != nil {
			return false
		}
		rDatas := make([]model.LabelPoliticianData, 0)
		err = parseLabelData(data2, &rDatas)
		if err != nil {
			return false
		}

		for _, v1 := range sDatas {
			res := false
			for _, v2 := range rDatas {
				if v1.Class == v2.Class {
					res = true
					break
				}
			}
			if !res {
				return false
			}
		}
	}

	return true
}

func parseLabelData(src interface{}, dest interface{}) error {
	tmpbs, err := json.Marshal(src)
	if err != nil {
		return err
	}

	return json.Unmarshal(tmpbs, dest)
}
