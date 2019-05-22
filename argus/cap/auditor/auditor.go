package auditor

import (
	// "container/list"
	"context"
	"encoding/json"
	"errors"
	// "fmt"
	// "gopkg.in/mgo.v2"
	"math/rand"
	"strconv"
	"strings"
	// "sync"
	"time"

	"github.com/qiniu/uuid"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/cap/dao"
	"qiniu.com/argus/cap/enums"
	"qiniu.com/argus/cap/model"
	"qiniu.com/argus/cap/sand"
)

// IAuditor ...
////////////////////////////////////////////////////////////////////////////////
type IAuditor interface {
	GetAuditorAttr(context.Context, string) (*model.GetAuditorAttrResp, error)
	FetchTasks(context.Context, string) (model.GetRealtimeTaskResp, error)
	CancelTasks(context.Context, string, []string, string) error
	SaveTasks(context.Context, string, []model.TaskResult, string) error
}

////////////////////////////////////////////////////////////////////////////////

var _ IAuditor = &_Auditor{}

type _Auditor struct {
	model.AuditorConfig
	dao.ITaskDAO
	dao.IAuditorDAO
	dao.IGroupDAO
	dao.ILabelDAO
	sand.ISandMixer
	chCancel map[string]chan string
	// sync.Locker
}

// NewAuditor new
func NewAuditor(taskDAO dao.ITaskDAO, auditorDAO dao.IAuditorDAO, groupDAO dao.IGroupDAO,
	labelDAO dao.ILabelDAO, sandMixer sand.ISandMixer, config model.AuditorConfig) IAuditor {

	if config.IntervalSecs <= 0 {
		config.IntervalSecs = 2
	}
	if config.MaxTasksNum <= 0 {
		config.MaxTasksNum = 2
	}
	if config.SingleTimeoutSecs <= 0 {
		config.SingleTimeoutSecs = 10
	}
	if config.PackSize <= 0 {
		config.PackSize = 2
	}
	if config.NoSandLimitint <= 0 {
		config.NoSandLimitint = 2
	}
	if config.RecordReserveSecond <= 0 {
		config.RecordReserveSecond = 3600 * 12
	}
	return &_Auditor{
		AuditorConfig: config,
		ITaskDAO:      taskDAO,
		IAuditorDAO:   auditorDAO,
		IGroupDAO:     groupDAO,
		ILabelDAO:     labelDAO,
		ISandMixer:    sandMixer,
		chCancel:      map[string]chan string{},
		// Locker:        new(sync.Mutex),
	}
}

func (a *_Auditor) GetAuditorAttr(ctx context.Context, aid string) (*model.GetAuditorAttrResp, error) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	auditorInfo, err := a.getAuditorInfo(ctx, aid)
	if err != nil {
		xl.Errorf("a.getAuditorInfo error: %#v", err.Error())
		return nil, err
	}

	return &model.GetAuditorAttrResp{
		AuditorID:     aid,
		Valid:         auditorInfo.Valid,
		RealTimeLevel: model.RealTimeLevel, // Tips: 统一使用LabelX实时模版
		CurLabel:      auditorInfo.LabelModeName,
	}, nil
}

func (a *_Auditor) FetchTasks(ctx context.Context, aid string) (model.GetRealtimeTaskResp, error) {
	var (
		xl       = xlog.FromContextSafe(ctx)
		packSize = a.AuditorConfig.PackSize
	)
	auditorInfo, err := a.getAuditorInfo(ctx, aid)
	if err != nil {
		xl.Errorf("a.getAuditorInfo for auditor %s error: %#v", aid, err.Error())
		return model.GetRealtimeTaskResp{}, err
	}

	xl.Infof("begin assignTasks for auditor: %#v", auditorInfo.AuditorID)

	//TODO: fix used for Sand
	// generate noSandTasks number within one stand package
	notSandCount := 0
	for i := 0; i < packSize; i++ {
		if rand.Intn(100) > a.SandPercentage {
			notSandCount++
		}
	}
	sandCount := packSize - notSandCount
	// xl.Infof("FetchTasks step 1 - dispatch algorithm:  NOT sand tasks num: %v; sands tasks num: %v ", notSandCount, sandCount)

	// obtain tasks from mongo db
	taskInMgo, err := a.ITaskDAO.FetchTasksForAuditor(ctx, auditorInfo.LabelModeName, notSandCount)
	if err != nil {
		xl.Warnf("a.ITaskDAO.FetchTasksForAuditor error: %#v", err.Error())
		return model.GetRealtimeTaskResp{}, err
	}
	// if no tasks obtained, then return
	if len(taskInMgo) <= 0 {
		xl.Warnf("Quite with no tasks assigned")
		return model.GetRealtimeTaskResp{}, nil
	}

	//xl.Infof("========> FetchTasks step 2 - get NOT sands tasks: %v", len(tasks))
	// 在labelX出口处增加沙子
	if notSandCount > a.AuditorConfig.NoSandLimitint && sandCount > 0 {
		sandTasks := []*model.TaskModel{}
		labelTypes, err := a.getCurLabelMode(auditorInfo.LabelModeName)
		if err != nil {
			xl.Warnf("get current label err:", err)
		} else {
			xl.Infof("get current label :%v", labelTypes)
			countArr, err := a.getRandsBySum(sandCount, len(labelTypes))
			if err != nil {
				xl.Warn(err)
			} else {
				for i, labelType := range labelTypes {
					sands := a.ISandMixer.QuerySandsByType(ctx, countArr[i], labelType)
					xl.Infof("get label:%s sand num :%d", labelType, countArr[i])

					if len(sands) > 0 {
						sandTasks = append(sandTasks, sands...)
					}
				}
			}
		}
		//	xl.Infof("========> assign task step 3: get sand task assigned: %v", len(sandTasks))
		if len(sandTasks) > 0 {
			for _, sand := range sandTasks {
				taskInMgo = append(taskInMgo, *model.ToTaskInMgo(sand))
			}
		}
	}

	// wrapping response package
	expireDuration := int64(len(taskInMgo)) * a.AuditorConfig.SingleTimeoutSecs
	pid := genUUID()
	packResp, taskIDs := a.genPackResp(ctx, &auditorInfo, taskInMgo, expireDuration, pid)
	xl.Infof("Package %v is assigned to %v", pid, aid)

	a.chCancel[pid] = make(chan string)
	go func() {
		// time.Sleep(time.Second * time.Duration(expireDuration))
		// if task is set as done before this cancel, the cancellation will do noting,
		// as it only focuses on the tasks with DOING status

		tcTimer := time.NewTimer(time.Second * time.Duration(expireDuration))

		for {
			select {
			case <-tcTimer.C:
				err := a.ITaskDAO.CancelTasks(ctx, auditorInfo.LabelModeName, taskIDs)
				if err != nil {
					xl.Errorf("a.CancelTasks error: %#v, %#v", aid, taskIDs)
				}
				xl.Infof("Overdue Package %v Canceled", pid)
				close(a.chCancel[pid])
				delete(a.chCancel, pid)
				return
			case msg := <-a.chCancel[pid]:
				// when user cancel or submit result, will trigger this action to exit gorouting
				xl.Infof("channel closed by: %v; originaly created by %v", msg, aid)
				close(a.chCancel[pid])
				delete(a.chCancel, pid)
				return
			}
		}
	}()

	return packResp, nil
}

func (a *_Auditor) CancelTasks(ctx context.Context, aid string, tids []string, pid string) error {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	auditorInfo, err := a.getAuditorInfo(ctx, aid)
	if err != nil {
		xl.Errorf("a.getAuditorInfo error: %#v", err.Error())
		return err
	}
	// close useless channel
	if _, ok := a.chCancel[pid]; ok {
		a.chCancel[pid] <- aid
		xl.Infof("Package %v Cancelled Success by Aid: %#v", pid, aid)
		return a.ITaskDAO.CancelTasks(ctx, auditorInfo.LabelModeName, tids)
	}

	xl.Infof("Package %v No Longer Exists: %#v", pid, aid)
	return nil
}

func (a *_Auditor) SaveTasks(ctx context.Context, aid string, tasks []model.TaskResult, pid string) error {
	xl := xlog.FromContextSafe(ctx)
	xl.Infof("begin SaveTasks from auditor:, %v", aid)

	auditorInfo, err := a.getAuditorInfo(ctx, aid)
	if err != nil {
		return err
	}

	for _, task := range tasks {
		// sand check
		if a.ISandMixer.IsSand(ctx, task.TaskID) {
			xl.Infof("Sand Checking ... Task Is Sand: %v", task.TaskID)

			isOK := a.ISandMixer.Check(ctx, &task)
			am, _ := a.IAuditorDAO.QueryByAID(ctx, aid)
			now := time.Now()
			record := dao.SandRecord{
				Time:   now,
				TaskID: task.TaskID,
			}
			am.SandAllNum++

			if isOK {
				am.SandOKNum++
				record.Correct = 1
				// 沙子验证通过
				// _TaskCorrectGauge.Inc()
			}
			//修复CI错误，先全部注释掉
			// else {
			// 	// _TaskIncorrectGauge.Inc()
			// }
			am.SandRecords = append(am.SandRecords, record)
			// 沙子计数更新到DB中
			err := a.IAuditorDAO.Update(ctx, am)
			xl.Infof("Update Auditor Sand Num: %v, %v", am, err)
			continue
		}

		// not sand, update to db
		// check the status of task, if one task is settled back to TODO, then the whole package
		// should be treated as expired
		taskInMgo, err := a.ITaskDAO.QueryByID(ctx, auditorInfo.LabelModeName, task.TaskID)
		if taskInMgo.Status != enums.TaskDoing || err != nil {
			return err
		}

		taskInMgo.Status = enums.TaskDone
		taskInMgo.AuditorID = aid
		result, err := json.Marshal(task.Labels)
		if err != nil {
			xl.Errorf("json.Marshal error: %#v", err.Error())
			return err
		}
		taskInMgo.Result = result
		err = a.ITaskDAO.Update(ctx, auditorInfo.LabelModeName, taskInMgo)
		if err != nil {
			return err
		}
	}

	// close useless channel
	if _, ok := a.chCancel[pid]; ok {
		a.chCancel[pid] <- aid
	}
	return nil
}

//===============================================================================

func (a *_Auditor) getAuditorInfo(ctx context.Context, aid string) (model.AuditorModel, error) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	am, err := a.IAuditorDAO.QueryByAID(ctx, aid)
	if err != nil {
		xl.Errorf("a.IAuditorDAO.QueryByAID error: %#v", err.Error())
		return model.AuditorModel{}, err
	}

	group, err := a.IGroupDAO.QueryByGID(ctx, am.CurGroup)
	if err != nil {
		xl.Errorf("a.IGroupDAO.QueryByGID error: %#v", err.Error())
		return model.AuditorModel{}, err
	}

	mode, err := a.ILabelDAO.QueryByName(ctx, group.LabelModeName)
	if err != nil {
		xl.Errorf("a.ILabelDAO.QueryByName error: %#v", err.Error())
		return model.AuditorModel{}, err
	}
	aLabels := make(map[string][]model.LabelTitle)
	for k, v := range mode.Labels {
		labelTiles := make([]model.LabelTitle, 0)
		for _, v1 := range v {
			labelTiles = append(labelTiles, model.LabelTitle{
				Title:    v1.Title,
				Desc:     v1.Desc,
				Selected: v1.Selected,
			})
		}
		aLabels[k] = labelTiles
	}

	return model.AuditorModel{
		AuditorID:     aid,
		Valid:         am.Valid,
		CurGroupID:    am.CurGroup,
		LabelModeName: group.LabelModeName,
		LabelTypes:    mode.LabelTypes,
		Labels:        aLabels,
	}, nil
}

func (a *_Auditor) genPackResp(ctx context.Context, aInfo *model.AuditorModel, tasks []dao.TaskInMgo, expireDuration int64, pid string) (model.GetRealtimeTaskResp, []string) {
	var (
		taskIDs []string
	)
	packResp := model.GetRealtimeTaskResp{
		AuditorID:  aInfo.AuditorID,
		PID:        pid,
		Mode:       model.RealTimeLevel,                                                                          // Tips: 统一使用LabelX实时模版
		Type:       string(enums.MimeTypeImage),                                                                  //TODO:添加视频
		ExpiryTime: strconv.FormatInt(time.Now().Add(time.Second*time.Duration(expireDuration)).Unix()*1000, 10), // return milliseconds
		TaskType:   aInfo.LabelTypes,
		Labels:     aInfo.Labels,
	}

	//添加task的信息
	for _, task := range tasks {
		tModel := model.FromTaskInMgo(&task)
		packResp.IndexData = append(packResp.IndexData, *model.ToTaskResult(tModel))
		taskIDs = append(taskIDs, string(task.TaskID))
	}

	return packResp, taskIDs
}

//======================================================================================
//For Sand
func (a *_Auditor) getCurLabelMode(LabelModeName string) ([]string, error) {

	splits := strings.Split(LabelModeName, "_")
	if len(splits) >= 2 {
		return splits[1:], nil
	}
	return []string{}, errors.New("not found")
}

func (a *_Auditor) getRandsBySum(sum int, num int) ([]int, error) {
	if sum < 0 || num <= 0 {
		return []int{}, errors.New("error input")
	}
	randArr := []int{}
	randSum := 0
	for i := 0; i < num; i++ {
		r := rand.Intn(100) + 1
		randSum += r
		randArr = append(randArr, r)
	}
	ret := []int{}
	curSum := 0
	for _, n := range randArr[:num-1] {
		m := sum * n / randSum
		curSum += m
		ret = append(ret, m)
	}
	ret = append(ret, sum-curSum)
	return ret, nil
}

// GenUUID for pid
func genUUID() string {
	u, err := uuid.Gen(16)
	if err != nil {
		return ""
	}
	return u
}
