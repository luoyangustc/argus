package cap

import (
	"context"

	restrpc "github.com/qiniu/http/restrpc.v1"
	"qiniu.com/argus/argus/com/util"
	"qiniu.com/argus/cap/dao"
	"qiniu.com/argus/cap/model"
)

// IAdminService IAdminService
type IAdminService interface {
	////====Auditors====////
	PostAuditors(
		ctx context.Context,
		auditor *model.Auditor,
		env *restrpc.Env,
	) error
	PostAuditorsDelete_(
		ctx context.Context,
		req *struct {
			CmdArgs []string // id
		},
		env *restrpc.Env,
	) error
	GetAuditors(
		ctx context.Context,
		env *restrpc.Env,
	) ([]model.Auditor, error)
	GetAuditors_(
		ctx context.Context,
		req *struct {
			CmdArgs []string // id
		},
		env *restrpc.Env,
	) (*model.Auditor, error)

	////====LabelModes====////
	PostLabelModes(
		ctx context.Context,
		labelMode *model.LabelMode,
		env *restrpc.Env,
	) error
	PostLabelModesDelete_(
		ctx context.Context,
		req *struct {
			CmdArgs []string //modeName
		},
		env *restrpc.Env,
	) error
	GetLabelModes(
		ctx context.Context,
		env *restrpc.Env,
	) ([]model.LabelMode, error)
	GetLabelModes_(
		ctx context.Context,
		req *struct {
			CmdArgs []string //modeName
		},
		env *restrpc.Env,
	) (*model.LabelMode, error)

	////====Groups====////
	PostGroups(
		ctx context.Context,
		auditorGroup *model.AuditorGroup,
		env *restrpc.Env,
	) error
	PostGroupsDelete_(
		ctx context.Context,
		req *struct {
			CmdArgs []string // GroupID
		},
		env *restrpc.Env,
	) error
	GetGroups(
		ctx context.Context,
		env *restrpc.Env,
	) ([]model.AuditorGroup, error)
	GetGroups_(
		ctx context.Context,
		req *struct {
			CmdArgs []string // GroupID
		},
		env *restrpc.Env,
	) (*model.AuditorGroup, error)
}

// NewAdminService NewAdminService
func NewAdminService(
	labelDAO dao.ILabelDAO,
	groupDAO dao.IGroupDAO,
	auditorDAO dao.IAuditorDAO,
) (IAdminService, error) {
	return &_AdminService{
		labelDAO:   labelDAO,
		groupDAO:   groupDAO,
		auditorDAO: auditorDAO,
	}, nil
}

////////////////////////////////////////////////////////////////

type _AdminService struct {
	labelDAO   dao.ILabelDAO
	groupDAO   dao.IGroupDAO
	auditorDAO dao.IAuditorDAO
}

func (serv *_AdminService) PostAuditors(
	ctx context.Context,
	auditor *model.Auditor,
	env *restrpc.Env,
) error {
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Infof("PostAuditors req, %+v", auditor)

	err := serv.auditorDAO.Insert(ctx, dao.AuditorInMgo{
		AuditorID:  auditor.ID,
		Valid:      auditor.Valid,
		CurGroup:   auditor.CurGroup,
		AbleGroups: auditor.AbleGroups,
		SandOKNum:  auditor.SandOKNum,
		SandAllNum: auditor.SandAllNum,
	})
	if err != nil {
		xl.Errorf("PostAuditors err, %v", err)
		return err
	}
	xl.Info("PostAuditors success")
	return nil
}

func (serv *_AdminService) PostAuditorsDelete_(
	ctx context.Context,
	req *struct {
		CmdArgs []string // id
	},
	env *restrpc.Env,
) error {
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Infof("PostAuditorsDelete_ req, %+v", req)

	id := req.CmdArgs[0]
	err := serv.auditorDAO.Remove(ctx, id)
	if err != nil {
		xl.Errorf("PostAuditorsDelete_ err, %v", err)
		return err
	}
	xl.Info("PostAuditorsDelete_ success")
	return nil
}

func (serv *_AdminService) GetAuditors(
	ctx context.Context,
	env *restrpc.Env,
) ([]model.Auditor, error) {
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Info("GetAuditors")

	auditors, err := serv.auditorDAO.QueryAll(ctx)
	if err != nil {
		xl.Errorf("GetAuditors err, %v", err)
		return nil, err
	}
	xl.Infof("GetAuditors success, %d", len(auditors))

	var auditorArr []model.Auditor
	for _, ar := range auditors {
		auditorArr = append(auditorArr, model.Auditor{
			ID:         ar.AuditorID,
			Valid:      ar.Valid,
			CurGroup:   ar.CurGroup,
			AbleGroups: ar.AbleGroups,
			SandOKNum:  ar.SandOKNum,
			SandAllNum: ar.SandAllNum,
		})
	}

	return auditorArr, nil
}

func (serv *_AdminService) GetAuditors_(
	ctx context.Context,
	req *struct {
		CmdArgs []string // id
	},
	env *restrpc.Env,
) (*model.Auditor, error) {
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Infof("GetAuditors_ req, %+v", req)

	id := req.CmdArgs[0]
	auditor, err := serv.auditorDAO.QueryByAID(ctx, id)
	if err != nil {
		xl.Errorf("GetAuditors_ err, %v", err)
		return nil, err
	}
	xl.Infof("GetAuditors_ success, %+v", auditor)

	auditorRet := model.Auditor{
		ID:         auditor.AuditorID,
		Valid:      auditor.Valid,
		CurGroup:   auditor.CurGroup,
		AbleGroups: auditor.AbleGroups,
		SandOKNum:  auditor.SandOKNum,
		SandAllNum: auditor.SandAllNum,
	}

	return &auditorRet, nil
}

func (serv *_AdminService) PostLabelModes(
	ctx context.Context,
	labelMode *model.LabelMode,
	env *restrpc.Env,
) error {
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Infof("PostLabelModes req, %+v", labelMode)

	labelInMgo := model.ToLabelInMgo(labelMode)
	err := serv.labelDAO.Insert(ctx, *labelInMgo)
	if err != nil {
		xl.Errorf("PostLabelModes err, %v", err)
		return err
	}
	xl.Info("PostLabelModes success")
	return nil
}

func (serv *_AdminService) PostLabelModesDelete_(
	ctx context.Context,
	req *struct {
		CmdArgs []string //modeName
	},
	env *restrpc.Env,
) error {
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Infof("PostLabelModesDelete_ req, %+v", req)

	name := req.CmdArgs[0]
	err := serv.labelDAO.Remove(ctx, name)
	if err != nil {
		xl.Errorf("PostLabelModesDelete_ err, %v", err)
		return err
	}
	xl.Info("PostLabelModesDelete_ success")
	return nil
}

func (serv *_AdminService) GetLabelModes(
	ctx context.Context,
	env *restrpc.Env,
) ([]model.LabelMode, error) {
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Info("GetLabelModes")

	labels, err := serv.labelDAO.QueryAll(ctx)
	if err != nil {
		xl.Errorf("GetLabelModes err, %v", err)
		return nil, err
	}
	xl.Infof("GetLabelModes success, %d", len(labels))
	var labelArr []model.LabelMode
	for i, llen := 0, len(labels); i < llen; i++ {
		labelArr = append(labelArr, *model.FromLabelInMgo(labels[i]))
	}

	return labelArr, nil
}

func (serv *_AdminService) GetLabelModes_(
	ctx context.Context,
	req *struct {
		CmdArgs []string // JobID
	},
	env *restrpc.Env,
) (*model.LabelMode, error) {
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Infof("GetLabelModes_ req, %+v", req)

	id := req.CmdArgs[0]
	label, err := serv.labelDAO.QueryByName(ctx, id)
	if err != nil {
		xl.Errorf("GetLabelModes_ err, %v", err)
		return nil, err
	}
	xl.Infof("GetLabelModes success, %+v", label)
	return model.FromLabelInMgo(label), nil
}

func (serv *_AdminService) PostGroups(
	ctx context.Context,
	auditorGroup *model.AuditorGroup,
	env *restrpc.Env,
) error {
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Infof("PostGroups req, %+v", auditorGroup)

	var group dao.GroupInMgo
	group.GroupID = auditorGroup.GroupID
	group.LabelModeName = auditorGroup.Mode
	group.RealTimeLevel = auditorGroup.RealTimeLevel
	group.Level = auditorGroup.Level

	err := serv.groupDAO.Insert(ctx, group)
	if err != nil {
		xl.Errorf("PostGroups err, %v", err)
		return err
	}
	xl.Info("PostGroups success")
	return nil
}

func (serv *_AdminService) PostGroupsDelete_(
	ctx context.Context,
	req *struct {
		CmdArgs []string // GroupID
	},
	env *restrpc.Env,
) error {
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Infof("PostGroupsDelete_ req, %+v", req)

	id := req.CmdArgs[0]
	err := serv.groupDAO.Remove(ctx, id)
	if err != nil {
		xl.Errorf("PostGroupsDelete_ err, %v", err)
		return err
	}
	xl.Info("PostGroupsDelete_ success")
	return nil
}

func (serv *_AdminService) GetGroups(
	ctx context.Context,
	env *restrpc.Env,
) ([]model.AuditorGroup, error) {
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Info("GetGroups")

	groups, err := serv.groupDAO.QueryAll(ctx)
	if err != nil {
		xl.Errorf("GetGroups err, %v", err)
		return nil, err
	}
	xl.Infof("GetGroups success, %d", len(groups))
	var groupArr []model.AuditorGroup
	for i, llen := 0, len(groups); i < llen; i++ {
		var auditorGroup model.AuditorGroup
		auditorGroup.GroupID = groups[i].GroupID
		auditorGroup.Mode = groups[i].LabelModeName
		auditorGroup.RealTimeLevel = groups[i].RealTimeLevel
		auditorGroup.Level = groups[i].Level
		groupArr = append(groupArr, auditorGroup)
	}

	return groupArr, nil
}

func (serv *_AdminService) GetGroups_(
	ctx context.Context,
	req *struct {
		CmdArgs []string // JobID
	},
	env *restrpc.Env,
) (*model.AuditorGroup, error) {
	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Infof("GetGroups_ req, %+v", req)

	id := req.CmdArgs[0]
	group, err := serv.groupDAO.QueryByGID(ctx, id)
	if err != nil {
		xl.Errorf("GetGroups_ err, %v", err)
		return nil, err
	}
	xl.Infof("GetGroups_ success, %+v", group)
	return &model.AuditorGroup{
		GroupID:       group.GroupID,
		Mode:          group.LabelModeName,
		RealTimeLevel: group.RealTimeLevel,
		Level:         group.Level,
	}, nil
}
