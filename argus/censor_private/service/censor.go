package service

import (
	"context"
	"encoding/csv"
	"strconv"
	"strings"

	httputil "github.com/qiniu/http/httputil.v1"
	restrpc "github.com/qiniu/http/restrpc.v1"
	xlog "github.com/qiniu/xlog.v1"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/censor_private/dao"
	"qiniu.com/argus/censor_private/proto"
)

const DEFAULT_QUERY_LIMIT = 20

type CensorEntriesReq struct {
	dao.EntryFilter
	dao.Paginator
}

type CensorEntriesResp struct {
	Total  int            `json:"total"`
	Marker string         `json:"marker"`
	Datas  []*proto.Entry `json:"datas"`
}

func (s *Service) PostCensorEntries(
	ctx context.Context,
	req *CensorEntriesReq,
	env *restrpc.Env,
) (*CensorEntriesResp, error) {
	xl := xlog.FromContextSafe(ctx)

	if len(req.Suggestion) == 0 {
		// 默认选择所有
		req.Suggestion = proto.SuggestionAll
	}

	limit := req.Paginator.Limit
	if limit <= 0 {
		limit = DEFAULT_QUERY_LIMIT
	}

	// get entries
	entries, nextMarker, err := dao.EntryDao.Query(&req.EntryFilter, req.Paginator.Marker, limit)
	if err != nil {
		xl.Errorf("CensorEntries entryDao.Query(%#v, %#v): %v", req.EntryFilter, req.Paginator, err)
		return nil, err
	}
	for i := range entries {
		if entries[i].MimeType == proto.MimeTypeVideo {
			entries[i].CoverUri = s.proxy.URI(s.fileSaver.URI(entries[i].CoverUri))
		}
	}

	// get total count
	total, err := dao.EntryDao.Count(&req.EntryFilter)
	if err != nil {
		xl.Errorf("entryDao.Count(%#v): %v", req.EntryFilter, err)
		return nil, err
	}

	resp := &CensorEntriesResp{
		Total:  total,
		Marker: nextMarker,
		Datas:  entries,
	}
	return resp, nil
}

type CensorEntryReq struct {
	Ids        []string
	Suggestion proto.Suggestion `json:"suggestion"`
	Scenes     []proto.Scene    `json:"scenes"`
}

func (s *Service) PostCensorUpdateEntries(
	ctx context.Context,
	req *CensorEntryReq,
	env *restrpc.Env,
) error {
	xl := xlog.FromContextSafe(ctx)

	// validate
	if len(req.Ids) == 0 {
		xl.Errorf("empty ids")
		return proto.ErrEmptyIds
	}

	if !req.Suggestion.IsValid() {
		xl.Errorf("invalid suggestion: %v", req.Suggestion)
		return proto.ErrInvalidSuggestion
	}

	// validate & build final suggestion
	final := &proto.FinalSuggestion{
		Suggestion: req.Suggestion,
	}
	for _, scenes := range req.Scenes {
		if !scenes.IsValid() || !scenes.IsContained(s.config.Scenes) {
			xl.Errorf("invalid scene: %v", scenes)
			return proto.ErrInvalidScene
		}
		final.Scenes[scenes] = req.Suggestion
	}

	// update
	err := dao.EntryDao.PatchMulti(req.Ids, bson.M{"final": final})
	if err != nil {
		xl.Errorf("entryDao.PatchMulti(%#v, %#v): %v", req.Ids, final, err)
		return err
	}

	return nil
}

func (_ *Service) PostCensorEntriesDownload(
	ctx context.Context,
	req *dao.EntryFilter,
	env *restrpc.Env,
) {
	xl := xlog.FromContextSafe(ctx)

	// get entries
	if len(req.Suggestion) == 0 {
		// 默认选择所有
		req.Suggestion = proto.SuggestionAll
	}
	entries, _, err := dao.EntryDao.Query(req, "", 0)
	if err != nil {
		xl.Errorf("entryDao.Query(%#v): %v", req, err)
		httputil.Error(env.W, err)
		return
	}

	//set 名称,数据地址,鉴黄结果,鉴暴结果,鉴证结果,是否人审核过,人审结果
	header := []string{
		"set_name",
		"url",
		"pulp_result",
		"terror_result",
		"politician_result",
		"rechecked",
		"rechecked_result",
	}
	writer := csv.NewWriter(env.W)
	env.W.Header().Set("Content-Disposition", "attachment; filename="+getAttachName(req)+".csv")
	writer.Write(header)
	for _, entry := range entries {
		set, err := dao.SetCache.MustGet(entry.SetId)
		if err != nil {
			xl.Errorf("dao.SetCache.MustGet(%#v): %v", entry.SetId, err)
			httputil.Error(env.W, err)
			return
		}

		chkResult, err := entry.GetSceneSuggestions()
		if err != nil {
			xl.Errorf("failed to parse entry(%s) suggestion: %v", entry.Id, err)
			continue
		}

		var checked, result = "N", ""
		if entry.Final != nil {
			checked = "Y"
			result = string(entry.Final.Suggestion)
		}
		line := []string{
			set.Name,
			entry.Uri,
			string(chkResult[proto.ScenePulp]),
			string(chkResult[proto.SceneTerror]),
			string(chkResult[proto.ScenePolitician]),
			checked,
			result,
		}
		writer.Write(line)
	}
	writer.Flush()
}

func getAttachName(req *dao.EntryFilter) string {
	name := ""
	if req.Suggestion != "" {
		name = strings.ToUpper(string(req.Suggestion))
	} else {
		name = "ALL"
	}
	if req.StartAt > 1 && req.EndAt > 1 {
		name += "-" + strconv.FormatInt(req.StartAt, 10) + " to " + strconv.FormatInt(req.EndAt, 10)
	}
	return name
}

type CensorEntryCutsReq struct {
	CmdArgs []string
	dao.VideoCutFilter
	dao.Paginator
}

type CensorEntryCutsResp struct {
	Total  int               `json:"total"`
	Marker string            `json:"marker"`
	Datas  []*proto.VideoCut `json:"datas"`
}

func (s *Service) GetCensorEntry_Cuts(
	ctx context.Context,
	req *CensorEntryCutsReq,
	env *restrpc.Env,
) (*CensorEntryCutsResp, error) {
	xl := xlog.FromContextSafe(ctx)

	req.VideoCutFilter.EntryId = req.CmdArgs[0]
	if len(req.VideoCutFilter.EntryId) == 0 {
		return nil, proto.ErrEmptyId
	}

	if len(req.Suggestion) == 0 {
		// 默认选择所有
		req.Suggestion = proto.SuggestionAll
	}

	limit := req.Paginator.Limit
	if limit <= 0 {
		limit = DEFAULT_QUERY_LIMIT
	}

	// get cuts
	cuts, nextMarker, err := dao.VideoCutDao.Query(&req.VideoCutFilter, req.Paginator.Marker, limit)
	if err != nil {
		xl.Errorf("CensorEntry_Cuts videoCutDao.Query(%#v, %#v): %v", req.VideoCutFilter, req.Paginator, err)
		return nil, err
	}
	for i := range cuts {
		cuts[i].Uri = s.proxy.URI(s.fileSaver.URI(cuts[i].Uri))
	}

	// get total count
	total, err := dao.VideoCutDao.Count(&req.VideoCutFilter)
	if err != nil {
		xl.Errorf("entryDao.Count(%#v): %v", req.VideoCutFilter, err)
		return nil, err
	}

	resp := &CensorEntryCutsResp{
		Total:  total,
		Marker: nextMarker,
		Datas:  cuts,
	}
	return resp, nil
}
