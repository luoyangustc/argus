package censor

import (
	"context"
	"encoding/json"

	"github.com/qiniu/xlog.v1"

	job "qiniu.com/argus/bjob/proto"
	"qiniu.com/argus/bjob/workers"
	"qiniu.com/argus/censor/biz"
	"qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
)

var _ job.TaskWorker = VideoWorker{}

type VideoParams struct {
	Scenes []biz.Scene `json:"scenes,omitempty"`
	Params struct {
		Scenes  map[biz.Scene]json.RawMessage `json:"scenes"`
		Vframe  *vframe.VframeParams          `json:"vframe"`
		Save    json.RawMessage               `json:"save,omitempty"`
		HookURL string                        `json:"hookURL"`
	} `json:"params,omitempty"`
}

type VideoWorker struct {
	workers.InferenceVideoWorker
}

func NewVideoWorker(worker workers.InferenceVideoWorker) VideoWorker {
	return VideoWorker{InferenceVideoWorker: worker}
}

func (w VideoWorker) Do(ctx context.Context, task job.Task) ([]byte, error) {
	var (
		xl     = xlog.FromContextSafe(ctx)
		_task  Task
		result []byte
	)

	if err := json.Unmarshal(task.Value(ctx), &_task); err != nil {
		xl.Info("parse task error", err)
		return nil, err
	}

	var req = VideoParams{}
	_ = json.Unmarshal(_task.Params, &req)
	if len(req.Scenes) == 0 {
		req.Scenes = biz.DefaultScenes
	}

	var req1 = video.VideoRequest{}
	req1.Data.URI = _task.URI
	req1.Params.Vframe = req.Params.Vframe
	if req.Params.Save != nil {
		req1.Params.Save = &req.Params.Save
	}
	for _, scene := range req.Scenes {
		req1.Ops = append(req1.Ops, struct {
			OP             string         `json:"op"`
			CutHookURL     string         `json:"cut_hook_url"`
			SegmentHookURL string         `json:"segment_hook_url"`
			HookURL        string         `json:"hookURL"`
			Params         video.OPParams `json:"params"`
		}{OP: string(scene)})
		switch scene {
		case biz.POLITICIAN:
			req1.Ops[len(req1.Ops)-1].Params.Other = struct {
				All bool `json:"all"`
			}{All: true}
		}
	}
	bs, _ := json.Marshal(req1)

	bs, _ = json.Marshal(workers.InferenceVideoTask{
		UID: _task.UID, Utype: _task.Utype,
		URI:    _task.URI,
		Params: bs,
	})
	bs, err := w.InferenceVideoWorker.Do(ctx, NewTask(bs))
	if err != nil {
		return nil, err
	}

	var resp = map[string]biz.OriginVideoOPResult{}
	err = json.Unmarshal(bs, &resp)
	if err != nil {
		return nil, err
	}

	var ret = biz.CensorResponse{Suggestion: biz.PASS, Scenes: map[biz.Scene]interface{}{}}
	for op, ret0 := range resp {
		switch op {
		case "pulp":
			ret1 := biz.ParseOriginVideoOPResult(ret0, func(cut biz.OriginCutResult) biz.CutResult {
				return biz.ParseCutPulpResult(cut, biz.PulpThreshold{})
			})
			ret.Suggestion = ret.Suggestion.Update(ret1.Suggestion)
			ret.Scenes[biz.PULP] = ret1
		case "terror":
			ret1 := biz.ParseOriginVideoOPResult(ret0, func(cut biz.OriginCutResult) biz.CutResult {
				return biz.ParseCutTerrorResult(cut, biz.TerrorThreshold{})
			})
			ret.Suggestion = ret.Suggestion.Update(ret1.Suggestion)
			ret.Scenes[biz.TERROR] = ret1
		case "politician":
			ret1 := biz.ParseOriginVideoOPResult(ret0, func(cut biz.OriginCutResult) biz.CutResult {
				return biz.ParseCutPoliticianResult(cut, biz.PoliticianThreshold{})
			})
			ret.Suggestion = ret.Suggestion.Update(ret1.Suggestion)
			ret.Scenes[biz.POLITICIAN] = ret1
		}
	}

	result, _ = json.Marshal(ret)
	return result, nil
}
