package tuso_job

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/qiniu/http/httputil.v1"
	xlog "github.com/qiniu/xlog.v1"
	job "qiniu.com/argus/bjob/proto"
	"qiniu.com/argus/tuso/hub"
	"qiniu.com/argus/tuso/io"
	"qiniu.com/argus/tuso/proto"
	"qiniu.com/argus/tuso/search"
)

type SearchTask struct {
	Features  []proto.Feature
	Hub       string
	Version   int
	TopN      int
	Threshold float32
	Kind      proto.SearchKind
	Offset    int
	Limit     int
}

type SearchTaskResult struct {
	Result [][]proto.DistanceItem
}

var _ job.JobCreator = SearchNode{}

type SearchConfig struct {
	BatchSize int `json:"batch_size"`
}
type SearchNode struct {
	SearchConfig

	proto.InternalApi
	proto.ImageFeatureApi
}

func (node SearchNode) NewMaster(ctx context.Context, reqBody []byte, env job.Env) (job.JobMaster, error) {
	var (
		searchReq proto.PostSearchJobReqJob
		xl        = xlog.FromContextSafe(ctx)
	)
	if err := json.Unmarshal(reqBody, &searchReq); err != nil {
		xl.Info("parse search request error", err)
		return nil, err
	}

	hubMeta, err := node.InternalApi.GetHubInfo(ctx, &proto.GetHubInfoReq{
		HubName: searchReq.Hub,
		Version: searchReq.Version,
	}, nil)
	if err != nil {
		xl.Infof("get hub info error: %s %d", searchReq.Hub, searchReq.Version)
		return nil, err
	}

	searchMaster := SearchMaster{
		SearchConfig:    node.SearchConfig,
		InternalApi:     node.InternalApi,
		ImageFeatureApi: node.ImageFeatureApi,
		hub:             searchReq.Hub,
		version:         searchReq.Version,
		images:          searchReq.Images,
		topN:            searchReq.TopN,
		threshold:       searchReq.Threshold,
		kind:            searchReq.Kind,
		maxOffset:       hubMeta.FeatureFileIndex * proto.KodoBlockFeatureSize,
		Mutex:           new(sync.Mutex),
	}

	searchMaster.results = make([]Result, len(searchReq.Images))
	for index, image := range searchReq.Images {
		// TODO: apply retry mechanism
		featureResp, err := node.ImageFeatureApi.PostEvalFeature(ctx, proto.PostEvalFeatureReq{
			Image: image,
		})

		if err != nil {
			xl.Error("eval feature error", err, image)
			searchMaster.results[index].Err = proto.ErrorMsg{Msg: fmt.Sprintf("get feature fail, err: %v", err)}
			searchMaster.features = append(searchMaster.features, nil)
		} else {
			if searchMaster.kind == proto.TopNSearch {
				searchMaster.results[index].Searcher = search.NewTopNSearcher(nil, searchMaster.topN)
			} else if searchMaster.kind == proto.ThresholdSearch {
				searchMaster.results[index].Searcher = search.NewThresholdSearcher(nil, searchMaster.threshold, 0)
			}
			searchMaster.features = append(searchMaster.features, featureResp.Feature)
		}
	}

	return &searchMaster, nil
}

type Result struct {
	Searcher search.Searcher
	Err      proto.ErrorMsg
}
type SearchMaster struct {
	SearchConfig

	proto.InternalApi
	proto.ImageFeatureApi

	hub      string
	version  int
	images   []proto.Image
	features []proto.Feature

	maxOffset, offset int

	topN      int
	threshold float32
	kind      proto.SearchKind

	*sync.Mutex
	results []Result
}

func (m *SearchMaster) NextTask(ctx context.Context) ([]byte, string, bool) {
	if m.offset >= m.maxOffset {
		return nil, "", false
	}

	xl := xlog.FromContextSafe(ctx)
	xl.Info("SearchMaster NextTask", m.maxOffset, m.offset, m.BatchSize)

	var offset = m.offset
	m.offset = offset + m.BatchSize
	bs, _ := json.Marshal(
		SearchTask{
			Features:  m.features,
			Hub:       m.hub,
			Version:   m.version,
			Offset:    offset,
			Limit:     m.BatchSize,
			TopN:      m.topN,
			Threshold: m.threshold,
			Kind:      m.kind,
		})
	return bs, "", true
}
func (m SearchMaster) Error(ctx context.Context) error { return nil }
func (m SearchMaster) Stop(ctx context.Context)        {}

func (m SearchMaster) AppendResult(ctx context.Context, result job.TaskResult) error {
	var (
		searchResult SearchTaskResult
		xl           = xlog.FromContextSafe(ctx)
	)
	if err := json.Unmarshal(result.Value(ctx), &searchResult); err != nil {
		xl.Info("parse search result error", err)
		return err
	}

	for _, items := range searchResult.Result {
		for index, item := range items {
			fileMetaInfo, err := m.InternalApi.GetFilemetaInfo(ctx, &proto.GetFileMetaInfoReq{
				HubName:           m.hub,
				FeatureFileIndex:  item.Index,
				FeatureFileOffset: item.Offset,
			}, nil)

			if err != nil {
				if httputil.DetectCode(err) == 404 {
					continue
				}
				xl.Info("get filemeta info error", m.hub, item.Index, item.Offset)
				continue
			}

			if fileMetaInfo.Status != string(hub.FileMetaStatusOK) {
				xl.Info("get filemeta info status", m.hub, item.Index, item.Offset, fileMetaInfo.Status)
				continue
			}

			items[index].Key = fileMetaInfo.Key
		}
	}

	m.Lock()
	defer m.Unlock()
	for index, items := range searchResult.Result {
		if items == nil || len(items) == 0 {
			continue
		}

		for _, item := range items {
			if len(item.Key) > 0 {
				m.results[index].Searcher.AppendDistanceItem(item)
			}
		}
	}

	return nil
}

func (m SearchMaster) Result(ctx context.Context) ([]byte, error) {
	var searchImageRespImages []proto.SearchImageRespImage

	for index, result := range m.results {
		var searchImageResp proto.SearchImageRespImage
		searchImageResp.OriginImage = m.images[index]

		if result.Searcher == nil {
			searchImageResp.Result.Err = result.Err
		} else {
			items := result.Searcher.SortedResult()
			for _, item := range items {
				searchImageResp.Result.Keys = append(searchImageResp.Result.Keys, item.Key)
			}
		}
		searchImageRespImages = append(searchImageRespImages, searchImageResp)
	}

	bytes, _ := json.Marshal(searchImageRespImages)
	return bytes, nil
}

//----------------------------------------------------------------------------//

type SearchWorkerConfig struct{}
type SearchWorker struct {
	SearchWorkerConfig
	hub.KodoConfig
}

func (w SearchWorker) Do(ctx context.Context, task job.Task) ([]byte, error) {
	var (
		searchTask SearchTask
		scanner    io.BlockScanner
		xl         = xlog.FromContextSafe(ctx)
	)

	if err := json.Unmarshal(task.Value(ctx), &searchTask); err != nil {
		xl.Info("parse search task error", err)
		return nil, err
	}

	searchers := make([]search.Searcher, len(searchTask.Features))
	for index, feature := range searchTask.Features {
		if feature == nil || len(feature) == 0 {
			continue
		}

		if searchTask.Kind == proto.TopNSearch {
			searchers[index] = search.NewTopNSearcher(feature, searchTask.TopN)
		} else if searchTask.Kind == proto.ThresholdSearch {
			searchers[index] = search.NewThresholdSearcher(feature, searchTask.Threshold, 0)
		}
	}

	{
		config := io.BucketConfig{
			Ak:        w.KodoConfig.Ak,
			Sk:        w.KodoConfig.Sk,
			Zone:      w.KodoConfig.Region,
			Bucket:    w.KodoConfig.Bucket,
			Domain:    w.KodoConfig.Domain,
			Prefix:    w.KodoConfig.Prefix,
			IoHost:    w.KodoConfig.IoHost,
			BlockSize: proto.KodoBlockSize,
		}
		bucket := io.NewBucket(config, searchTask.Hub, searchTask.Version, proto.FeatureSize)
		scanner = bucket.NewScanner(ctx, searchTask.Offset, searchTask.Limit)
	}

	offset := searchTask.Offset
	for scanner.Scan(ctx) {
		scannedFeature := scanner.Bytes(ctx) // one feature

		for _, searcher := range searchers {
			if searcher != nil {
				searcher.Append(proto.FeatureItem{
					Feature: scannedFeature,
					Index:   offset / proto.KodoBlockFeatureSize,
					Offset:  offset % proto.KodoBlockFeatureSize,
				})
			}
		}

		offset++
	}
	xl.Info("scan over", searchTask.Offset, searchTask.Limit, offset)

	err := scanner.Error(ctx)
	if err != nil {
		// TODO: 重试错误
		xl.Error("scanner error", err)
	}

	results := make([][]proto.DistanceItem, len(searchTask.Features))
	for index, searcher := range searchers {
		if searcher != nil {
			results[index] = searcher.Result()
		}
	}

	bs, _ := json.Marshal(SearchTaskResult{
		Result: results,
	})
	return bs, nil
}
