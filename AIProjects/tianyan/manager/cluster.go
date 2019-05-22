package manager

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"strings"

	httputil "github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"qiniu.com/argus/AIProjects/tianyan/serving"
)

const (
	defaultClusterSize      = 10000
	defaultClusterTimeout   = 3600 * 24 // 24hours
	defaultSearchThreshold  = 0.7
	defaultClusterThreshold = 0.65
)

// -----------------------------------------------------------------------------------
type postFaceCluster_NewReq struct {
	CmdArgs   []string
	Size      int   `json:"size"`
	Dimension int   `json:"dimension,omitempty"`
	Precision int   `json:"precision,omitempty"`
	Timeout   int64 `json:"timeout,omitempty"`
}

func (s *Service) PostFaceCluster_New(ctx context.Context, args *postFaceCluster_NewReq, env *restrpc.Env) (err error) {
	xl, ctx := s.initContext(ctx, env)

	var (
		cluster = args.CmdArgs[0]
	)

	if len(cluster) == 0 {
		return httputil.NewError(http.StatusBadRequest, "empty face cluster name")
	}

	if args.Timeout < 0 {
		return httputil.NewError(http.StatusBadRequest, "invalid timeout param")
	}

	req := serving.FSCreateReq{Name: cluster, Size: args.Size, Precision: args.Precision, Dimension: args.Dimension, State: GroupInitialized, Timeout: args.Timeout}
	err = s.FeatureSearch.Create(ctx, req)
	if err != nil {
		xl.Errorf("fail to create feature cluster %s, size: %d, precision: %d, dimension: %d, err: %v", cluster, args.Size, args.Precision, args.Dimension, err)
		return
	}
	xl.Debugf("Create feature cluster %s, size: %d, precision: %d, dimension: %d, timeout: %ds", cluster, args.Size, args.Precision, args.Dimension, args.Timeout)

	return
}

// -----------------------------------------------------------------------------------
func (s *Service) PostFaceCluster_Remove(ctx context.Context, args *BaseReq, env *restrpc.Env) (err error) {
	xl, ctx := s.initContext(ctx, env)
	var (
		cluster = args.CmdArgs[0]
	)

	if len(cluster) == 0 {
		return httputil.NewError(http.StatusBadRequest, "empty face cluster name")
	}
	if err = s.FeatureSearch.Destroy(ctx, cluster); err != nil {
		xl.Errorf("PostFaceCluster_Remove: call feature-search.Destroy failed, error: %v", err)
		return
	}
	return
}

// -----------------------------------------------------------------------------------

func (s *Service) PostFaceCluster_Search(ctx context.Context, args *BaseReq, env *restrpc.Env) {
	xl, ctx := s.initContext(ctx, env)

	var (
		req faceSearchReq
	)

	groups := strings.Split(args.CmdArgs[0], ",")
	req.Params.Cluster = groups[0]
	groups = groups[1:]

	switch ct := env.Req.Header.Get(CONTENT_TYPE); {
	case IsJsonContent(ct):
		bs, err := ioutil.ReadAll(args.ReqBody)
		defer args.ReqBody.Close()
		if err != nil {
			xl.Warnf("read requests body failed. %v", err)
			httputil.ReplyErr(env.W, http.StatusBadRequest, err.Error())
			return
		}
		if err = json.Unmarshal(bs, &req); err != nil {
			xl.Warnf("unmarshal face search request failed, %v", err)
			httputil.ReplyErr(env.W, http.StatusBadRequest, "parse task request failed")
			return
		}

	case ct == CT_STREAM:
		bs, err := ioutil.ReadAll(args.ReqBody)
		defer args.ReqBody.Close()
		if err != nil {
			xl.Warnf("read requests body failed. %v", err)
			httputil.ReplyErr(env.W, http.StatusBadRequest, err.Error())
			return
		}
		req.Data.URI = "data:application/octet-stream;base64," + string(base64.StdEncoding.EncodeToString(bs))

	default:
		xl.Warnf("PostFaceGroup_Search: bad content type: %s", ct)
		httputil.ReplyErr(env.W, http.StatusBadRequest, "wrong content type")
		return
	}

	s.search(ctx, groups, &req, env, "PostFaceCluster_Search")
	return
}
