package facec

import (
	"context"
	"encoding/base64"
	"io/ioutil"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/argus/facec/client"
	"qiniu.com/argus/argus/facec/db"
	"qiniu.com/argus/argus/facec/imgprocess"
)

type FeatureWorkerConfig struct {
	URL        string        `json:"url"`
	Period     time.Duration `json:"period"`
	Concurrent int           `json:"concurrent"`
	BatchSize  int           `json:"batch_size"`
}

type FeatureWorker struct {
	config  *FeatureWorkerConfig
	stopped int32
	wg      sync.WaitGroup
}

func NewFeatureWorker(cfg *FeatureWorkerConfig) *FeatureWorker {
	w := &FeatureWorker{
		config: cfg,
	}
	w.run()
	return w
}

func (r *FeatureWorker) run() {
	for i := 0; i < r.config.Concurrent; i++ {
		r.wg.Add(1)
		go func() {
			taskCh := make(chan []db.FeatureTask)
			defer func() {
				r.wg.Done()
			}()

			go func() {
				dao, _ := db.NewFeatureTaskDao()
				for atomic.LoadInt32(&r.stopped) == 0 {
					tasks, _ := dao.FindTasks(context.Background(), r.config.BatchSize)
					if len(tasks) == 0 {
						time.Sleep(time.Second)
						continue
					}
					taskCh <- tasks
				}
				close(taskCh)
			}()

			for tasks := range taskCh {

				var (
					xl  = xlog.NewDummy()
					ctx = xlog.NewContext(context.Background(), xl)
				)

				buildFeature2(ctx, tasks, r.config.URL)

				fdao, _ := db.NewFeatureTaskDao()
				cdao, _ := db.NewClusterTaskDao()
				ids := make([]bson.ObjectId, 0, len(tasks))
				for _, task := range tasks {
					ids = append(ids, task.ID)
					_ = cdao.UpsertTask(ctx,
						db.ClusterTask{
							UID:       task.UID,
							Euid:      task.Euid,
							CreatedAt: task.CreatedAt,
						},
					)
				}
				_ = fdao.Remove(ctx, ids...)
			}
		}()
	}
}

func (r *FeatureWorker) Stop() {
	atomic.StoreInt32(&r.stopped, 1)
	r.wg.Wait()
}

var TRIMFACE bool = true

func buildFeature2(ctx context.Context, tasks []db.FeatureTask, featureAPI string) error {

	xl := xlog.FromContextSafe(ctx)

	xl.Infof("build feature: %d", len(tasks))

	getFeatures := func(ctx context.Context, urls []string, pts [][][]int64, uids []uint32) ([]string, error) {
		type fQueueElem struct {
			Index int
			Req   client.FacexFeatureReq
		}
		var (
			xl       = xlog.FromContextSafe(ctx)
			procNum  = 20
			fQueue   = make(chan fQueueElem, procNum)
			fwaiter  = sync.WaitGroup{}
			fMap     = make(map[int]string)
			mutex    = sync.Mutex{}
			features = make([]string, 0)
		)
		go func() {
			for i, img := range urls {
				var fReq client.FacexFeatureReq
				fReq.Data.URI = img
				fReq.Data.Attribute.Pts = pts[i]
				fQueue <- fQueueElem{
					Index: i,
					Req:   fReq,
				}
			}
			close(fQueue)
		}()

		fwaiter.Add(procNum)
		for i := 0; i < procNum; i++ {
			go func() {
				defer fwaiter.Done()

				ctex := xlog.NewContext(ctx, xlog.NewDummy())
				for {
					r, ok := <-fQueue
					if !ok {
						break
					}
					cli := client.NewRPCClient(client.EvalEnv{Uid: uids[r.Index]}, time.Second*120)
					rep, err := cli.DoRequestWithJson(ctex, "POST", featureAPI, r.Req)
					if err != nil || rep != nil && rep.StatusCode/100 != 2 {
						xl.Error("request feature error:%v,resp:%v,url:%v", err, rep, featureAPI)
						mutex.Lock()
						fMap[r.Index] = ""
						mutex.Unlock()
						continue
					}
					ret, err := ioutil.ReadAll(rep.Body)
					defer rep.Body.Close()
					if err != nil || len(ret) == 0 {
						xl.Error("read body error: %v,resp:%v", err, len(ret))
						mutex.Lock()
						fMap[r.Index] = ""
						mutex.Unlock()
						continue
					}
					mutex.Lock()
					fMap[r.Index] = base64.StdEncoding.EncodeToString(ret)
					mutex.Unlock()
				}
			}()
		}
		fwaiter.Wait()

		for i, _ := range urls {
			features = append(features, fMap[i])
		}
		return features, nil
	}

	var (
		urls    []string    = make([]string, 0, len(tasks))
		pts     [][][]int64 = make([][][]int64, 0, len(tasks))
		uids    []uint32    = make([]uint32, 0, len(tasks))
		indexes []int       = make([]int, 0, len(tasks))
	)

	if TRIMFACE {
		images := make([]imgprocess.Image, 0, len(tasks))
		for _, task := range tasks {
			images = append(images, imgprocess.NewTrimrectImage(task.Face.File, task.Face.Pts.Det))
		}
		imageProc := imgprocess.New(images, newImageProcessClient())
		imageProc.FetchAll(ctx)
		for i, task := range tasks {
			if _, ok := images[i].OK(); ok {
				urls = append(urls, images[i].NewUrl())
				_, _pts := images[i].Zoom(task.Face.Pts.Det)
				pts = append(pts, _pts)
				ud, _ := strconv.ParseUint(task.UID, 10, 32)
				uids = append(uids, uint32(ud))
				indexes = append(indexes, i)
			}
		}
	} else {
		for i, task := range tasks {
			urls = append(urls, task.Face.File)
			pts = append(pts, task.Face.Pts.Det)
			indexes = append(indexes, i)
		}
	}
	features, err := getFeatures(ctx, urls, pts, uids)
	if err != nil {
		xl.Error("get feature error", err)
		return err
	}

	var toUpdatedFaces []db.Face = make([]db.Face, 0, len(features))
	for i, feature := range features {
		if feature == "" {
			continue
		}
		task := tasks[indexes[i]]
		toUpdatedFaces = append(
			toUpdatedFaces,
			db.Face{
				ID:   task.Face.ID,
				UID:  task.UID,
				Euid: task.Euid,
				File: task.Face.File,
				Pts: db.FacePts{
					Det:    task.Face.Pts.Det,
					ModelV: task.Face.Pts.ModelV,
				},
				Feature: db.FaceFeature{
					Feature: feature,
					ModelV:  "v1", // TODO
				},
			},
		)
	}
	faceDao, err := db.NewFaceDao()
	if err != nil {
		xl.Error("new face dao error", err)
		return err
	}
	if _, err = faceDao.UpdateFeatures(context.Background(), toUpdatedFaces); err != nil {
		xl.Error("update features error", err)
		return err
	}
	return nil
}

func flatFaceIDs(file2FaceIDs map[string][]bson.ObjectId) []bson.ObjectId {
	faceCount := 0
	for _, ids := range file2FaceIDs {
		faceCount += len(ids)
	}

	// BAD `bson.ObjectId`
	faceIDs := make([]bson.ObjectId, 0, faceCount)
	for _, ids := range file2FaceIDs {
		faceIDs = append(faceIDs, ids...)
	}

	return faceIDs
}

func newImageProcessClient() *http.Client { return &http.Client{Timeout: 30 * time.Second} }
