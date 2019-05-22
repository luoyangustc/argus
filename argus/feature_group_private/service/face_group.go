package service

import (
	"context"
	"encoding/base64"
	"hash/crc32"
	"io/ioutil"
	"strings"
	"sync"
	"time"

	"github.com/pkg/errors"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/com/uri"
	"qiniu.com/argus/com/util"
	"qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/feature"
	"qiniu.com/argus/feature_group_private/proto"
)

const (
	MultiFacesModeDeny = iota
	MultiFacesModeLargest
	// TODO: more mode
)

var (
	ErrNoFaceFound      = errors.New("No face found in image")
	ErrMultiFaceFound   = errors.New("Multiple faces found in image")
	ErrBlurFace         = errors.New("detected blur face")
	ErrSmallFace        = errors.New("detected too small face")
	ErrCoverFace        = errors.New("detected covered face")
	ErrPoseFace         = errors.New("detected face with too large pose")
	ErrParseFeature     = errors.New("Failed to parse feature in image")
	ErrOrientationNotUp = errors.New("face orientation is not up")
)

type FaceGroupsConfig struct {
	BaseGroupsConfig
	Enable             bool          `json:"enable"`
	SearchCache        bool          `json:"search_cache"`
	SearchThreshold    float32       `json:"search_threshold"`
	MultiFacesMode     int           `json:"multi_faces_mode"`
	FaceFeatureHost    string        `json:"feature_host"`
	MinFaceWidth       int           `json:"min_face_width"`
	MinFaceHeight      int           `json:"min_face_height"`
	FaceFeatureTimeout time.Duration `json:"feature_timeout"`
	GroupsNumber       int           `json:"max_multisearch_groups_number"`
}

const (
	defaultMinFaceWidth  = 50
	defaultMinFaceHeight = 50
)

//------------------- FaceGroups -------------------//
var _ feature_group.IFaceGroups = new(FaceGroups)

type FaceGroups struct {
	baseGroups  *BaseGroups
	config      FaceGroupsConfig
	facefeature feature.FaceFeature
}

func NewFaceGroups(ctx context.Context, baseGroups *BaseGroups, config FaceGroupsConfig) (*FaceGroups, error) {
	s := &FaceGroups{
		baseGroups: baseGroups,
		config:     config,
		facefeature: feature.NewFaceFeature(
			config.FaceFeatureHost,
			config.FaceFeatureTimeout*time.Second,
			config.Sets.Precision*config.Sets.Dimension,
		),
	}
	if config.MinFaceWidth <= 0 {
		s.config.MinFaceWidth = defaultMinFaceWidth
	}
	if config.MinFaceHeight <= 0 {
		s.config.MinFaceHeight = defaultMinFaceHeight
	}
	return s, nil
}

func (s *FaceGroups) New(ctx context.Context, name proto.GroupName, config proto.GroupConfig) (err error) {
	return s.baseGroups.New(ctx, false, name, config)
}

func (s *FaceGroups) All(ctx context.Context) ([]proto.GroupName, error) {
	return s.baseGroups.All(ctx)
}

func (s *FaceGroups) Get(ctx context.Context, name proto.GroupName) (feature_group.IFaceGroup, error) {
	if len(name) == 0 {
		return nil, errors.New("Invalid Group Name")
	}
	group, err := s.baseGroups.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	baseGroup, ok := group.(*_BaseGroup)
	if !ok {
		return nil, errors.New("baseGroups.Get get invalid *_BaseGroup")
	}
	fs := &FaceGroup{
		baseGroup: baseGroup,
		manager:   s,
	}
	return fs, nil
}

func (fgs *FaceGroups) DetectAndFetchFeature(ctx context.Context, useQuality bool, data []proto.Data,
) (
	fvss [][]proto.FeatureValue,
	faceBoxess [][]proto.BoundingBox,
	err error,
) {
	xl := xlog.FromContextSafe(ctx)
	durations := map[string]time.Duration{"fd": 0, "ff": 0}
	defer func() {
		for index, du := range durations {
			xl.Xprof2(index, du, nil)
		}
	}()

	var faceBoxes []proto.BoundingBox
	var goerr error
	for _, image := range data {
		iURI := image.URI
		if fgs.config.SearchCache && (strings.HasPrefix(string(iURI), "http://") || strings.HasPrefix(string(iURI), "https://")) {
			now := time.Now()
			cli := uri.New(uri.WithHTTPHandler())
			resp, err := cli.Get(ctx, uri.Request{URI: string(iURI)})
			if err != nil {
				return nil, nil, err
			}
			defer resp.Body.Close()
			buf, err := ioutil.ReadAll(resp.Body)
			if err != nil || len(buf) == 0 {
				xl.Errorf("fetch %s into cache got empty body", string(iURI))
				return nil, nil, errors.New("fetch image got empty body")
			}
			iURI = proto.ImageURI("data:application/octet-stream;base64," + base64.StdEncoding.EncodeToString(buf))
			durations["if"] += time.Since(now)
		}

		if len(image.Attribute.BoundingBoxes) == 0 {
			now := time.Now()
			var newerr error
			if useQuality {
				boxes, newerr := fgs.facefeature.FaceBoxesQuality(ctx, iURI)
				if newerr == nil {
					var newFaceBoxes []proto.BoundingBox
					for _, box := range boxes {
						if box.Quality.Quality == proto.FaceQualityClear && box.Quality.Orientation == proto.FaceOrientationUp {
							newFaceBoxes = append(newFaceBoxes, box.BoundingBox)
						}
					}
					faceBoxes = newFaceBoxes
				}
			} else {
				faceBoxes, newerr = fgs.facefeature.FaceBoxes(ctx, iURI)
			}
			if newerr != nil {
				return nil, nil, errors.Wrapf(newerr, "faceFeature.FaceBoxes failed, image url : %v", image.URI)
			}
			durations["fd"] += time.Since(now)
		} else {
			faceBoxes = image.Attribute.BoundingBoxes
		}

		now := time.Now()
		fvSlice := make([]proto.FeatureValue, len(faceBoxes))
		wg := sync.WaitGroup{}
		wg.Add(len(faceBoxes))
		for serial, fb := range faceBoxes {
			go func(ctx context.Context, i int, facebox proto.BoundingBox) {
				defer wg.Done()
				fv, err := fgs.facefeature.Face(ctx, iURI, facebox.Pts)
				if err != nil {
					xlog.FromContextSafe(ctx).Errorf("faceFeature.Face failed, iamge url : %v", image.URI)
					goerr = err
					return
				} else {
					fvSlice[i] = fv
				}
			}(util.SpawnContext(ctx), serial, fb)
		}
		if goerr != nil {
			break
		}
		wg.Wait()
		durations["ff"] += time.Since(now)

		fvss = append(fvss, fvSlice)
		faceBoxess = append(faceBoxess, faceBoxes)
	}
	if goerr != nil {
		return nil, nil, goerr
	}
	return fvss, faceBoxess, nil
}

//------------------- FaceGroup -------------------//
var _ feature_group.IFaceGroup = new(FaceGroup)

type FaceGroup struct {
	baseGroup *_BaseGroup
	manager   *FaceGroups
}

func (s *FaceGroup) parseImageFeatures(ctx context.Context, reject bool, faceThresholdSize [2]int, images ...proto.Image) (features []proto.Feature, faceNum int, err error) {
	xl := xlog.FromContextSafe(ctx)
	for _, i := range images {
		var validFacesCommon proto.FaceDetectBox
		if len(i.BoundingBox.Pts) == 0 {
			faceBoxes, err := s.manager.facefeature.FaceBoxesQuality(ctx, i.URI)
			if err != nil {
				xl.Errorf("Parse faces error: %s", err)
				return nil, 0, err
			}
			// 过滤小脸
			validFaces := make([]proto.FaceDetectBox, 0)
			for _, fb := range faceBoxes {
				// 判断小脸 faceThresholdSize, 小于这个值的忽略掉
				pts := fb.BoundingBox.Pts
				if pts[0][0]+faceThresholdSize[0] < pts[2][0] && pts[0][1]+faceThresholdSize[1] < pts[2][1] {
					validFaces = append(validFaces, fb)
				}
			}

			if len(validFaces) == 0 {
				xl.Errorf("No face found in image %v", i.ID)
				return nil, 0, ErrNoFaceFound
			}
			faceNum = len(validFaces)
			if len(validFaces) != 1 {
				switch s.manager.config.MultiFacesMode {
				case MultiFacesModeDeny:
					xl.Errorf("Multiple faces found in image %v", i.ID)
					return nil, faceNum, ErrMultiFaceFound
				case MultiFacesModeLargest:
					// select the largest face
					var index, maxFace int
					for i, fb := range validFaces {
						if maxFace < (fb.BoundingBox.Pts[1][0]-fb.BoundingBox.Pts[0][0])*(fb.BoundingBox.Pts[2][1]-fb.BoundingBox.Pts[1][1]) {
							maxFace = (fb.BoundingBox.Pts[1][0] - fb.BoundingBox.Pts[0][0]) * (fb.BoundingBox.Pts[2][1] - fb.BoundingBox.Pts[1][1])
							index = i
						}
					}
					validFaces = validFaces[index : index+1]
				}
			}
			validFacesCommon = validFaces[0]

			// check reject_bad_face
			if reject {
				switch validFacesCommon.Quality.Quality {
				case proto.FaceQualityBlur:
					return nil, faceNum, ErrBlurFace
				case proto.FaceQualitySmall:
					return nil, faceNum, ErrSmallFace
				case proto.FaceQualityCover:
					return nil, faceNum, ErrCoverFace
				case proto.FaceQualityPose:
					return nil, faceNum, ErrPoseFace
				}
				switch validFacesCommon.Quality.Orientation {
				case proto.FaceOrientationUp:
				default:
					return nil, faceNum, ErrOrientationNotUp
				}
			}

		} else {
			validFacesCommon.BoundingBox.Pts = i.BoundingBox.Pts
		}
		faceFeatureValue, err := s.manager.facefeature.Face(ctx, i.URI, validFacesCommon.BoundingBox.Pts)
		if err != nil {
			xl.Errorf("Failed to parse feature in image %v, err: %s", i.ID, err)
			return nil, faceNum, ErrParseFeature
		}
		features = append(features, proto.Feature{
			ID:          i.ID,
			Value:       faceFeatureValue,
			Tag:         i.Tag,
			Desc:        i.Desc,
			HashKey:     proto.FeatureHashKey(crc32.ChecksumIEEE([]byte(i.ID))),
			BoundingBox: validFacesCommon.BoundingBox,
			FaceQuality: validFacesCommon.Quality,
		})
	}
	return
}

func (s *FaceGroup) AddFace(ctx context.Context, reject bool, images ...proto.Image) ([]proto.FaceDetectBox, int, error) {
	xl := xlog.FromContextSafe(ctx)

	features, faceNum, err := s.parseImageFeatures(ctx, reject, [2]int{s.manager.config.MinFaceWidth, s.manager.config.MinFaceHeight}, images...)
	if err != nil {
		xl.Errorf("Group.Add parseImageFeatures error: %s", err)
		return nil, faceNum, err
	}

	err = s.baseGroup.Add(ctx, false, features...)
	if err != nil {
		xl.Errorf("Group.Add error: %s", err)
		return nil, faceNum, err
	}
	var boxes []proto.FaceDetectBox
	for _, feature := range features {
		boxes = append(boxes, proto.FaceDetectBox{
			BoundingBox: feature.BoundingBox,
			Quality:     feature.FaceQuality,
		})
	}
	return boxes, faceNum, nil
}

func (s *FaceGroup) UpdateFace(ctx context.Context, reject bool, images ...proto.Image) error {
	xl := xlog.FromContextSafe(ctx)

	features, _, err := s.parseImageFeatures(ctx, reject, [2]int{s.manager.config.MinFaceWidth, s.manager.config.MinFaceHeight}, images...)
	if err != nil {
		xl.Errorf("Group.Update parseImageFeatures error: %s", err)
		return err
	}

	err = s.baseGroup.Update(ctx, false, features...)
	if err != nil {
		xl.Errorf("Group.Update error: %s", err)
		return err
	}
	return nil
}

func (s *FaceGroup) SearchFace(ctx context.Context,
	threshold float32, limit int, fvs [][]proto.FeatureValue,
) (
	[][]feature_group.FaceSearchRespItem,
	error,
) {
	xl := xlog.FromContextSafe(ctx)
	durations := map[string]time.Duration{"fs": 0}
	rets := make([][]feature_group.FaceSearchRespItem, len(fvs))
	defer func() {
		for index, du := range durations {
			xl.Xprof2(index, du, nil)
		}
	}()

	matchSlice := make([]int, 0)
	multiFaces := make([]proto.FeatureValue, 0)
	for i, pic := range fvs {
		for _, face := range pic {
			multiFaces = append(multiFaces, face)
			matchSlice = append(matchSlice, i)
		}
	}

	now := time.Now() //多图搜索时间
	if threshold == 0 {
		threshold = s.manager.config.SearchThreshold
	}
	searchResult, err := s.baseGroup.Search(ctx, false, threshold, limit, multiFaces...)
	if err != nil {
		xl.Errorf("Group.Search error: %s", err)
		return nil, err
	}
	durations["fs"] += time.Since(now)

	for index, row := range searchResult { //searchResult[n]:第n张脸，searchResult[n][]:与底库匹配中的多个结果
		face := feature_group.FaceSearchRespItem{} //BoundingBox在外层face_service做此操作
		for _, feature := range row {
			face.Faces = append(face.Faces,
				feature_group.FaceSearchRespFaceItem{
					ID:          feature.Value.ID,
					Score:       feature.Score,
					Tag:         feature.Value.Tag,
					Desc:        feature.Value.Desc,
					Group:       feature.Value.Group,
					BoundingBox: feature.Value.BoundingBox,
				})
		}
		rets[matchSlice[index]] = append(rets[matchSlice[index]], face)
	}

	return rets, nil
}

func (s *FaceGroup) ListImage(ctx context.Context,
	tag proto.FeatureTag, marker string, limit int,
) (
	images []feature_group.ImageListRespItem,
	nextMarker string,
	err error,
) {
	xl := xlog.FromContextSafe(ctx)
	if limit == 0 {
		limit = defaultSearchLimit
	}
	if marker != "" {
		marker, err = decodeMarker(marker)
		if err != nil {
			return nil, "", errors.New("Invalid marker")
		}
	}

	features, nextMarker, err := s.baseGroup.FilterByTag(ctx, tag, marker, limit)
	if err != nil {
		xl.Errorf("Group.ListImage error: %s", err)
		return nil, "", err
	}
	images = make([]feature_group.ImageListRespItem, 0)
	for _, feature := range features {
		images = append(images, feature_group.ImageListRespItem{
			ID:          feature.ID,
			Tag:         feature.Tag,
			Desc:        feature.Desc,
			BoundingBox: feature.BoundingBox,
		})
	}

	return images, encodeMarker(nextMarker), nil
}

func (s FaceGroup) AddFeature(ctx context.Context, features ...proto.Feature) (err error) {
	xl := xlog.FromContextSafe(ctx)
	err = s.baseGroup.Add(ctx, false, features...)
	if err != nil {
		xl.Errorf("Group.ClusterAddFace error: %s", err)
		return err
	}
	return nil
}
