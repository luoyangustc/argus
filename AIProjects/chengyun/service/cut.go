package service

import (
	"context"
	"fmt"
	"math"
	"path"
	"sort"
	"strings"
	"time"

	"github.com/pkg/errors"

	xlog "github.com/qiniu/xlog.v1"
)

type ZhatuResult struct {
	rawResp EvalZhatuResp
	// uri         string
	hasZhatuChe bool

	imagePath string

	image zhatuPartImage
}

type zhatuPart struct {
	startIndex int // 渣土车可以检测出的时间起点
	endIndex   int

	move string // 渣土车移动方向 up down all（多辆渣土车） unknow

	startOffset int // 估算的渣土车在视频里面出现的时间点
	endOffset   int

	partName string

	distance int // 这个判断距离captureTime的时间

	partIndex int

	video *checkVideoResult

	images []zhatuPartImage
}

type zhatuPartImage struct {
	uri   string
	score float32
	pts   [4]int
}

//根据抓拍照片的时间，找出附近的合适的视频片段
func (m *Manager) searchVideo(ctx context.Context, captureTime time.Time, dir string, rate int, video checkVideoResult) ([]*zhatuPart, error) {
	xl := xlog.FromContextSafe(ctx)

	var rr []*ZhatuResult
	{
		for _, image := range video.images {
			var req EvalZhatuReq
			req.Data.URI = m.FileServer + strings.TrimPrefix(image, m.Workspace)
			req.Data.Attribute.Name = path.Base(req.Data.URI)
			req.Data.Attribute.Video = true
			resp, err := m.Zhatu.Eval(ctx, req)
			if err != nil {
				return nil, errors.Wrapf(err, "req:%#v", req)
			}
			var result ZhatuResult
			result.rawResp = resp
			result.imagePath = image

			rr = append(rr, &result)
		}
	}

	// 第一次扫描，找出有渣土车的截图
	for _, v := range rr {
		if len(v.rawResp.Result.Detections) > 0 {
			v.hasZhatuChe = true
			for _, d := range v.rawResp.Result.Detections {
				if d.Score > v.image.score {
					v.image = zhatuPartImage{
						uri:   v.imagePath,
						score: d.Score,
						pts:   d.PTS,
					}
				}
			}
		}
	}

	// // 第二次扫描，允许片段有短暂间隔
	for i, v := range rr {
		if !v.hasZhatuChe {
			if i-1 < 0 || i+1 >= len(rr) {
				continue
			}
			if rr[i-1].hasZhatuChe && rr[i+1].hasZhatuChe {
				v.hasZhatuChe = true
			}
		}
	}

	// 找出所有片段
	{
		captureTimeSec := int(captureTime.Sub(video.video.Start).Seconds())
		s := make([]byte, len(rr))
		for i, v := range rr {
			if v.hasZhatuChe {
				s[i] = '1'
			} else {
				s[i] = '0'
			}
			if i == captureTimeSec {
				s[i] = '#'
			}
		}
		xl.Debugf("video info %#v %s", video.video, string(s))
	}

	parts := make([]*zhatuPart, 0)
	{
		lastEnd := -1
		for start := 0; start < len(rr); start++ {
			if start <= lastEnd {
				continue
			}
			if !rr[start].hasZhatuChe {
				continue
			}
			end := start
			for {
				end++
				if end >= len(rr) {
					break
				}
				if rr[end].hasZhatuChe {
					continue
				}
				break
			}
			end--
			lastEnd = end
			part := &zhatuPart{startIndex: start, endIndex: end, move: "all"}
			for i := start; i <= end; i++ {
				part.images = append(part.images, rr[i].image)
			}
			parts = append(parts, part)
		}
	}

	// 判断车辆移动方向
	// 如果随时间推移，检测框向上运动，则车辆向上移动，标记 "up"， 反之亦然
	// 如果同一张图片有两个检测框，并且相距比较远，则里面出现多辆渣土车，标记 "all"
	// 然后根据方向算出裁剪偏移值
	{
		for index, part := range parts {
			part.partIndex = index
			part.video = &video
			if len(rr[part.startIndex].rawResp.Result.Detections) == 0 {
				continue
			}
			if len(rr[part.endIndex].rawResp.Result.Detections) == 0 {
				continue
			}
			if rr[part.startIndex].rawResp.Result.Detections[0].PTS[1] > rr[part.endIndex].rawResp.Result.Detections[0].PTS[1] {
				part.move = "up"
			} else {
				part.move = "down"
			}
			for i := part.startIndex; i < part.endIndex; i++ {
				if len(rr[i].rawResp.Result.Detections) >= 2 {
					r := rr[i].rawResp.Result.Detections
					if math.Abs(float64(r[0].PTS[0]-r[1].PTS[0])) > 50 {
						part.move = "all"
					}
				}
			}
			part.partName = fmt.Sprintf("%d-%d.mp4", part.startIndex, part.endIndex)
			switch part.move {
			case "up":
				part.startOffset = part.startIndex - 5
				part.endOffset = part.endIndex + 15
			case "down":
				part.startOffset = part.startIndex - 15
				part.endOffset = part.endIndex + 5
			case "all":
				part.startOffset = part.startIndex - 15
				part.endOffset = part.endIndex + 15
			}

			part.distance = int(math.Abs(video.video.Start.Add(time.Second * time.Duration(part.startIndex)).Sub(captureTime).Seconds()))
			xl.Infof("part %#v", part, video.images[part.startIndex], video.images[part.endIndex])
		}
	}
	sort.Slice(parts, func(i, j int) bool {
		return parts[i].distance < parts[j].distance
	})
	return parts, nil
}
